# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from .Modules.Networks import Networks 
from .Modules.Optimizer import WarpedAdam
from .Common import voxelize_sample, adaptive_voxel_size, RenderResults, calculate_morton_order,compress_gpcc, decompress_gpcc
from functools import reduce
from torch_scatter import scatter_max
from einops import repeat
import time
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self.xyz = torch.empty(0)
        self.deformation = deform_network(W=args.net_width, D=args.defor_depth, 
                                           min_embeddings=args.min_embeddings, max_embeddings=args.max_embeddings, 
                                           num_frames=args.total_num_frames,
                                           args=args)

        self.scaling = torch.empty(0)
        self.rotation = torch.empty(0)
        self.ref_embedding = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.aux_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.densify_until_iter = args.voxelize_iter
        self.use_significance_weight = args.use_significance_weight
        #coupled GS
        self.res_embedding = torch.empty(0) # residual features of coupled primitives, shape (N, K, res_feats_dim)
        #network  
        ref_embedding_dim, res_embedding_dim, derive_factor = args.ref_embedding_dim, args.res_embedding_dim, args.derive_factor
        ref_hyper_dim = args.ref_hyper_dim
        res_hyper_dim = args.res_hyper_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.position_embedding_dim = args.position_embedding_dim

        self.ref_embedding_dim = ref_embedding_dim
        self.res_embedding_dim = res_embedding_dim
        self.derive_factor = derive_factor #args.derive_factor
        self.network = Networks(
            ref_feats_dim=self.ref_embedding_dim, ref_hyper_dim=ref_hyper_dim,
            res_feats_dim=self.res_embedding_dim, res_hyper_dim=res_hyper_dim,
            derive_factor=self.derive_factor).to(self.device)
        
        # create auxiliary variables
        self.accumulated_opacities = torch.empty(0)  # accumulated opacities of preDicted Gaussian primitives, shape (N, 1)
        self.anchor_denorm = torch.empty(0)  # times of anchor primitives accessed in rendering, shape (N, 1)
        self.accumulated_grads = torch.empty(0)  # accumulated gradients of preDicted Gaussian primitives, shape (N * K, 1)
        self.coupled_denorm = torch.empty(0)  # times of coupled primitives accessed in rendering, shape (N * K, 1)

        self.voxel_size_init = args.voxel_size_init
        self.voxel_size_compress = args.voxel_size_compress

        
    def capture(self):
        return (
            self.active_sh_degree,
            self.xyz,
            self.deformation.state_dict(),
            self.network.state_dict(),
            self.scaling,
            self.rotation,
            self.ref_embedding,
            self.res_embedding,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.aux_optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self.xyz, 
        self.deformation,
        self.network,
        self.ref_embedding,
        self.res_embedding,
        self.scaling, 
        self.rotation, 
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.aux_optimizer.load_state_dict(opt_dict)
    @property
    def get_scaling_exp(self):
        return self.scaling_activation(self.scaling)
    
    @property
    def get_scaling_before_exp(self):
        return self.scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)
    
    @property
    def get_xyz(self):
        return self.xyz
    
    @property
    @torch.no_grad()
    def get_anchor_scales(self) -> torch.Tensor:
        quant_scaling_factor = self.network.quantize_scales(scaling_factors_before_exp=self.scaling, ref_feats=self.ref_embedding)
        return torch.exp(quant_scaling_factor[:, :3])
    
    @property
    def get_ref_embedding(self):
        return self.ref_embedding
    #add res embedding
    @property
    def get_res_embedding(self):
        return self.res_embedding
    
    @property
    def get_cat_embedding(self):
        ref_embedding = repeat(self.ref_embedding, 'n c -> (n k) c', k=self.derive_factor)
        res_embedding = self.res_embedding.reshape(-1, self.res_embedding.shape[-1])

        return torch.cat([ref_embedding, res_embedding], dim=-1)
    @property
    def aux_loss(self) -> torch.Tensor:
        return self.network.aux_loss()
    
    @property
    def num_coupled_primitive(self):
        return self.xyz.shape[0] * self.derive_factor

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    #add ref res embedding

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        #Voxelization
        print("pcd.points", adaptive_voxel_size(pcd.points))
        fused_point_cloud = voxelize_sample(pcd.points, voxel_size= self.voxel_size_init)
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(fused_point_cloud)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        scales = torch.clamp(scales, max=1.0)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        ref_embedding = torch.zeros((fused_point_cloud.shape[0], self.ref_embedding_dim)).float().cuda()  # [jm]
        res_embedding = torch.zeros(fused_point_cloud.shape[0], self.derive_factor, self.res_embedding_dim).float().cuda()
        
        self.xyz = nn.Parameter(torch.tensor(np.asarray(fused_point_cloud)).float().cuda().requires_grad_(True))
        self.deformation = self.deformation.to("cuda") 
        self.network = self.network.to("cuda")
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.ref_embedding = nn.Parameter(ref_embedding.requires_grad_(True))  # [jm]
        self.res_embedding = nn.Parameter(res_embedding.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]*self.derive_factor), device="cuda")

        self.accumulated_grads = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.coupled_denorm = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.accumulated_opacities = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.anchor_denorm = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]*self.derive_factor, 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]*self.derive_factor, 1), device="cuda")
        #prediction network
        prediction_net = self.network.prediction_net
        means_pred_mlp = prediction_net.means_pred_mlp
        covariance_pred_mlp = prediction_net.covariance_pred_mlp
        opacity_pred_mlp = prediction_net.opacity_pred_mlp
        color_pred_mlp = prediction_net.color_pred_mlp
        
        entropy_model = self.network.entropy_model

        print("self.spatial_lr_scale:",self.spatial_lr_scale)
        
        l = [
            {'params': [self.xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self.deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self.deformation.offsets], 'lr': training_args.offsets_lr, "name": "offsets"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr_init, "name": "scaling"},
            {'params': [self.rotation], 'lr': training_args.rotation_lr_init, "name": "rotation"},
            {'params': [self.ref_embedding], 'lr': training_args.ref_embedding_lr_init, "name": "ref_embedding"},
            {'params': [self.res_embedding], 'lr': training_args.res_embedding_lr_init, "name": "res_embedding"},
            {'params': means_pred_mlp.parameters(), 'lr': training_args.means_pred_mlp_lr_init, "name": "means_pred_mlp"},
            {'params': covariance_pred_mlp.parameters(), 'lr': training_args.covariance_pred_mlp_lr_init, "name": "covariance_pred_mlp"},
            {'params': opacity_pred_mlp.parameters(), 'lr': training_args.opacity_pred_mlp_lr_init, "name": "opacity_pred_mlp"},
            {'params': color_pred_mlp.parameters(), 'lr': training_args.color_pred_mlp_lr_init, "name": "color_pred_mlp"},
            {'params': entropy_model.parameters(), 'lr': training_args.ref_entropy_model_lr_init, "name": "entropy_model"}]

        self.optimizer = WarpedAdam(l, lr=0.0, eps=1e-15)
        self.aux_optimizer = WarpedAdam(self.network.get_lr_aux_param_pairs(), lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init,
                                                    lr_final=training_args.scaling_lr_final,
                                                    lr_delay_mult=training_args.scaling_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init,
                                                    lr_final=training_args.rotation_lr_final,
                                                    lr_delay_mult=training_args.rotation_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.ref_embedding_scheduler_args = get_expon_lr_func(lr_init=training_args.ref_embedding_lr_init,
                                                    lr_final=training_args.ref_embedding_lr_final,
                                                    lr_delay_mult=training_args.ref_embedding_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.res_embedding_scheduler_args = get_expon_lr_func(lr_init=training_args.res_embedding_lr_init,
                                                    lr_final=training_args.res_embedding_lr_final,
                                                    lr_delay_mult=training_args.res_embedding_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.deformation_lr_max_steps)    
        self.means_pred_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.means_pred_mlp_lr_init,
                                                    lr_final=training_args.means_pred_mlp_lr_final,
                                                    lr_delay_mult=training_args.means_pred_mlp_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.covariance_pred_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.covariance_pred_mlp_lr_init,
                                                    lr_final=training_args.covariance_pred_mlp_lr_final,
                                                    lr_delay_mult=training_args.covariance_pred_mlp_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.opacity_pred_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_pred_mlp_lr_init,
                                                    lr_final=training_args.opacity_pred_mlp_lr_final,
                                                    lr_delay_mult=training_args.opacity_pred_mlp_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.color_pred_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.color_pred_mlp_lr_init,
                                                    lr_final=training_args.color_pred_mlp_lr_final,
                                                    lr_delay_mult=training_args.color_pred_mlp_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
        self.entropy_model_scheduler_args = get_expon_lr_func(lr_init=training_args.ref_entropy_model_lr_init,
                                                    lr_final=training_args.ref_entropy_model_lr_final,
                                                    lr_delay_mult=training_args.ref_entropy_model_lr_delay_mult,
                                                    max_steps=training_args.network_lr_max_steps)
      
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if iteration >= self.densify_until_iter:
                    lr = 0.0  # 在 20000 迭代后，学习率为 0
                else:
                    lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr  
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "means_pred_mlp":
                lr = self.means_pred_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "covariance_pred_mlp":
                lr = self.covariance_pred_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "opacity_pred_mlp":
                lr = self.opacity_pred_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "color_pred_mlp":
                lr = self.color_pred_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "ref_embedding":
                lr = self.ref_embedding_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "res_embedding":
                lr = self.res_embedding_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "entropy_model":
                lr = self.entropy_model_scheduler_args(iteration)
                param_group['lr'] = lr
                              
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.ref_embedding.shape[1]):
            l.append('ref_embedding_{}'.format(i))
        for i in range(self.res_embedding.shape[1]*self.res_embedding.shape[2]):
            l.append('res_embedding_{}'.format(i))   
        return l

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self.deformation.load_state_dict(weight_dict)
        self.deformation = self.deformation.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.network.load_state_dict(torch.load(os.path.join(path,"network.pth"),map_location="cuda"))
        self.network = self.network.to("cuda")
    
    def load_prediction_net(self, path):
        self.network.load_state_dict(torch.load(os.path.join(path,"network.pth"),map_location="cuda"))
        self.network = self.network.to("cuda")
        print("加载与训练model:", os.path.join(path,"network.pth"))

    def save_deformation(self, path):
        torch.save(self.deformation.state_dict(),os.path.join(path, "deformation.pth"))
        self.network.update()
        torch.save(self.network.state_dict(),os.path.join(path, "network.pth"))
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()
        ref_embedding = self.ref_embedding.detach().cpu().numpy()
        res_embedding = self.res_embedding.clone().detach().flatten(start_dim=1).cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, scale, rotation, ref_embedding, res_embedding), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ref_embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ref_embedding")]
        ref_embedding_names = sorted(ref_embedding_names, key = lambda x: int(x.split('_')[-1]))
        ref_embeddings = np.zeros((xyz.shape[0], len(ref_embedding_names)))
        for idx, attr_name in enumerate(ref_embedding_names):
            ref_embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])

        res_embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("res_embedding")]
        res_embedding_names = sorted(res_embedding_names, key = lambda x: int(x.split('_')[-1]))
        res_embeddings = np.zeros((xyz.shape[0], len(res_embedding_names)))
        for idx, attr_name in enumerate(res_embedding_names):
            res_embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])
        res_embeddings = res_embeddings.reshape(xyz.shape[0], self.derive_factor, self.res_embedding_dim)

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.ref_embedding = nn.Parameter(torch.tensor(ref_embeddings, dtype=torch.float, device="cuda").requires_grad_(True))
        self.res_embedding = nn.Parameter(torch.tensor(res_embeddings, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
    def set_prediction_requires_grad(self, Flag = True):
        self.xyz.requires_grad = Flag
        self.scaling.requires_grad = Flag
        self.rotation.requires_grad = Flag
        self.ref_embedding.requires_grad = Flag
        self.res_embedding.requires_grad = Flag
        prediction_net = self.network.prediction_net
        means_pred_mlp = prediction_net.means_pred_mlp
        covariance_pred_mlp = prediction_net.covariance_pred_mlp
        opacity_pred_mlp = prediction_net.opacity_pred_mlp
        color_pred_mlp = prediction_net.color_pred_mlp
        for param in means_pred_mlp.parameters():
            param.requires_grad = Flag

        for param in covariance_pred_mlp.parameters():
            param.requires_grad = Flag

        for param in opacity_pred_mlp.parameters():
            param.requires_grad = Flag

        for param in color_pred_mlp.parameters():
            param.requires_grad = Flag

    def print_deformation_weight_grad(self):
        for name, weight in self.deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)

    @property
    def position_embedding(self):
        frequencies = torch.arange(0, self.position_embedding_dim // (2 * 3), device=self.xyz.device)  # 保证 frequencies 在同一设备上
        div_term = 10000 ** (2 * frequencies / self.position_embedding_dim)
        pos_encodings = []
        for i in range(3):
            pos = self.xyz[:, i].unsqueeze(1)
            div_term = div_term.to(pos.device)  
            sin_enc = torch.sin(pos / div_term)
            cos_enc = torch.cos(pos / div_term)
            pos_encodings.append(sin_enc)
            pos_encodings.append(cos_enc)

        return torch.cat(pos_encodings, dim=-1)
 
    @property
    @torch.no_grad()
    def pred_gaussian_means(self) -> torch.Tensor:
        """
        Return predicted means of Gaussian primitives with shape (N * K, 3), used in adaptive control.
        """
        return self.network.pred_gaussian_means(
            means=self.xyz, scaling_factors_before_exp=self.scaling,
            ref_feats=self.ref_embedding, res_feats=self.res_embedding)

    @property
    @torch.no_grad()
    def pred_gaussian_scales(self) -> torch.Tensor:
        """
        Return predicted means of Gaussian primitives with shape (N * K, 3), used in adaptive control.
        """
        return self.network.pred_gaussian_scales(
            means=self.xyz, scaling_factors_before_exp=self.scaling,
            ref_feats=self.ref_embedding, res_feats=self.res_embedding)

    @property
    @torch.no_grad()
    def pred_gaussian_opacities(self) -> torch.Tensor:
        """
        Return predicted means of Gaussian primitives with shape (N * K, 3), used in adaptive control.
        """
        return self.network.pred_gaussian_opacities(
            means=self.xyz, scaling_factors_before_exp=self.scaling,
            ref_feats=self.ref_embedding, res_feats=self.res_embedding)
    
    @property
    def num_anchor_primitive(self):
        return self.xyz.shape[0]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "offsets":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.ref_embedding = optimizable_tensors["ref_embedding"]
        self.res_embedding = optimizable_tensors["res_embedding"]
        valid_points_mask = repeat(valid_points_mask,'n -> (n k)',k=self.derive_factor)
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1 or group["name"] == "offsets":continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scaling, new_rotation, new_ref_embedding, new_res_embedding):
        d = {"xyz": new_xyz,  
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "ref_embedding": new_ref_embedding,
        "res_embedding": new_res_embedding,
       }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.ref_embedding = optimizable_tensors["ref_embedding"]
        self.res_embedding = optimizable_tensors["res_embedding"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]*self.derive_factor, 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]*self.derive_factor, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]*self.derive_factor), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        coupled_xyz = self.pred_gaussian_means
        new_xyz = coupled_xyz[selected_pts_mask]


        scaling = repeat(self.scaling, 'n c -> (n k) c', k=self.derive_factor)
        rotation = repeat(self.rotation, 'n c -> (n k) c', k=self.derive_factor)
        ref_embedding = repeat(self.ref_embedding, 'n c -> (n k) c', k=self.derive_factor)
        res_embedding = repeat(self.res_embedding, 'n g c -> (n k) g c', k=self.derive_factor)

        new_scaling = scaling[selected_pts_mask]
        new_rotation = rotation[selected_pts_mask]
        new_ref_embedding = ref_embedding[selected_pts_mask]
        new_res_embedding = res_embedding[selected_pts_mask]
        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_ref_embedding, new_res_embedding)

    def prune(self, max_grad, min_opacity, extent, max_screen_size, use_mean=False):
        opacity = self.pred_gaussian_opacities
        if use_mean:
            prune_mask = (opacity < (opacity.max() - (opacity.max() - opacity.min())*0.5) ).squeeze()        
        else:
            prune_mask = (opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        k = self.derive_factor  # Group size

        grouped_prune_mask = prune_mask.view(-1, k).all(dim=1)  # Group and check if all are 1s

        self.prune_points(grouped_prune_mask)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if self.use_significance_weight:
            opacity = self.pred_gaussian_opacities
            scales = self.pred_gaussian_scales

            volumes = torch.prod(scales, dim=1)
            volumes = torch.abs(volumes)  # 取绝对值
            V_max90 = torch.quantile(volumes, 0.8)
            V_norm = torch.min(torch.max(volumes / V_max90, torch.tensor(0.0)), torch.tensor(1.0))
            volume_powers = V_norm ** 0.01

            significance = opacity * volume_powers
            significance_weight = significance_weight[update_filter]
        else:
            significance_weight = 1
        self.xyz_gradient_accum[update_filter] +=  torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True) * significance_weight
        self.denom[update_filter] += significance_weight

    def compress_gaussians(self, path, npz_path, gpcc_codec_path):
        mkdir_p(path)
        
        torch.save(self.deformation.state_dict(),os.path.join(path, "deformation.pth"))
        self.network.update()
        torch.save(self.network.state_dict(),os.path.join(path, "network.pth"))

        means, scaling_factors_before_exp = self.xyz, self.scaling
        ref_feats, res_feats = self.ref_embedding, self.res_embedding

        # voxelize means
        grid_size = self.voxel_size_compress
        means = torch.round(means / grid_size)

        # sorted all parameters by the Morton order of means
        sorted_indices = calculate_morton_order(means)
        means, scaling_factors_before_exp = means[sorted_indices], scaling_factors_before_exp[sorted_indices]
        ref_feats, res_feats = ref_feats[sorted_indices], res_feats[sorted_indices]

        # compress features
        strings = self.network.compress(ref_feats=ref_feats, res_feats=res_feats, scaling_factors_before_exp=scaling_factors_before_exp)

        # compress means by G-PCC
        means_strings = compress_gpcc(means, gpcc_codec_path=gpcc_codec_path)

        strings['means_strings'] = means_strings

        # save to bin file
        np.savez_compressed(npz_path, voxel_size=grid_size, **strings)

        # collect bitstream size in MB
        size = {f'{key}_size': len(value) / 1024 / 1024 for key, value in strings.items()}

        size['deform_net_size'] = os.path.getsize(os.path.join(path, "deformation.pth")) / 1024 / 1024
        size['predict_net_size'] = os.path.getsize(os.path.join(path, "network.pth")) / 1024 / 1024

        size['total'] = os.path.getsize(npz_path)  / 1024 / 1024 + size['deform_net_size'] + size['predict_net_size']
        out_txt_path = os.path.join(path, 'size.txt')
        with open(out_txt_path, 'w') as f:
            for key, value in size.items():
                f.write(f'{key}: {value:.2f} MB\n')
        print('size:', size)
    
    def decompress_gaussians(self, path, npz_path, gpcc_codec_path):
        start_time = time.time()
        
        self.deformation.load_state_dict(torch.load(os.path.join(path, "deformation.pth")))
        self.deformation = self.deformation.to(self.device)
        self.network.load_state_dict(torch.load(os.path.join(path, "network.pth")))
        self.network = self.network.to(self.device)

        data_dict = np.load(npz_path)
        grid_size = float(data_dict['voxel_size'])

        means_strings = data_dict['means_strings'].tobytes()
        scale_strings = data_dict['scale_strings'].tobytes()
        ref_feats_strings = data_dict['ref_feats_strings'].tobytes()
        ref_hyper_strings = data_dict['ref_hyper_strings'].tobytes()
        res_feats_pre_strings = data_dict['res_feats_pre_strings'].tobytes()
        res_feats_dfm_strings = data_dict['res_feats_dfm_strings'].tobytes()
        res_hyper_strings = data_dict['res_hyper_strings'].tobytes()

        # decompress means by G-PCC
        means = decompress_gpcc(means_strings, gpcc_codec_path=gpcc_codec_path).to(self.device)
        sorted_indices = calculate_morton_order(means)

        # decompress features
        ref_feats, res_feats, scaling_factors_before_exp = self.network.decompress(
            ref_feats_strings=ref_feats_strings, ref_hyper_strings=ref_hyper_strings,
            res_feats_pre_strings=res_feats_pre_strings, res_hyper_strings=res_hyper_strings, res_feats_dfm_strings=res_feats_dfm_strings,
            scale_strings=scale_strings, num_anchor_primitives=means.shape[0])
        # sorted means by the Morton order
        means = means[sorted_indices]

        # devoxelize means
        means = means * grid_size

        # create rotation quaternions
        rotations = torch.zeros((means.shape[0], 4), dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.

        self.xyz = nn.Parameter(torch.tensor(means, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scaling_factors_before_exp, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rotations, dtype=torch.float, device="cuda").requires_grad_(True))
        self.ref_embedding = nn.Parameter(torch.tensor(ref_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self.res_embedding = nn.Parameter(torch.tensor(res_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        
        end_time = time.time()
        print('decompress time:', end_time - start_time)
    def voxelize(self):
        grid_size = self.voxel_size_compress

        self.xyz = torch.round(self.xyz / grid_size) * grid_size  # 更新到网格中心
        # 清理缓存
        torch.cuda.empty_cache()


