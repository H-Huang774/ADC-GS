from typing import Iterator

import torch
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F, Parameter

from .BasicBlocks import BatchEntropyBottleneck, BatchGaussianConditional, ResidualMLP
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
class EntropyModel(CompressionModel):
    def __init__(self, ref_feats_dim: int, ref_hyper_dim: int, res_feats_dim: int, res_hyper_dim: int, derive_factor: int, scale_dim:int = 6, quant_step: float = 1) -> None:
        super().__init__()
        self.h_a_ref = nn.Sequential(
            nn.Linear(in_features=ref_feats_dim, out_features=ref_hyper_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=ref_hyper_dim, out_features=ref_hyper_dim)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quant_step_ref = nn.Parameter(torch.ones(ref_feats_dim, dtype=torch.float32, device=device) * quant_step * 0.1, requires_grad=True)
        self.h_s_ref = ResidualMLP(in_dim=ref_hyper_dim, internal_dim=ref_feats_dim, out_dim=2 * ref_feats_dim, num_res_layer=2)
        self.entropy_bottleneck_ref = BatchEntropyBottleneck(channels=ref_hyper_dim)

        self.derive_factor = derive_factor      
        self.quant_step_res =  nn.Parameter(torch.ones(int(res_feats_dim/2), dtype=torch.float32, device=device) * quant_step * 0.1, requires_grad=True)
        self.quant_step_scale = nn.Parameter(torch.ones(scale_dim, dtype=torch.float32, device=device) * quant_step * 0.01, requires_grad=True)
        self.res_dim = res_feats_dim
        res_feats_dim = res_feats_dim * derive_factor
        res_hyper_dim = res_hyper_dim * derive_factor
        self.h_a_res = nn.Sequential(
            nn.Linear(in_features=int(res_feats_dim/2), out_features=res_hyper_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=res_hyper_dim, out_features=res_hyper_dim)
        )
        self.h_s_res_pre = ResidualMLP(in_dim=res_hyper_dim+ref_feats_dim, internal_dim=ref_feats_dim, out_dim=2 * int(res_feats_dim/2), num_res_layer=2)
        self.h_s_res_dfm = ResidualMLP(in_dim=ref_feats_dim+int(res_feats_dim/2), internal_dim=ref_feats_dim, out_dim=2 * int(res_feats_dim/2), num_res_layer=2)
        self.h_s_scale = ResidualMLP(in_dim=ref_feats_dim, internal_dim=ref_feats_dim, out_dim=2 * scale_dim, num_res_layer=2)
        self.entropy_bottleneck_res = BatchEntropyBottleneck(channels=res_hyper_dim)

        self.gaussian_conditional = BatchGaussianConditional(scale_table=None)
    def forward(self, y_ref: torch.Tensor, y_res: torch.Tensor, y_scale: torch.Tensor, iteration: int):
        z_ref = self.h_a_ref(y_ref)
        z_ref_hat, z_likelihoods_ref = self.entropy_bottleneck_ref(z_ref)
        gaussian_params_ref = self.h_s_ref(z_ref_hat)  # shape (N, 2 * C)
        means_hat_ref, scales_hat_ref = torch.chunk(gaussian_params_ref, dim=-1, chunks=2)
        y_ref = y_ref / self.quant_step_ref
        means_hat_ref = means_hat_ref / self.quant_step_ref
        scales_hat_ref = scales_hat_ref / self.quant_step_ref
        y_ref_hat, y_likelihoods_ref = self.gaussian_conditional(y_ref, scales=torch.sigmoid(scales_hat_ref), means=means_hat_ref)
        y_ref_hat = y_ref_hat * self.quant_step_ref
        
        y_res_pre = y_res[:, :, :int(self.res_dim/2)]
        y_res_dfm = y_res[:,:, int(self.res_dim/2):]
        z_res = self.h_a_res(rearrange(y_res_pre, 'n k g -> n (k g)'))
        z_res_hat, z_likelihoods_res = self.entropy_bottleneck_res(z_res)
        z_likelihoods_res_pre = rearrange(z_likelihoods_res, 'n (k g) -> n k g', k=self.derive_factor)
        gaussian_params_res_pre = rearrange(self.h_s_res_pre(torch.cat([z_res_hat, y_ref_hat], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_pre, scales_hat_res_pre = torch.chunk(gaussian_params_res_pre, dim=-1, chunks=2)
        means_hat_res_pre = rearrange(means_hat_res_pre, 'n k g -> (n k) g')
        scales_hat_res_pre = rearrange(scales_hat_res_pre, 'n k g -> (n k) g')
        y_res_pre = rearrange(y_res_pre, 'n k g -> (n k) g')
        y_res_pre = y_res_pre / self.quant_step_res
        means_hat_res_pre = means_hat_res_pre / self.quant_step_res
        scales_hat_res_pre = scales_hat_res_pre / self.quant_step_res
        y_res_pre_hat, y_likelihoods_res = self.gaussian_conditional(y_res_pre, scales=torch.sigmoid(scales_hat_res_pre), means=means_hat_res_pre)
        y_res_pre_hat = y_res_pre_hat * self.quant_step_res

        y_res_pre_hat_asctx = rearrange(y_res_pre_hat, '(n k) g -> n (k g)', k=self.derive_factor)
        gaussian_params_res_dfm = rearrange(self.h_s_res_dfm(torch.cat([y_res_pre_hat_asctx, y_ref_hat], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_dfm, scales_hat_res_dfm = torch.chunk(gaussian_params_res_dfm, dim=-1, chunks=2)
        means_hat_res_dfm = rearrange(means_hat_res_dfm, 'n k g -> (n k) g')
        scales_hat_res_dfm = rearrange(scales_hat_res_dfm, 'n k g -> (n k) g')
        y_res_dfm = rearrange(y_res_dfm, 'n k g -> (n k) g')
        y_res_dfm = y_res_dfm / self.quant_step_res
        means_hat_res_dfm = means_hat_res_dfm / self.quant_step_res
        scales_hat_res_dfm = scales_hat_res_dfm / self.quant_step_res
        y_res_dfm_hat, y_likelihoods_res_dfm = self.gaussian_conditional(y_res_dfm, scales=torch.sigmoid(scales_hat_res_dfm), means=means_hat_res_dfm)
        y_res_dfm_hat = y_res_dfm_hat * self.quant_step_res

        gaussian_params_scale = self.h_s_scale(y_ref_hat)
        means_hat_scale, scales_hat_scale = torch.chunk(gaussian_params_scale, dim=-1, chunks=2)
        means_hat_scale = means_hat_scale / self.quant_step_scale
        scales_hat_scale = scales_hat_scale / self.quant_step_scale
        y_scale = y_scale / self.quant_step_scale
        y_scale_hat, y_likelihoods_scale = self.gaussian_conditional(y_scale, scales=torch.sigmoid(scales_hat_scale), means=means_hat_scale)
        y_scale_hat = y_scale_hat * self.quant_step_scale

        y_res_hat = torch.cat([y_res_pre_hat, y_res_dfm_hat], dim=1)
        likelihoods = {
            'ref_feats': y_likelihoods_ref, 'ref_hyper': z_likelihoods_ref,
            'res_feats_pre': y_likelihoods_res, 'res_feats_dfm': y_likelihoods_res_dfm, 'res_hyper': z_likelihoods_res_pre,
            'scales': y_likelihoods_scale
        }
        return y_ref_hat, y_res_hat, y_scale_hat, likelihoods
    
    @torch.no_grad()
    def compress(self, ref_feats: torch.Tensor, res_feats: torch.Tensor, scaling_factors_before_exp: torch.Tensor):
        z_ref = self.h_a_ref(ref_feats)
        z_ref_strings = self.entropy_bottleneck_ref.compress(z_ref)
        z_ref_hat = self.entropy_bottleneck_ref.decompress(z_ref_strings, N=ref_feats.shape[0])
        gaussian_params_ref = self.h_s_ref(z_ref_hat)
        means_hat_ref, scales_hat_ref = torch.chunk(gaussian_params_ref, dim=-1, chunks=2)
        means_hat_ref = means_hat_ref / self.quant_step_ref
        scales_hat_ref = scales_hat_ref / self.quant_step_ref
        ref_feats = ref_feats / self.quant_step_ref
        indices_ref = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_ref))
        y_ref_strings = self.gaussian_conditional.compress(ref_feats, indices=indices_ref,means=means_hat_ref)
        y_ref_dec = self.gaussian_conditional.decompress(y_ref_strings, indices=indices_ref, means=means_hat_ref)
        y_ref_dec = y_ref_dec * self.quant_step_ref

        y_res_pre = res_feats[:, :, :int(self.res_dim/2)]
        y_res_dfm = res_feats[:, :, int(self.res_dim/2):]
        z_res_pre = self.h_a_res(rearrange(y_res_pre, 'n k g -> n (k g)'))
        z_res_strings = self.entropy_bottleneck_res.compress(z_res_pre)
        z_res_hat = self.entropy_bottleneck_res.decompress(z_res_strings, N=y_res_pre.shape[0])
        gaussian_params_res_pre = rearrange(self.h_s_res_pre(torch.cat([z_res_hat, y_ref_dec], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_pre, scales_hat_res_pre = torch.chunk(gaussian_params_res_pre, dim=-1, chunks=2)
        means_hat_res_pre = rearrange(means_hat_res_pre, 'n k g -> (n k) g')
        scales_hat_res_pre = rearrange(scales_hat_res_pre, 'n k g -> (n k) g')
        y_res_pre = rearrange(y_res_pre, 'n k g -> (n k) g')
        means_hat_res_pre = means_hat_res_pre / self.quant_step_res
        scales_hat_res_pre = scales_hat_res_pre / self.quant_step_res
        y_res_pre = y_res_pre / self.quant_step_res
        indices_res_pre = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_res_pre))
        y_res_pre_strings = self.gaussian_conditional.compress(y_res_pre, indices=indices_res_pre, means=means_hat_res_pre)
        y_res_pre_dec = self.gaussian_conditional.decompress(y_res_pre_strings, indices=indices_res_pre, means=means_hat_res_pre)
        y_res_pre_dec = y_res_pre_dec * self.quant_step_res

        y_res_pre_hat_asctx = rearrange(y_res_pre_dec, '(n k) g -> n (k g)', k=self.derive_factor)
        gaussian_params_res_dfm = rearrange(self.h_s_res_dfm(torch.cat([y_res_pre_hat_asctx, y_ref_dec], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_dfm, scales_hat_res_dfm = torch.chunk(gaussian_params_res_dfm, dim=-1, chunks=2)
        means_hat_res_dfm = rearrange(means_hat_res_dfm, 'n k g -> (n k) g')
        scales_hat_res_dfm = rearrange(scales_hat_res_dfm, 'n k g -> (n k) g')
        y_res_dfm = rearrange(y_res_dfm, 'n k g -> (n k) g')
        means_hat_res_dfm = means_hat_res_dfm / self.quant_step_res
        scales_hat_res_dfm = scales_hat_res_dfm / self.quant_step_res
        y_res_dfm = y_res_dfm / self.quant_step_res
        indices_res_dfm = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_res_dfm))
        y_res_dfm_strings = self.gaussian_conditional.compress(y_res_dfm, indices=indices_res_dfm, means=means_hat_res_dfm)

        gaussian_params_scale = self.h_s_scale(y_ref_dec)
        means_hat_scale, scales_hat_scale = torch.chunk(gaussian_params_scale, dim=-1, chunks=2)
        means_hat_scale = means_hat_scale / self.quant_step_scale
        scales_hat_scale = scales_hat_scale / self.quant_step_scale
        scaling_factors_before_exp = scaling_factors_before_exp / self.quant_step_scale
        indices_scale = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_scale))
        y_scale_strings = self.gaussian_conditional.compress(scaling_factors_before_exp, indices=indices_scale, means=means_hat_scale)

        strings = {
            'ref_feats_strings': y_ref_strings, 'ref_hyper_strings': z_ref_strings,
            'res_feats_pre_strings': y_res_pre_strings, 'res_hyper_strings': z_res_strings, 'res_feats_dfm_strings':y_res_dfm_strings,
            'scale_strings': y_scale_strings,
        }
        return strings
    
    @torch.no_grad()
    def decompress(self, y_ref_strings: bytes, z_ref_strings: bytes, y_res_pre_strings: bytes, z_res_strings: bytes, y_res_dfm_strings: bytes, y_scale_strings: bytes, N: int):
        z_ref_hat = self.entropy_bottleneck_ref.decompress(z_ref_strings, N=N)
        gaussian_params_ref = self.h_s_ref(z_ref_hat)
        means_hat_ref, scales_hat_ref = torch.chunk(gaussian_params_ref, dim=-1, chunks=2)
        means_hat_ref = means_hat_ref / self.quant_step_ref
        scales_hat_ref = scales_hat_ref / self.quant_step_ref
        indices_ref = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_ref))
        y_ref_dec = self.gaussian_conditional.decompress(y_ref_strings, indices=indices_ref, means=means_hat_ref)
        y_ref_dec = y_ref_dec.squeeze(dim=0)
        y_ref_dec = y_ref_dec * self.quant_step_ref

        z_res_hat = self.entropy_bottleneck_res.decompress(z_res_strings, N=N)
        gaussian_params_res_pre = rearrange(self.h_s_res_pre(torch.cat([z_res_hat, y_ref_dec], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_pre, scales_hat_res_pre = torch.chunk(gaussian_params_res_pre, dim=-1, chunks=2)
        means_hat_res_pre = rearrange(means_hat_res_pre, 'n k g -> (n k) g')
        scales_hat_res_pre = rearrange(scales_hat_res_pre, 'n k g -> (n k) g')
        means_hat_res_pre = means_hat_res_pre / self.quant_step_res
        scales_hat_res_pre = scales_hat_res_pre / self.quant_step_res
        indices_res_pre = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_res_pre))
        y_res_pre_dec_asctx = self.gaussian_conditional.decompress(y_res_pre_strings, indices=indices_res_pre, means=means_hat_res_pre)
        y_res_pre_dec_asctx = y_res_pre_dec_asctx * self.quant_step_res
        y_res_pre_dec = rearrange(y_res_pre_dec_asctx, '(n k) g -> n k g', k=self.derive_factor)
        
        y_res_pre_hat_asctx = rearrange(y_res_pre_dec_asctx, '(n k) g -> n (k g)', k=self.derive_factor)
        gaussian_params_res_dfm = rearrange(self.h_s_res_dfm(torch.cat([y_res_pre_hat_asctx, y_ref_dec], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat_res_dfm, scales_hat_res_dfm = torch.chunk(gaussian_params_res_dfm, dim=-1, chunks=2)
        means_hat_res_dfm = rearrange(means_hat_res_dfm, 'n k g -> (n k) g')
        scales_hat_res_dfm = rearrange(scales_hat_res_dfm, 'n k g -> (n k) g')
        means_hat_res_dfm = means_hat_res_dfm / self.quant_step_res
        scales_hat_res_dfm = scales_hat_res_dfm / self.quant_step_res
        indices_res_dfm = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_res_dfm))
        y_res_dfm_dec = self.gaussian_conditional.decompress(y_res_dfm_strings, indices=indices_res_dfm, means=means_hat_res_dfm)
        y_res_dfm_dec = y_res_dfm_dec * self.quant_step_res
        y_res_dfm_dec = rearrange(y_res_dfm_dec, '(n k) g -> n k g', k=self.derive_factor)

        gaussian_params_scale = self.h_s_scale(y_ref_dec)
        means_hat_scale, scales_hat_scale = torch.chunk(gaussian_params_scale, dim=-1, chunks=2)
        means_hat_scale = means_hat_scale / self.quant_step_scale
        scales_hat_scale = scales_hat_scale / self.quant_step_scale
        indices_scale = self.gaussian_conditional.build_indexes(F.softmax(scales_hat_scale))
        y_scale_dec = self.gaussian_conditional.decompress(y_scale_strings, indices=indices_scale, means=means_hat_scale)
        y_scale_dec = y_scale_dec * self.quant_step_scale
        
        y_res_dec = torch.cat([y_res_pre_dec,y_res_dfm_dec],dim=2)
        return y_ref_dec, y_res_dec, y_scale_dec

    def aux_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Auxiliary parameters of entropy bottleneck for training.
        :param recurse: whether to include parameters of submodules.
        """
        parameters_aux = set(n for n, p in self.named_parameters(recurse=recurse) if n.endswith('.quantiles') and p.requires_grad)
        params_dict = dict(self.named_parameters(recurse=recurse))
        params_aux = (params_dict[n] for n in sorted(list(parameters_aux)))

        return params_aux