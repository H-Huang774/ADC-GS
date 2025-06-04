import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.Common import RenderResults
from utils.sh_utils import eval_sh
from time import time as get_time
from einops import repeat

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cam_no=None, iter=None, train_coarse=False, \
    num_down_emb_c=5, num_down_emb_f=5,skip_quant=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    scalings_before_exp = pc.get_scaling_before_exp
    ref_embedding = pc.get_ref_embedding
    res_embedding = pc.get_res_embedding


    camera_center = viewpoint_camera.camera_center

    if (iter <= 8000): 

        gaussian_primitives, _, _, bpp, ref_feats, deform_res_feats = pc.network(
            means=means3D, scaling_factors_before_exp=scalings_before_exp, ref_feats=ref_embedding, res_feats=res_embedding,
            cam_center=camera_center, skip_quant=skip_quant, iteration=iter)

        means3D_final = gaussian_primitives.means
        scales_final = gaussian_primitives.scales
        rotations_final = gaussian_primitives.rotations
        opacity = gaussian_primitives.opacities
        colors_precomp = gaussian_primitives.colors
    else:

        gaussian_primitives, _, _, bpp, ref_feats, deform_res_feats = pc.network(
            means=means3D, scaling_factors_before_exp=scalings_before_exp, ref_feats=ref_embedding, res_feats=res_embedding,
            cam_center=camera_center, skip_quant=skip_quant, iteration=iter)
        
    if (iter >= 8000):
        anchor_position_embedding = pc.position_embedding
        ref_feats = torch.cat([ref_feats, anchor_position_embedding], dim=1)
        ref_feats = repeat(ref_feats, 'n c -> (n k) c', k=pc.derive_factor)

        deform_embedding = torch.cat([ref_feats, deform_res_feats], dim=1)

        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0]*pc.derive_factor,1)
        means3D_final, scales_final, rotations_final, opacity_final, colors_precomp = pc.deformation(gaussian_primitives.means, gaussian_primitives.scales,
            gaussian_primitives.rotations, gaussian_primitives.opacities, time, cam_no, deform_embedding, None, gaussian_primitives.colors, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)
        scales_final = torch.sigmoid(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)

    depth = None
    projected_means = torch.zeros_like(gaussian_primitives.means, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
    try:
        projected_means.retain_grad()
    except:
        pass

    means2D = projected_means
    outputs = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    if len(outputs) == 2:
        rendered_image, radii = outputs
    elif len(outputs) == 3:
        rendered_image, radii, depth = outputs
    else:
        assert False, "only (depth-)diff-gaussian-rasterization supported!"

    render_results = RenderResults(
        rendered_img=torch.clamp(rendered_image, min=0., max=1.),  # clamp to [0, 1]
        projected_means=projected_means,
        visibility_mask=radii > 0,
        pred_opacities=opacity,
        scales=scales_final,
        anchor_primitive_visible_mask=None,
        coupled_primitive_mask=None,
        bpp=bpp,
        radii=radii
    )

    return render_results