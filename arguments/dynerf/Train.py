_base_ = './default.py'

ModelHiddenParams = dict(
    voxel_size_init = 0.03,
)

OptimizationParams = dict(
    maxtime = 50,
    
    iterations = 120_000,
    densify_until_iter_coarse = 120_000,
    densify_until_iter = 120_000,
    position_lr_max_steps_coarse = 120_000,
    position_lr_max_steps = 120_000,
    deformation_lr_max_steps = 120_000,
    reg_coef=0.01,

    network_lr_max_steps=120_000,

    densify_grad_threshold_fine_init = 0.001,
    densify_grad_threshold_after = 0.001,
    opacity_threshold_fine_init = 0.003,
    opacity_threshold_fine_after = 0.003,

    position_lr_init = 0.004,
    position_lr_final = 0.000004,
    position_lr_delay_mult = 0.01,
    deformation_lr_init = 0.01,
    deformation_lr_final = 0.00001,
    deformation_lr_delay_mult = 0.01,  

)