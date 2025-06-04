_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 20,
    max_embeddings = 100,
    c2f_temporal_iter = 20000,
    total_num_frames = 207,
    voxel_size_init = 0.5,
)

OptimizationParams = dict(
    maxtime = 207,
    iterations = 60_000,
    position_lr_max_steps = 60_000,
    deformation_lr_max_steps = 60_000,
    network_lr_max_steps=60_000,

    densify_from_iter = 9000,
    pruning_from_iter = 9000,
    # densify_grad_threshold_fine_init = 0.01,
    # densify_grad_threshold_after = 0.01,
    # opacity_threshold_fine_init = 0.003,
    # opacity_threshold_fine_after = 0.003,
    densify_grad_threshold_fine_init = 0.005,
    densify_grad_threshold_after = 0.005,
    opacity_threshold_fine_init = 0.003,
    opacity_threshold_fine_after = 0.003,

    rate_control_start_iter = 15_000,
    compress_start_iter = 10_000,
    lambda_dssim = 0.2,
    use_colmap = True,
    reg_coef = 0.01,

    position_lr_init = 0.004,
    position_lr_final = 0.000004,
    position_lr_delay_mult = 0.01,
    deformation_lr_init = 0.0004,
    deformation_lr_final = 0.000004,
    deformation_lr_delay_mult = 0.01,
    # pruning_interval = 2000
)