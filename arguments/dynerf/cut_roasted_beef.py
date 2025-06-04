_base_ = './default.py'
ModelParams = dict(
    loader = "dynerf"
)

ModelHiddenParams = dict(
    defor_depth = 1,
    net_width = 128,
    no_ds = False,
    no_do = False,
    no_dc = False,
    
    use_coarse_temporal_embedding = True,
    c2f_temporal_iter = 10000,
    deform_from_iter = 5000,
    total_num_frames = 300,
    voxelize_iter = 60000,
    ref_embedding_dim = 32,
    res_embedding_dim = 16,
    derive_factor = 10,
    ref_hyper_dim = 4,
    res_hyper_dim = 1,
    position_embedding_dim = 72,
    voxel_size_init = 0.5,
    voxel_size_compress = 0.001,
    use_significance_weight = False
)


OptimizationParams = dict(
    dataloader = True,
    batch_size = 1,
    iterations = 80_000,
    #30_000
    maxtime = 300,
    #300

    densify_from_iter = 9000,    
    pruning_from_iter = 9000,

    densify_grad_threshold_fine_init = 0.01,
    densify_grad_threshold_after = 0.01,

    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    densify_until_iter = 60_000,
    position_lr_max_steps = 80_000,
    deformation_lr_max_steps = 80_000,
    network_lr_max_steps=80_000,

    rate_control_start_iter = 15_000,
    compress_start_iter = 10_000,

    lambda_dssim = 0.2,
    num_multiview_ssim = 5,
    use_colmap = True,
    reg_coef = 0.01,

    position_lr_init = 0.004,
    position_lr_final = 0.000004,
    position_lr_delay_mult = 0.01,
    deformation_lr_init = 0.0004,
    deformation_lr_final = 0.000004,
    deformation_lr_delay_mult = 0.01,
)