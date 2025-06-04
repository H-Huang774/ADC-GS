ModelParams = dict(
    loader = "nerfies",
    shuffle = False
)

ModelHiddenParams = dict(
    defor_depth = 0,
    net_width = 128,
    no_ds = False,
    no_do = False,
    no_dc = False,
    
    temporal_embedding_dim = 256,
    gaussian_embedding_dim = 32,
    zero_temporal = True,
    use_anneal = False,
    use_coarse_temporal_embedding = True,
    deform_from_iter = 5000,
    ref_embedding_dim = 32,
    res_embedding_dim = 16,
    derive_factor = 10,
    ref_hyper_dim = 4,
    res_hyper_dim = 1,
    position_embedding_dim = 72,
    voxel_size_compress = 0.001,
    use_significance_weight = False,

    voxelize_iter = 50_000,

)

OptimizationParams = dict(
    dataloader = True,
    batch_size = 2,
    opacity_reset_interval = 6000000,  
    
    scene_bbox_min = [-3.0, -1.8, -1.2],
    scene_bbox_max = [3.0, 1.8, 1.2],
    num_pts = 2000,
    threshold = 3,
    downsample = 1.0,

    coarse_stage_frame_num = 0,
    num_multiview_ssim = 0,
    offsets_lr = 0,
    coef_tv_temporal_embedding = 0,

    lambda_rate_control = 0.001,
    #0.0005-0.05
    densify_until_iter = 50_000,
)