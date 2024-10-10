OptimizationParams = dict(
    prompt='The tree cut off by a harvester',
    # first one is static
    obj_prompt = [
        'a tree',
        'a tree is cut off',
        'a harvester',
    ],
    scales = [1.0, 0.6, 1],
    func_name = [],
    video_sds_type = 'zeroscope',
    cfg_temporal = 100,
    cfg = 100,
    static_iterations = 0,
    coarse_iterations = 0,
    iterations = 4500, 
    position_lr_max_steps = 20000,
    position_lr_delay_mult = 1, 
    pruning_interval = 100,
    pruning_interval_fine = 100000,
    percent_dense = 0.01,
    densify_grad_threshold_fine_init = 0.5,
    densify_grad_threshold_coarse = 0.5,
    densify_grad_threshold_after = 0.1,
    deformation_lr_delay_mult = 1,
    deformation_lr_init = 0.0002,
    deformation_lr_final = 0.0002,
    grid_lr_init = 0.016,
    grid_lr_final = 0.016,
    densification_interval = 100,
    opacity_reset_interval = 300,
    lambda_lpips = 2,
    lambda_dssim = 2,
    lambda_pts = 0,
    lambda_zero123 = 0.5, # not used
    fine_rand_rate = 1,
)

ModelParams = dict(
    frame_num = 16,
    name="rose",
    rife=False,
    radius=3.0,
    render_W=512,
    render_H=512,
    cloud_path = [
        'input_ply/tree_cut/tree.ply',
        'input_ply/tree_cut/a_broken_tree.ply',
        'input_ply/tree_cut/harvest.ply',
    ]
)

ModelHiddenParams = dict(
    no_grid = True,
    grid_merge = 'cat', # not used
    multires = [1, 2, 4, 8 ], # not used
    defor_depth = 5,
    net_width = 128,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    bounds=2,
    no_ds=True,
    no_dr=True,
    no_do=True,
    no_dc=True,
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 24]  #8 is frame numbers/2
    }
)

TrajParams = dict(
    init_pos = [
        (-0.2, 0.0, 0.2),
        (-0.3, -0.3, 0.0),
        (2.0, -0.1, -0.2)
    ],
    move_list = [
        [(0.0, 0.0, 0.0), (-3.0, -3.0, 0.0), (0.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0)],
        [(-3.6, 0.0, 0.0), (0.0, 0.0, 0.0)],
    ],
    move_time = [
        [0.45, 0.6],
        [],
        [0.4],
    ],
    init_angle = [
        (0, 0, 0),
        (0, 180, 0),
        (0, 180, 0),
    ],
    rotations = [
        [(0, 0, 0), (0, 0, 12.5), (0,0,0)],
        [(0, 0, 0)],
        [(0, 0, 0)],
    ],
    rotations_time = [
        [0.45, 0.6],
        [],
        [],
    ],
    appear_init = [1,0,1],
    appear_trans_time = [
        [0.6],
        [0.6],
        [],
    ],
    img_prompt_list = [],
    img_prompt_time = [0.6],
    trans_list = [(0,1)],
    trans_period = [(0.54, 0.66)]
)
