from .config_eco_seg import *


class ECOSegConfig:
    root = '/home/cxq/study/objectTracking/code/ECOSeg'

    cnn_params = {'fname': "cnn-resnet50",
                  'compressed_dim': [16, 64]
                  }
    features = [cnn_params]

    # feature parameters
    normalize_power = 2
    normalize_size = True
    normalize_dim = True
    square_root_normalization = False

    # image sample parameters
    search_area_shape = 'square'
    search_area_scale = 4.5
    min_sample_size = 200
    max_sample_size = 250
    min_image_sample_size = min_sample_size ** 2
    max_image_sample_size = max_sample_size ** 2

    # detection parameters
    refinement_iterations = 1  # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5
    clamp_position = False  # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 8.  # label function sigma
    learning_rate = 0.010
    num_samples = 50
    sample_replace_startegy = 'lowest_prior'
    lt_size = 0
    train_gap = 5
    skip_after_frame = 1
    use_detection_sample = True

    # factorized convolution parameters
    use_projection_matrix = True
    update_projection_matrix = True
    proj_init_method = 'pca'
    projection_reg = 5e-8

    # generative sample space model parameters
    use_sample_merge = True
    sample_merge_type = 'merge'
    distance_matrix_update_type = 'exact'

    # CG paramters
    CG_iter = 5
    init_CG_iter = 15 * 15
    init_GN_iter = 15
    CG_use_FR = False
    CG_standard_alpha = True
    CG_forgetting_rate = 75
    precond_data_param = 0.3
    precond_reg_param = 0.015
    precond_proj_param = 35

    # regularization window paramters
    use_reg_window = True
    reg_window_min = 1e-4
    reg_window_edge = 10e-3
    reg_window_power = 2
    reg_sparsity_threshold = 0.05

    # interpolation parameters
    interp_method = 'bicubic'
    interp_bicubic_a = -0.75
    interp_centering = True
    interp_windowing = False

    # scale parameters
    number_of_scales = 1
    scale_step = 1.02  # 1.015
    use_scale_filter = False

    # gpu
    use_gpu = True
    gpu_id = 2

    # ECO Seg

    # model path
    model = 'resnet50'
    checkpoint_path = os.path.join(root, 'train_output/history/5/model_final.pth')
    total_stride = 16
    num_dims = [64, 1024]
    feature_idx = [0, 1]
    mask_feature_idx = [1]

    ## roi head
    # loss
    roi_head_fg_iou_thresh = 0.5
    roi_head_bg_iou_thresh = 0.5

    ## roi head box
    roi_box = True
    # Feature extractor
    roi_box_pooler_resolution = 14  # divide roi into 7*7
    roi_box_pooler_scales = (0.0625,)
    roi_box_pooler_sampling_scale = 0

    roi_box_head_dim = 1024
    roi_box_num_class = 2

    # Post process
    roi_box_use_fpn = False
    roi_box_reg_weights = (10.0, 10.0, 5.0, 5.0)
    roi_box_score_thresh = 0.05
    roi_box_nms = 0.5
    roi_box_detecions_per_image = 100  # need to change to 1 or 2?

    # loss
    roi_box_batch_sz_per_img = 512
    roi_box_positive_fraction = 0.25
    roi_box_agnostic_bbox_reg = False  # 2 class or not

    ## roi head mask
    # Feature extractor
    roi_mask_pooler_resolution = 14  # divide roi into 7*7
    roi_mask_pooler_scales = (0.25, 0.125, 0.0625, 0.03125)
    roi_mask_pooler_sampling_scale = 2

    roi_mask_conv_layers = (256, 256, 256, 256)
    roi_mask_dialation = 1

    roi_mask_share_box_feature = True

    # Masker
    mask_thresh = 0.5
    mask_padding = 1

    # Train
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    output_dir = './train_output'

    # CheckPoint
    paths_catalog = os.path.join(root, 'seg/config/paths_catalog.py')
    model_backbone_conv_body = 'R-50-C4'
    model_weight = 'catalog://ImageNetPretrained/MSRA/R-50'
    # Solver
    base_lr = 0.0025
    weight_decay = 0.0001
    base_lr_factor = 2
    weight_decay_bias = 0
    momentum = 0.9

    images_per_batch = 2
    test_images_per_batch = 1
    max_iter = 720000
    checkpoint_save_period = 2500  # checkpoint save period(iters)

    # LR schedular
    steps = (480000, 640000)
    gamma = 0.1
    warmup_factor = 1 / 3
    warmup_iters = 500
    warmup_method = 'linear'
    # DataLoader
    aspect_ratio_grouping = True
    # Datasets
    datasets_train = ('coco_2014_train', 'coco_2014_valminusminival')
    datasets_test = ('coco_2014_minival',)
    category = [1]  # category id to use, None for all categories
    min_size_train = min_sample_size
    max_size_train = max_sample_size
    min_size_test = min_sample_size
    max_size_test = max_sample_size
    to_bgr255 = True
    pixel_mean = [102.9801, 115.9465, 122.7717]
    pixel_std = [1.0, 1.0, 1.0]
    size_divisibility = 0
    num_workers = 4

    data_argument = DataArgument()


config = ECOSegConfig()
