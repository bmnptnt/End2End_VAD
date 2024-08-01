
def build_config(feat_dir,gt_dir,list_dir,mode,ckpt):
    cfg = type('', (), {})()
    if mode in ['ucf', 'ucf-crime','UCF','UCF-Crimes','UCF_Testing','Custom_Dataset','Custom']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = feat_dir
        cfg.test_list = list_dir
        cfg.token_feat = './prompt/ucf-prompt.npy'
        cfg.gt = gt_dir
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 9
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 9
        # test settings
        cfg.test_bs = 1
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 7  # smooth window
        cfg.ckpt_path = f"./anomaly_detector_PEL/ckpt/{ckpt}"

    elif mode in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.model_name = 'xd_'
        cfg.metrics = 'AP'
        cfg.feat_prefix = feat_dir
        cfg.test_list = list_dir
        cfg.token_feat = './prompt/xd-prompt.npy'
        cfg.gt = gt_dir
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.06
        cfg.bias = 0.02
        cfg.norm = False
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 1
        cfg.seed = 4
        # test settings
        cfg.test_bs = 5
        cfg.smooth = 'fixed'  # ['fixed': 8, slide': 3]
        cfg.kappa = 8  # smooth window
        cfg.ckpt_path = f"./anomaly_detector_PEL/ckpt/{ckpt}"

    elif mode in ['sh', 'SHTech']:
        cfg.dataset = 'shanghaiTech'
        cfg.model_name = 'SH_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = feat_dir
        cfg.test_list = list_dir
        cfg.token_feat = './prompt/sh-prompt.npy'
        cfg.abn_label = './list/sh/relabel.list'
        cfg.gt = gt_dir
        # TCA settings
        cfg.win_size = 5
        cfg.gamma = 0.08
        cfg.bias = 0.1
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.2
        cfg.lamda = 9
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window
        cfg.ckpt_path = f"./anomaly_detector_PEL/ckpt/{ckpt}"

    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 5e-4
    cfg.dropout = 0.1
    cfg.train_bs = 128
    cfg.max_seqlen = 200
    cfg.max_epoch = 50
    cfg.workers = 8
    cfg.save_dir = './ckpt/'
    cfg.logs_dir = './anomaly_detector_PEL/log_info.log'

    return cfg
