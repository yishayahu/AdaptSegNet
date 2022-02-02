from dataclasses import dataclass



@dataclass
class BaseConfig:
    batch_size = 8
    use_accumulate_for_loss = True
    debug= False
    exp_dir = None

@dataclass
class DebugConfigCC359:
    n_clusters = 2
    batch_size = 2
    lr = 1e-5
    use_accumulate_for_loss = True
    data_len = 3
    save_pred_every =  5
    debug= True
    epoch_every = 20
    num_steps = 50
    exp_dir = None

@dataclass
class DebugMsm:
    n_clusters = 2
    batch_size = 2
    lr = 1e-3
    use_accumulate_for_loss = True
    save_pred_every =  5
    debug= True
    epoch_every = 20
    num_steps = 50
    exp_dir = None
    input_size = (384,384)
    base_splits_path ='/home/dsi/shaya/unsup_splits_msm/'
    base_res_path ='/home/dsi/shaya/unsup_resres_msm/'
    msm = True
    n_channels= 3

@dataclass
class MsmConfig(BaseConfig):
    n_clusters = 9
    lr = 1e-5
    input_size = (384,384)
    base_splits_path ='/home/dsi/shaya/unsup_splits_msm/'
    base_res_path ='/home/dsi/shaya/unsup_resres_msm/'
    msm = True
    n_channels= 3
    save_pred_every = 100
    epoch_every = 100
    num_steps = 1000
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices_msm.json'

@dataclass
class CcConfig(BaseConfig):
    n_clusters = 9
    lr = 1e-5
    data_len = 45
    input_size = (256,256)
    base_splits_path ='/home/dsi/shaya/unsup_splits/'
    base_res_path = '/home/dsi/shaya/unsup_resres/'
    msm = False
    n_channels = 1
    save_pred_every =  500
    epoch_every = 1000
    num_steps = 10000
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
