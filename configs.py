from dataclasses import dataclass


@dataclass
class CC359BaseConfig:
    use_accumulate_for_loss = True
    debug= False
    exp_dir = None
    data_len = 45
    input_size = (256,256)
    base_splits_path ='/home/dsi/shaya/unsup_splits/'
    base_res_path = '/home/dsi/shaya/unsup_resres/'
    msm = False
    n_channels = 1
    save_pred_every =  500
    epoch_every = 1000
    num_steps = 10000
    parallel_model = False

@dataclass
class AdabnCC359Config(CC359BaseConfig):
    batch_size = 16


@dataclass
class CC359ConfigPretrain(CC359BaseConfig):
    source_batch_size = 16
    target_batch_size = 1
    lr = 1e-3

@dataclass
class CC359ConfigFinetuneClustering(CC359BaseConfig):
    source_batch_size = 4
    target_batch_size = 12
    n_clusters = 12
    lr = 1e-5
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
    dist_loss_lambda = 0.1

@dataclass
class DebugConfigCC359(CC359BaseConfig):
    n_clusters = 2
    source_batch_size = 2
    target_batch_size = 2
    lr = 1e-5
    data_len = 3
    save_pred_every =  5
    debug= True
    epoch_every = 20
    num_steps = 50
    dist_loss_lambda = 0.1
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'




@dataclass
class MsmBaseConfig:
    use_accumulate_for_loss = True
    debug= False
    exp_dir = None
    input_size = (384,384)
    base_splits_path ='/home/dsi/shaya/unsup_splits_msm/'
    base_res_path ='/home/dsi/shaya/unsup_resres_msm/'
    msm = True
    n_channels= 3
    save_pred_every =  100
    epoch_every = 250
    num_steps = 2500

@dataclass
class MsmPretrainConfig(MsmBaseConfig):
    lr = 1e-3
    source_batch_size = 16
    target_batch_size = 1
    parallel_model = True

@dataclass
class MsmConfigFinetuneClustering(MsmBaseConfig):
    n_clusters = 12
    lr = 1e-4
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices_msm.json'
    source_batch_size = 4
    target_batch_size = 12
    dist_loss_lambda = 0.03
    parallel_model = True


@dataclass
class DebugMsm(MsmBaseConfig):
    n_clusters = 2
    source_batch_size = 2
    target_batch_size = 2
    lr = 1e-5
    save_pred_every =  5
    debug= True
    epoch_every = 20
    num_steps = 50

@dataclass
class AdabnMsmConfig(MsmBaseConfig):
    batch_size = 16





