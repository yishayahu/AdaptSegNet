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
    epoch_every = 500

    parallel_model = False

@dataclass
class AdabnCC359Config(CC359BaseConfig):
    batch_size = 16


@dataclass
class CC359ConfigTheir(CC359BaseConfig):
    source_batch_size = 8
    target_batch_size = 8
    num_steps = 5000
    lr = 1e-4
    sched = True
    sched_gamma = 0.1
    milestones = [3500,5000,6500]

@dataclass
class CC359ConfigPretrain(CC359BaseConfig):
    source_batch_size = 16
    target_batch_size = 1
    num_steps = 5000
    lr = 1e-3
    sched = True
    sched_gamma = 0.1
    milestones = [8000,9500]

@dataclass
class CC359ConfigFinetuneClustering(CC359BaseConfig):
    source_batch_size = 4
    target_batch_size = 12
    n_clusters = 12
    num_steps = 5000
    lr = 1e-6
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
    dist_loss_lambda = 2
    sched = True
    sched_gamma = 0.1
    acc_amount = 40
    milestones = [2500,4000]
    use_adjust_lr = False

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
    num_steps = 3500

@dataclass
class MsmPretrainConfig(MsmBaseConfig):
    lr = 1e-3
    source_batch_size = 16
    target_batch_size = 1
    parallel_model = False
    sched =False

@dataclass
class MsmConfigFinetuneClustering(MsmBaseConfig):
    n_clusters = 8
    lr = 1e-5
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices_msm.json'
    source_batch_size = 8
    target_batch_size = 8
    dist_loss_lambda = 0.1
    parallel_model = True
    sched = False
    acc_amount = 60
    sched_gamma = 0.1
    milestones = [1500,2500]
    use_adjust_lr = False



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
    sched = False
    dist_loss_lambda = 0.1
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices_msm.json'

@dataclass
class AdabnMsmConfig(MsmBaseConfig):
    batch_size = 16





