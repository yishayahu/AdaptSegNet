from dataclasses import dataclass

import numpy as np


@dataclass
class CC359BaseConfig:
    use_accumulate_for_loss = True
    debug= False
    exp_dir = None
    data_len = 45
    input_size = (256,256)
    base_splits_path ='/home/dsi/shaya/unsup_splits/'
    base_res_path = '/home/dsi/shaya/unsup_resres_zoom/'
    msm = False
    n_channels = 1
    save_pred_every =  500
    epoch_every = 1000
    n_chans_out = 2
    parallel_model = False
    drop_last = False

@dataclass
class AdabnCC359Config(CC359BaseConfig):
    batch_size = 16


@dataclass
class CC359ConfigTheir(CC359BaseConfig):
    source_batch_size = 8
    target_batch_size = 8
    num_steps = 6500
    lr = 5e-6
    sched = True
    sched_gamma = 0.1
    milestones = [5000,6000]

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
class CC359ConfigJdot(CC359BaseConfig):

    num_steps = 8000
    lr = 5e-4
    alpha = 0.001
    beta = 0.0001
    sched = False
    sched_gamma = 0.1
    milestones = [5000,6000]
    use_adjust_lr = False
    n_chans_out = 2
    random_patch = False
    patch_size = np.array([256,256])
    source_batch_size = 16
    target_batch_size = 16

    # random_patch = True
    # patch_size = np.array([64,64])
    # source_batch_size = 128
    # target_batch_size = 128
    drop_last = True


@dataclass
class CC359ConfigAblation(CC359BaseConfig):
    use_accumulate_for_loss = False
    source_batch_size = 8
    target_batch_size = 8
    n_clusters = 12
    num_steps = 6500
    lr = 5e-6
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
    dist_loss_lambda = 0.9
    sched = True
    sched_gamma = 0.1
    milestones = [5000,6000]
    use_adjust_lr = False


@dataclass
class CC359ConfigFinetuneClustering(CC359BaseConfig):
    source_batch_size = 8
    target_batch_size = 8
    n_clusters = 12
    num_steps = 6500
    lr = 5e-6
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
    dist_loss_lambda = 0.9
    sched = True
    sched_gamma = 0.1
    acc_amount = 35
    milestones = [5000,6000]
    use_adjust_lr = False

@dataclass
class DebugConfigCC359(CC359BaseConfig):
    n_clusters = 2
    source_batch_size = 4
    target_batch_size = 4
    lr = 1e-5
    data_len = 10
    save_pred_every =  5
    debug= True
    epoch_every = 40
    num_steps = 200
    dist_loss_lambda = 0.1
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices.json'
    sched = False
    use_adjust_lr = False
    acc_amount = 4




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
    save_pred_every = 50# 100
    epoch_every = 50#250
    num_steps = 2500#3500
    n_chans_out = 2
    drop_last = False

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
    lr = 1e-6
    use_slice_num = True
    id_to_num_slices = '/home/dsi/shaya/id_to_num_slices_msm.json'
    source_batch_size = 8
    target_batch_size = 8
    dist_loss_lambda = 0.8
    parallel_model = True
    sched = False
    acc_amount = 20
    sched_gamma = 0.1
    milestones = [1000,2000,3000]
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


@dataclass
class MsmConfigJdot(MsmBaseConfig):
    lr = 1e-4
    alpha = 0.005
    beta = 0.0005
    sched = True
    sched_gamma = 0.1
    milestones = [2000,3000]
    use_adjust_lr = False
    n_chans_out = 2
    # random_patch = False
    # patch_size = np.array([384,384])
    # source_batch_size = 8
    # target_batch_size = 8
    random_patch = True
    patch_size = np.array([64,64])
    source_batch_size = 32
    target_batch_size = 32
    drop_last = True
    parallel_model = False


