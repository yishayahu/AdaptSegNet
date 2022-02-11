import argparse
import itertools
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from multiprocessing import Process

import torch.cuda
from spottunet import paths

from tqdm import tqdm
from wandb.vendor.pynvml.pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetUtilizationRates, nvmlInit

from spottunet.paths import st_splits_dir, st_res_dir, msm_res_dir, msm_splits_dir

from configs import *


def find_available_device(my_devices, running_now,exp_name):
    if torch.cuda.is_available():
        if 'cluster' in exp_name:
            wanted_free_mem = 26 * 2 ** 30  # at least 26 GB avail
        else:
            wanted_free_mem = 16 * 2 ** 30
        while True:
            for device_num in range(nvmlDeviceGetCount()):
                if device_num in my_devices:
                    continue
                h = nvmlDeviceGetHandleByIndex(device_num)
                info = nvmlDeviceGetMemoryInfo(h)
                gpu_utilize = nvmlDeviceGetUtilizationRates(h)
                if info.free > wanted_free_mem and gpu_utilize.gpu < 3:
                    return device_num
            print(f'looking for device my device is {my_devices}')
            places = [x[0] for x in running_now]
            print(places)
            time.sleep(10)
    else:
        return 'cpu'


def run_single_exp(exp, device, source, target, scores_path, my_devices, ret_value):
    my_devices.append(device)
    print(f'training on source {source} target {target} exp {exp} on device {device} my devices is {my_devices}')
    with tempfile.NamedTemporaryFile() as out_file, tempfile.NamedTemporaryFile() as err_file:
        try:
            if 'adaBN' in exp:
                cmd = f'python adaBN.py --device cuda:{device} --source {source} --target {target} >  {out_file.name} 2> {err_file.name}'
            else:
                cmd = f'python train_gta2cityscapes_multi.py  --gpu {device}  --source {source} --target {target} --mode {exp}>  {out_file.name} 2> {err_file.name}'
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            # scores =json.load(open(scores_path))
            # sdice = scores['sdice/test']
            # best_sdice = scores['sdice/test_best']
            # sdice = max(sdice,best_sdice)
            ret_value.value = 0
        except subprocess.CalledProcessError:
            print(f'error in exp {exp}_{source}_{target}')
            shutil.copy(err_file.name, f'errs_and_outs/{exp}_{source}_{target}_logs_err.txt')
            shutil.copy(out_file.name, f'errs_and_outs/{exp}_{source}_{target}_logs_out.txt')

    my_devices.remove(device)


def run_cross_validation(experiments, combs, only_stats=False):
    if torch.cuda.is_available():
        nvmlInit()
    manager = multiprocessing.Manager()
    stats = {}
    running_now = []
    my_devices = manager.list()
    for combination in tqdm(combs):
        for exp in experiments:
            if exp not in stats:
                stats[exp] = {}

            source, target = combination
            base_res_dir = config.base_res_path
            src_ckpt_path = Path(config.base_res_path) / f'source_{source}' / 'pretrain' / 'best_model.pth'
            if not src_ckpt_path.exists():
                if only_stats:
                    continue
                curr_device = find_available_device(my_devices, running_now,'pretrain')
                print(f'training on source {source} to create {src_ckpt_path}')
                cmd = f'python train_gta2cityscapes_multi.py --source {source} --target {source} --mode pretrain --gpu {curr_device} >  errs_and_outs/pretrain{source}_logs_out.txt 2> errs_and_outs/pretarin{source}_logs_err.txt'
                print(cmd)
                my_devices.append(curr_device)
                subprocess.run(
                    cmd,
                    shell=True, check=True)
                my_devices.remove(curr_device)
            scores_path = f'{base_res_dir}/source_{source}_target_{target}/{exp}/scores_end.json'
            if not os.path.exists(scores_path):
                if only_stats:
                    continue
                curr_device = find_available_device(my_devices, running_now,exp)
                exp_dir_path = f'{base_res_dir}/source_{source}_target_{target}/{exp}'
                if os.path.exists(exp_dir_path):
                    shutil.rmtree(exp_dir_path, ignore_errors=True)
                print(f'lunch on source {source} target {target} exp {exp}')
                ret_value = multiprocessing.Value("d", 0.0, lock=False)
                p = Process(target=run_single_exp,
                            args=(exp, curr_device, source, target,scores_path, my_devices, ret_value))
                running_now.append([(exp, f's_{source} t_{target}'), p, ret_value])
                p.start()
                time.sleep(5)
            else:
                print(f'loading exists on source {source} target {target} exp {exp}')
                # scores =json.load(open(scores_path))
                sdice = 0#scores['sdice/test']
                best_sdice = 0#scores['sdice/test_best']
                sdice = max(sdice,best_sdice)
                stats[exp][f's_{source} t_{target}'] = sdice
    still_running = running_now
    while still_running:
        still_running = []
        places = []
        for place, p, ret_value in tqdm(running_now, desc='finishing running now'):
            p.join(timeout=0)
            if p.is_alive():
                still_running.append((place, p, ret_value))
                places.append(place)

            else:
                stats[place[0]][place[1]] = ret_value.value
        running_now = still_running
        print(places)
        if running_now:
            time.sleep(600)
    print(stats)
    json.dump(stats, open('all_stats.json', 'w'))
    return stats


def main():
    experiments = ['clustering_finetune','their','adaBN']
    combs = list(itertools.permutations(range(6), 2))
    random.shuffle(combs)
    # combs = [(1,3),(0, 4), (3, 1), (2, 5), (2, 3)]
    run_cross_validation(only_stats=False, experiments=experiments, combs=combs)


if __name__ == '__main__':

    config = CC359BaseConfig()
    main()
