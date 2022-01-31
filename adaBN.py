import argparse
import json
import random
from pathlib import Path
from torch.utils import data
import numpy as np
import torch
from torch import nn

from dpipe.io import load

from configs import *
from tqdm import tqdm

from utils import get_dice,get_sdice



from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from model.deeplab_multi import DeeplabMulti
from utils.loss import CrossEntropy2d


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.tensor(label.long()).to(gpu)
    criterion = CrossEntropy2d().to(gpu)

    return criterion(pred, label)

def run_adaBN(source, target, device, metric):

    ckpt_path = Path(config.base_res_path) / f'source_{source}' / 'pretrain' / 'best_model.pth'
    model = DeeplabMulti(num_classes=2,n_channels=config.n_channels)
    model.load_state_dict(torch.load(ckpt_path,map_location='cpu'))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()

    if config.msm:
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{target}/train_ids.json'))
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{target}/test_ids.json'))
    else:
        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{target}/train_ids.json'))
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{target}/test_ids.json'),slicing_interval=1)
    targetloader = data.DataLoader(target_ds, batch_size=config.batch_size, shuffle=True)
    model.train()
    interp = nn.Upsample(size=(config.input_size[1], config.input_size[0]), mode='bilinear',align_corners=True)
    for i in range(4):
        with torch.no_grad():
            for batch in tqdm(targetloader,desc='running train loader'):
                images, labels = batch
                images = torch.tensor(images).to(device)

                pred1, pred2 = model(images)
                pred1 = interp(pred1)
                pred2 = interp(pred2)

                # loss_seg1 = loss_calc(pred1, labels, args.gpu)
                loss_seg2 = loss_calc(pred2, labels, device)
                print(loss_seg2)
    if config.msm:
        sdice_test = get_dice(model,test_ds,device,config,interp)
    else:
        sdice_test = get_sdice(model,test_ds,device,config,interp)
    p1 = Path(f'{config.base_res_path}/source_{source}_target_{target}/adaBN/')
    p1.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),f'{config.base_res_path}/source_{source}_target_{target}/adaBN/model.pth')
    json.dump({"sdice/test": sdice_test, "sdice/test_best":sdice_test},open(p1 / 'scores.json','w'))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--device")
    cli.add_argument("--source")
    cli.add_argument("--target")

    opts = cli.parse_args()
    if opts.msm:
        assert opts.target ==opts.source
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    metric = 'sdice_score' if not config.msm else 'dice'
    run_adaBN(source=opts.source, target=opts.target, device=opts.device, metric=metric)


if __name__ == '__main__':
    config = MsmConfig()
    main()

