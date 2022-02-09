import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path
import PIL
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import torch
import torch.nn as nn
import wandb
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from dpipe.io import load
from tqdm import tqdm
from metric_utils import get_sdice,get_dice
from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from model.unet import UNet2D
from utils.loss import CrossEntropy2d, freeze_model
import fnmatch
from configs import *




def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """

    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                   for name in fnmatch.filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = tensor.detach().cpu()
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze()
    else:
        assert False
    return PIL.Image.fromarray(tensor)

ITER_SIZE = 1
NUM_WORKERS = 1
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234

SAVE_NUM_IMAGES = 2

WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--mode", type=str,default='pretrain')
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument('--exp_name',default='')
    parser.add_argument('--msm',action='store_true')
    return parser.parse_args()


args = get_arguments()
if args.msm:
    if args.mode == 'clustering_finetune':
        config = MsmConfigFinetuneClustering()
    elif args.mode == 'pretrain':
        config = MsmPretrainConfig()
    else:
        assert args.mode == 'their'
        raise NotImplemented()
else:
    if 'debug' in args.exp_name:
      config = DebugConfigCC359()
    elif args.mode == 'clustering_finetune':
        config = CC359ConfigFinetuneClustering()
    elif args.mode == 'pretrain':
        config = CC359ConfigPretrain()
    else:
        assert args.mode == 'their'
        config = CC359ConfigTheir()
if args.exp_name == '':
    args.exp_name  = args.mode +'_'+ str(args.source) + '_' + str(args.target)
best_sdice = -1
low_source_sdice = 1.1

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(gpu)
    criterion = CrossEntropy2d().to(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(config.lr, i_iter, config.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, config.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def after_step(num_step,val_ds,test_ds,model,val_ds_source):


    metric = 'dice' if config.msm else 'sdice'
    global best_sdice
    global low_source_sdice
    if num_step % config.save_pred_every == 0 and num_step!= 0:
        if config.msm:
            sdice1 = get_dice(model,val_ds,args.gpu,config)
        else:
            sdice1 = get_sdice(model,val_ds,args.gpu,config)
        if val_ds_source is not None:
            if config.msm:
                sdice_source = get_dice(model,val_ds_source,args.gpu,config)
            else:
                sdice_source = get_sdice(model,val_ds_source,args.gpu,config)
            if sdice_source < low_source_sdice:
                low_source_sdice = sdice_source
                torch.save(model.state_dict(), config.exp_dir / f'low_source_model.pth')
            wandb.log({f'{metric}/val_source':sdice_source},step=num_step)
        wandb.log({f'{metric}/val':sdice1},step=num_step)
        print(f'{metric} is ',sdice1)
        print ('taking snapshot ...')

        if sdice1 > best_sdice:
            best_sdice = sdice1
            torch.save(model.state_dict(), config.exp_dir / f'best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'model.pth')
    if num_step  == config.num_steps - 1 or num_step == 0:
        title = 'end' if num_step != 0 else 'start'
        scores = {}
        if config.msm:
            sdice_test = get_dice(model,test_ds,args.gpu,config)
        else:
            sdice_test = get_sdice(model,test_ds,args.gpu,config)
        scores[f'{metric}_{title}/test'] = sdice_test
        if num_step != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'best_model.pth',map_location='cpu'))
            if config.msm:
                sdice_test_best = get_dice(model,test_ds,args.gpu,config)
            else:
                sdice_test_best =get_sdice(model,test_ds,args.gpu,config)
            scores[f'{metric}_{title}/test_best'] = sdice_test_best
            if val_ds_source is not None:
                model.load_state_dict(torch.load(config.exp_dir / f'low_source_model.pth',map_location='cpu'))
                if config.msm:
                    sdice_test_low_source = get_dice(model,test_ds,args.gpu,config)
                else:
                    sdice_test_low_source =get_sdice(model,test_ds,args.gpu,config)
                scores[f'{metric}_{title}/test_low_source_on_target'] = sdice_test_low_source

        wandb.log(scores,step=num_step)
        json.dump(scores,open(config.exp_dir/f'scores_{title}.json','w'))

def train_their(model,optimizer,scheduler,trainloader,targetloader,val_ds,test_ds,val_ds_source):
    trainloader_iter = iter(trainloader)
    targetloader_iter = iter(targetloader)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    # interp = nn.Upsample(size=(config.input_size[1], config.input_size[0]), mode='bilinear')
    # interp_target = nn.Upsample(size=(config.input_size[1], config.input_size[0]), mode='bilinear')
    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.to(args.gpu)

    model_D2.train()
    model_D2.to(args.gpu)
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()
    # labels for adversarial training
    source_label = 0
    target_label = 1





    for i_iter in range(config.num_steps):
        model.train()
        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source
            try:
                batch = trainloader_iter.next()
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = trainloader_iter.next()

            images, labels = batch
            images = Variable(images).to(args.gpu)
            _, pred2 = model(images)
            # pred1 = interp(pred1)
            # pred2 = interp(pred2)

            # loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            # loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

            # train with target
            try:
                batch = targetloader_iter.next()
            except StopIteration:
                targetloader_iter = iter(targetloader)
                batch = targetloader_iter.next()

            images, _ = batch
            images = Variable(images).to(args.gpu)

            _, pred_target2 = model(images)
            # pred_target1 = interp_target(pred_target1)
            # pred_target2 = interp_target(pred_target2)

            # D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            # loss_adv_target1 = bce_loss(D_out1,
            #                             Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(
            #                                 args.gpu))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).to(
                                            args.gpu))

            loss =  args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            # loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            # pred1 = pred1.detach()
            pred2 = pred2.detach()

            # D_out1 = model_D1(F.softmax(pred1))
            D_out2 = model_D2(F.softmax(pred2))

            # loss_D1 = bce_loss(D_out1,
            #                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).to(args.gpu))

            # loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            # loss_D1.backward()
            loss_D2.backward()

            # loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            # pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            # D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            # loss_D1 = bce_loss(D_out1,
            #                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).to(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).to(args.gpu))

            # loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            # loss_D1.backward()
            loss_D2.backward()

            # loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        scheduler.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, config.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))
        after_step(i_iter,model =model,val_ds=val_ds,test_ds=test_ds,val_ds_source=val_ds_source)
def train_pretrain(model,optimizer,scheduler,trainloader):
    if config.msm:
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'),yield_id=True,test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/test_ids.json'),yield_id=True,test=True)
    else:
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'),yield_id=True,slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/test_ids.json'),yield_id=True,slicing_interval=1)
    trainloader_iter = iter(trainloader)
    for i_iter in range(config.num_steps):
        model.train()
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train with source
            try:
                batch = trainloader_iter.next()
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = trainloader_iter.next()

            images, labels = batch
            images = Variable(images).to(args.gpu)

            _, pred = model(images)
            loss_seg = loss_calc(pred, labels, args.gpu)
            loss = loss_seg
            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy() / args.iter_size

        optimizer.step()
        scheduler.step()


        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}'.format(
                i_iter, config.num_steps, loss_seg_value))
        after_step(i_iter,model =model,val_ds=val_ds,test_ds=test_ds,val_ds_source=None)

def get_best_match_aux(distss):
    n_clusters = len(distss)
    print('n_clusterss',n_clusters)
    res = linear_sum_assignment(distss)[1].tolist()
    targets = [None] *n_clusters
    for x,y in enumerate(res):
        targets[y] = x
    return targets


def get_best_match(sc, tc):
    dists = np.full((sc.shape[0],tc.shape[0]),fill_value=np.inf)
    for i in range(sc.shape[0]):
        for j in range(tc.shape[0]):
            dists[i][j] = np.mean((sc[i]-tc[j])**2)
    best_match = get_best_match_aux(dists.copy())

    return best_match

def train_clustering(model,optimizer,scheduler,trainloader,targetloader,val_ds,test_ds,val_ds_source):
    freeze_model(model,exclude_layers = ['init_path', 'down','bottleneck.0','bottleneck.1','bottleneck.2','bottleneck.3.conv_path.0','out_path'])
    trainloader.dataset.yield_id = True
    targetloader.dataset.yield_id = True
    trainloader_iter = iter(trainloader)
    targetloader_iter = iter(targetloader)
    dist_loss_lambda = config.dist_loss_lambda
    n_clusters = config.n_clusters
    slice_to_cluster = None
    source_clusters = None
    target_clusters = None
    best_matchs = None
    best_matchs_indexes = None
    accumulate_for_loss = None
    if config.use_accumulate_for_loss:
        accumulate_for_loss = []
        for _ in range(n_clusters):
            accumulate_for_loss.append([])
    slice_to_feature_source = {}
    slice_to_feature_target = {}
    id_to_num_slices = load(config.id_to_num_slices)
    epoch_seg_loss = []
    epoch_dist_loss = []
    optimizer.zero_grad()
    for i_iter in tqdm(range(config.num_steps)):
        if config.use_adjust_lr:
            adjust_learning_rate(optimizer, i_iter)
        if i_iter == 0:
            if config.parallel_model:
                model.module.get_bottleneck = False
            else:
                model.get_bottleneck = False
            after_step(i_iter,val_ds,test_ds,model,val_ds_source)
            continue
        if config.parallel_model:
            model.module.get_bottleneck = True
        else:
            model.get_bottleneck = True
        if best_matchs is None:
            model.eval()
        else:
            model.train()
        if i_iter % config.epoch_every == 0 and i_iter != 0:
            trainloader_iter = iter(trainloader)
            targetloader_iter = iter(targetloader)
            source_clusters = []
            target_clusters = []
            if config.use_accumulate_for_loss:
                accumulate_for_loss = []
                for _ in range(n_clusters):
                    accumulate_for_loss.append([])
            for i in range(n_clusters):
                source_clusters.append([])
                target_clusters.append([])
            p = PCA(n_components=20,random_state=42)
            t = TSNE(n_components=2,learning_rate='auto',init='pca',random_state=42)
            points = []
            slices = []
            for id_slc,feat in slice_to_feature_source.items():
                points.append(feat)
                id1, slc_num = id_slc.split('_')
                if config.msm:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                else:
                    slc_num = int(slc_num)/ id_to_num_slices[id1]
                slices.append(slc_num)
            for id_slc,feat in slice_to_feature_target.items():
                points.append(feat)
                id1, slc_num = id_slc.split('_')
                if config.msm:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                else:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                slices.append(slc_num)
            points = np.array(points)
            points = points.reshape(points.shape[0],-1)
            print('doing tsne')
            points = p.fit_transform(points)
            if config.use_slice_num:
                slices = np.expand_dims(np.array(slices),axis=1)
                points = np.concatenate([points,slices],axis=1)
            points = t.fit_transform(points)
            source_points,target_points = points[:len(slice_to_feature_source)],points[len(slice_to_feature_source):]
            # source_points,target_points = points[:max(len(slice_to_feature_source),n_clusters)],points[-max(len(slice_to_feature_target),n_clusters):]
            k1 = KMeans(n_clusters=n_clusters,random_state=42)
            print('doing kmean 1')
            sc = k1.fit_predict(source_points)
            k2 = KMeans(n_clusters=n_clusters,random_state=42,init=k1.cluster_centers_)
            print('doing kmean 2')
            tc = k2.fit_predict(target_points)
            print('getting best match')
            best_matchs_indexes=get_best_match(k1.cluster_centers_,k2.cluster_centers_)
            slice_to_cluster = {}
            items = list(slice_to_feature_source.items())
            for i in range(len(slice_to_feature_source)):
                source_clusters[sc[i]].append(items[i][1])
                slice_to_cluster[items[i][0]] = sc[i]
            items = list(slice_to_feature_target.items())
            for i in range(len(slice_to_feature_target)):
                slice_to_cluster[items[i][0]] = tc[i]
            for i in range(len(source_clusters)):
                source_clusters[i] = np.mean(source_clusters[i],axis=0)
            best_matchs = []
            for i in range(len(best_matchs_indexes)):
                best_matchs.append(torch.tensor(source_clusters[best_matchs_indexes[i]]))

            cm = plt.get_cmap('gist_rainbow')
            cNorm  = mplcolors.Normalize(vmin=0, vmax=n_clusters-1)
            scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
            colors = []
            for i in range(n_clusters):
                colors.append(scalarMap.to_rgba(i))

            im_path_source =str(config.exp_dir /  f'{i_iter}_source.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            curr_colors = []
            curr_points_x = []
            curr_points_y = []
            for i, slc_name in enumerate(slice_to_feature_source.keys()):
                curr_points_x.append(source_points[i][0])
                curr_points_y.append(source_points[i][1])
                curr_colors.append(colors[slice_to_cluster[slc_name]])
            ax.scatter(curr_points_x,curr_points_y,marker = '.',c=curr_colors)
            plt.savefig(im_path_source)
            plt.cla()
            plt.clf()
            plt.close()
            im_path_target = str(config.exp_dir /  f'{i_iter}_target.png')

            fig = plt.figure()
            ax = fig.add_subplot()
            curr_colors = []
            curr_points_x = []
            curr_points_y = []
            for i, slc_name in enumerate(slice_to_feature_target.keys()):
                curr_points_x.append(target_points[i][0])
                curr_points_y.append(target_points[i][1])
                curr_colors.append(colors[best_matchs_indexes[slice_to_cluster[slc_name]]])
            ax.scatter(curr_points_x,curr_points_y,marker = '.',c=curr_colors)
            plt.savefig(im_path_target)
            plt.cla()
            plt.clf()
            plt.close()
            im_path_clusters = str(config.exp_dir /  f'{i_iter}_clusters.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            for i,(p,marker) in enumerate([(k1.cluster_centers_,'.'),(k2.cluster_centers_,'^')]):
                if i ==0:
                    ax.scatter(p[:,0],p[:,1],marker = marker,c=colors[:len(p)])
                else:
                    ax.scatter(p[:,0],p[:,1],marker = marker,c=[colors[best_matchs_indexes[i]] for i in range(len(p))])
            plt.savefig(im_path_clusters)
            plt.cla()
            plt.clf()
            plt.close()
            slice_to_feature_source = {}
            slice_to_feature_target = {}
            vizviz = {}
            log_log = {f'figs/source': wandb.Image(im_path_source),f'figs/target': wandb.Image(im_path_target),f'figs/cluster': wandb.Image(im_path_clusters)}
            wandb.log(log_log,step=i_iter)
        log_log = {}
        for sub_i in range(args.iter_size):
            try:
                batch = trainloader_iter.next()
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = trainloader_iter.next()

            images, labels,ids,slice_nums = batch
            images = Variable(images).to(args.gpu)

            _, pred,features = model(images)
            features = features.detach().cpu().numpy()
            for id1,slc_num,feature,img in zip(ids,slice_nums,features,images):
                slice_to_feature_source[f'{id1}_{slc_num}'] = feature
                if best_matchs is not None and f'{id1}_{slc_num}' in slice_to_cluster:
                    src_cluster = slice_to_cluster[f'{id1}_{slc_num}']
                    if f'source_{src_cluster}' not in vizviz or len(vizviz[f'source_{src_cluster}']) < 4:
                        if f'source_{src_cluster}' not in vizviz:
                            vizviz[f'source_{src_cluster}'] = []
                        vizviz[f'source_{src_cluster}'].append(None)
                        im_path =  str(config.exp_dir / f'source_{src_cluster}_{i_iter}_{len(vizviz[f"source_{src_cluster}"])}.png')
                        if img.shape[0] == 3:
                            plt.imsave(im_path,  np.array(img[1].detach().cpu()), cmap='gray')
                        else:
                            img = tensor_to_image(img)
                            img.save(im_path)
                        log_log[f'{src_cluster}/source_{len(vizviz[f"source_{src_cluster}"])}'] = wandb.Image(im_path)
            loss_seg = loss_calc(pred, labels, args.gpu)
            loss = loss_seg
            # proper normalization
            loss = loss / args.iter_size

            try:
                batch = targetloader_iter.next()
            except StopIteration:
                targetloader_iter = iter(targetloader)
                batch = targetloader_iter.next()
            images, labels,ids,slice_nums = batch
            images = Variable(images).to(args.gpu)

            _, __,features = model(images)
            # features = features.mean(1)
            dist_loss = torch.tensor(0.0,device=args.gpu)
            for id1,slc_num,feature,img in zip(ids,slice_nums,features,images):
                slice_to_feature_target[f'{id1}_{slc_num}'] = feature.detach().cpu().numpy()
                if best_matchs is not None and  f'{id1}_{slc_num}' in slice_to_cluster:
                    if config.use_accumulate_for_loss:
                        accumulate_for_loss[slice_to_cluster[f'{id1}_{slc_num}']].append(feature)
                    else:
                        dist_loss+= torch.mean(torch.abs(feature - best_matchs[slice_to_cluster[f'{id1}_{slc_num}']].to(args.gpu)))
                    src_cluster = best_matchs_indexes[slice_to_cluster[f'{id1}_{slc_num}']]
                    if f'target_{src_cluster}' not in vizviz or len(vizviz[f'target_{src_cluster}']) < 4:
                        if f'target_{src_cluster}' not in vizviz:
                            vizviz[f'target_{src_cluster}'] = []
                        vizviz[f'target_{src_cluster}'].append(None)
                        im_path =  str(config.exp_dir /  f'target_{src_cluster}_{i_iter}_{len(vizviz[f"target_{src_cluster}"])}.png')
                        if img.shape[0] == 3:
                            plt.imsave(im_path,  np.array(img[1].detach().cpu()), cmap='gray')
                        else:
                            img = tensor_to_image(img)
                            img.save(im_path)
                        log_log[f'{src_cluster}/target_{len(vizviz[f"target_{src_cluster}"])}'] = wandb.Image(im_path)
            if accumulate_for_loss is not None:
                use_dist_loss = False
                lens1 = [len(x) for x in accumulate_for_loss]
                if np.sum(lens1) > config.acc_amount:
                    use_dist_loss = True
                if use_dist_loss:
                    for i,features in enumerate(accumulate_for_loss):
                        if len(features) > 0:
                            features = torch.mean(torch.stack(features),dim=0)
                            dist_loss += torch.mean((features - best_matchs[i].to(args.gpu))**2)
                            accumulate_for_loss[i] = []
            dist_loss *= dist_loss_lambda
            if float(dist_loss) > 0:
                epoch_dist_loss.append(float(dist_loss))
            epoch_seg_loss.append(float(loss))
            losses_dict = {'seg_loss': loss,'dist_loss':dist_loss,'total':loss+dist_loss}
            if accumulate_for_loss is None:
                losses_dict['total'].backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                if use_dist_loss:
                    losses_dict['total'].backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                elif best_matchs is None:
                    pass
                    # losses_dict['seg_loss'].backward()
                    # optimizer.step()
                    # scheduler.step()
                    # optimizer.zero_grad()
                else:
                    losses_dict['seg_loss'].backward(retain_graph=True)
                    scheduler.step()
            log_log['seg_loss'] = float(np.mean(epoch_seg_loss))
            if epoch_dist_loss:
                log_log['dist_loss'] = float(np.mean(epoch_dist_loss))
            log_log['lr'] = float(scheduler.get_last_lr()[0])
            wandb.log(log_log,step=i_iter)

        if config.parallel_model:
            model.module.get_bottleneck = False
        else:
            model.get_bottleneck = False
        after_step(i_iter,val_ds,test_ds,model,val_ds_source)
def main():
    """Create the model and start the training."""

    input_size = config.input_size
    input_size_target = config.input_size

    cudnn.enabled = True


    # Create network
    model = UNet2D(config.n_channels)

    if args.mode != 'pretrain':
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) /f'source_{args.source}_target_{args.target}' /  args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) /f'source_{args.source}_target_{args.target}' /  args.mode

        ckpt_path = Path(config.base_res_path) / f'source_{args.source}' / 'pretrain' / 'best_model.pth'
        state_dict = torch.load(ckpt_path,map_location='cpu')
        if config.msm:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k]=v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        if config.msm:
            optimizer = optim.Adam(model.parameters(),
                                  lr=config.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)
    else:
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) /f'source_{args.source}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) /f'source_{args.source}' / args.mode

        if config.msm:
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)
    if config.sched:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.sched_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
    if config.exp_dir.exists():
        shutil.rmtree(config.exp_dir)
    config.exp_dir.mkdir(parents=True,exist_ok=True)
    json.dump(dataclasses.asdict(config),open(config.exp_dir/'config.json','w'))
    shutil.copytree('.',config.exp_dir / 'code',ignore=include_patterns('*.py'))
    model.train()
    if not torch.cuda.is_available():
        print('training on cpu')
        args.gpu = 'cpu'
        config.parallel_model = False
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2, 5])
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config.msm:
        assert args.source == args.target
        source_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/train_ids.json'))
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json'))
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/val_ids.json'),yield_id=True,test=True)
        val_ds_source = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'),yield_id=True,test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'),yield_id=True,test=True)
        project = 'adaptSegUNetMsm'
    else:
        source_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/train_ids.json')[:config.data_len])
        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json')[:config.data_len])
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'),yield_id=True,slicing_interval=1)
        val_ds_source = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'),yield_id=True,slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'),yield_id=True,slicing_interval=1)
        project = 'adaptSegUNet'
    if config.debug:
        wandb.init(
            project='spot3',
            id=wandb.util.generate_id(),
            name=args.exp_name,
            dir='..'
        )
    else:
        wandb.init(
            project=project,
            id=wandb.util.generate_id(),
            name=args.exp_name,
            dir='..'
        )
    trainloader = data.DataLoader(source_ds,batch_size=config.source_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
    targetloader = data.DataLoader(target_ds, batch_size=config.target_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)


    optimizer.zero_grad()

    if args.mode == 'pretrain':
        train_pretrain(model,optimizer,scheduler,trainloader)
    elif args.mode == 'clustering_finetune':
        train_clustering(model,optimizer,scheduler,trainloader,targetloader,val_ds,test_ds,val_ds_source)
    else:
        assert args.mode == 'their'
        train_their(model,optimizer,scheduler,trainloader,targetloader,val_ds,test_ds,val_ds_source)




if __name__ == '__main__':
    main()
