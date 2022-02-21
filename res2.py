import cv2
import numpy as np
import torch
from dpipe.io import load
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from matplotlib.patches import Polygon

from dataset.msm_dataset import MultiSiteMri
from metric_utils import dice, _connectivity_region_analysis
from model.unet import UNet2D
mode_to_color = {'pretrain':'red','adaBN':'orange','their1':'yellow','msm1newnormdist':'green'}
model = UNet2D(3,n_chans_out=2)
# y = ['114_10','121_14','114_11','129_13','101_9','100_8']
# y=  ['111_8','114_11','114_10']
y = ['101_3','111_10','114_11']
dss = [[],[],[],[],[]]
target_ds = MultiSiteMri(load('/home/dsi/shaya/unsup_splits_msm/site_1/train_ids.json'),yield_id=True)
for i,mode in tqdm(enumerate(['pretrain','their1','adaBN','msm1newnormdist'])):
    if mode == 'pretrain':
        ckpt_path = '/home/dsi/shaya/unsup_resres_msm/source_1/pretrain/best_model.pth'
    else:
        ckpt_path = f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/{mode}/best_model.pth'
    state_dict = torch.load(ckpt_path,map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k]=v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    targetloader = data.DataLoader(target_ds, batch_size=1)
    slc = []
    for images ,segs,ids,slices in targetloader:
        if int(ids[0]) == 105:
            continue
        if str(int(ids[0]))+"_"+ str(int(slices[0])) not in y:
            continue
        images = Variable(images)
        if torch.count_nonzero(segs[0]) ==0:
            continue
        _, output = model(images)
        output = output.cpu().data.numpy()
        output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
        output =         _connectivity_region_analysis(output)
        d  = dice(segs,output)

        # plt.imsave(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/img.png',np.array(images[0][1].detach().cpu()), cmap='gray')

        plt.figure(1)
        plt.axis('off')
        plt.imshow(np.array(images[0][1].detach().cpu()), cmap='gray')
        coords,_ = cv2.findContours(segs[0][0].cpu().numpy().astype(np.uint8),mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        coords = coords[0][:,0,:]
        xs= []
        ys = []
        for trtr in range(coords.shape[0]):
            xs.append(coords[trtr][0])
            ys.append(coords[trtr][1])
        plt.plot(xs,ys,color='cyan')

        # plt.imshow(ttt, cmap='jet', alpha=0.25)
        plt.savefig(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/seg_{str(int(ids[0]))+"_"+ str(int(slices[0]))}.png',bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.figure(2)
        plt.axis('off')
        plt.imshow(np.array(images[0][1].detach().cpu()), cmap='gray')
        coords,_ = cv2.findContours(output[0].astype(np.uint8),mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        coords = coords[0][:,0,:]
        xs= []
        ys = []
        for trtr in range(coords.shape[0]):
            xs.append(coords[trtr][0])
            ys.append(coords[trtr][1])
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs,ys,color=mode_to_color[mode])
        plt.savefig(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/{mode}_{str(int(ids[0]))+"_"+ str(int(slices[0]))}.png',bbox_inches='tight')
        plt.cla()
        plt.clf()
        # Polygon(coords)
        # plt.imshow(np.array(images[0][1].detach().cpu()), cmap='gray')

        # plt.imsave(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/seg.png',np.array(segs[0][0].detach().cpu()), cmap='gray')
        # plt.imsave(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/{mode}.png',np.array(output[0]), cmap='gray')
        if mode == 'pretrain':
            dss[4].append(str(int(ids[0]))+"_"+ str(int(slices[0])))
        dss[i].append(d)
        slc.append(str(int(ids[0]))+"_"+ str(int(slices[0])))

# for yy in y:
#     fig = plt.figure()
#     images = []
#     for m_i,m in enumerate(['seg','pretrain','their1','adaBN','msm1newnormdist']):
#         im1 = cv2.imread(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/{m}_{yy}.png')
#         fig.add_subplot(2,3,m_i+1)
#         plt.imshow(im1)
#         plt.axis('off')
#         # plt.title(m)
#     plt.savefig(f'/home/dsi/shaya/unsup_resres_msm/source_1_target_1/all_{yy}.png')



best = -1
best_slc = None
bb = None
x0,x1,x2,x3,x4 =  dss
rr = []
for i in range(len(x1)):
    if x3[i]< 0.85:
        continue
    curr =np.min([x3[i]-x2[i],x3[i]-x1[i],x3[i]-x0[i]])
    rr.append((curr,(x0[i],x1[i],x2[i],x3[i],x4[i])))
    # if curr>best:
    #     best = curr
    #     bb = (x0[i],x1[i],x2[i],x3[i],x4[i])
    #     best_slc  = x4[i]
# print(bb)
# print(best)
# print(best)
print(sorted(rr,key=lambda x:x[0],reverse=True)[:15])
# print(sorted(dss,key=lambda x: ,reverse=True))

# plt.imsave(im_path,  np.array(img[1].detach().cpu()), cmap='gray')

x=[ (0.1437289279274716, (0.5741000467508182, 0.5570668243642815, 0.5741000467508182, 0.7178289746782898, '129_13')), (0.11911883174567184, (0.7856132552030713, 0.7716455696202532, 0.7856132552030713, 0.9047320869487432, '101_9')), (0.10882072679657273, (0.7360475188714268, 0.7312984738553916, 0.7360475188714268, 0.8448682456679996, '121_14')), (0.09854222430840431, (0.7639442231075697, 0.7584291947639825, 0.7639442231075697, 0.862486447415974, '100_8')), (0.09574638248628042, (0.6929029405261994, 0.6965250965250965, 0.6929029405261994, 0.792271479011377, '104_10')), (0.09038514257452668, (0.7167450871851647, 0.74800172823504, 0.7167450871851647, 0.8383868708095666, '121_12')), (0.08622585510631053, (0.8133367884598773, 0.8179487179487179, 0.8133367884598773, 0.9041745730550285, '101_4')), (0.07974290527754235, (0.6796586422111833, 0.7781569965870307, 0.6796586422111833, 0.8578999018645731, '111_13')), (0.07481885802383759, (0.6951231289232255, 0.6474705058988203, 0.6951231289232255, 0.7699419869470631, '100_6')), (0.07204619697687908, (0.8328067403896787, 0.8341846758349706, 0.8328067403896787, 0.9062308728118497, '111_10')), (0.06637818742947377, (0.7448300241560125, 0.7722415795586527, 0.7448300241560125, 0.8386197669881265, '110_10')), (0.06619865248615275, (0.7459815546772068, 0.7359054386081413, 0.7459815546772068, 0.8121802071633596, '111_8')), (0.0637398321920517, (0.6942948567860766, 0.7182041066503217, 0.6942948567860766, 0.7819439388423735, '122_8'))]
