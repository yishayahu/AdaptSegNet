import numpy as np
import torch
from scipy import ndimage
from torch.utils import data
from tqdm import tqdm
import surface_distance.metrics as surf_dc
from torch.autograd import Variable
def sdice(a, b, spacing, tolerance=1):
    if np.count_nonzero(a) == 0:
        non_zeros = np.count_nonzero(b)
        if  non_zeros == 0:
            return 1
    surface_distances = surf_dc.compute_surface_distances(b, a, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def _connectivity_region_analysis(mask):
    label_im, nb_labels = ndimage.label(mask)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im
def dice(gt,pred):
    gt = gt.squeeze(1)
    g = np.zeros(gt.shape)
    p = np.zeros(pred.shape)
    g[gt == 1] = 1
    p[pred == 1] = 1
    return (2*np.sum(g*p))/(np.sum(g)+np.sum(p))

def get_sdice(model,ds,device,config,interp):
    loader= data.DataLoader(ds,batch_size=config.batch_size, shuffle=False)
    model.eval()
    sdice_for_id={}
    with torch.no_grad():

        for images,segs,ids,_ in tqdm(loader,desc='running test loader'):
            _, output2 = model(images.to(device))
            output = interp(output2).cpu().data.numpy()
            output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            segs = segs.squeeze(1).numpy().astype(bool)
            for out1,seg1,id1 in zip(output,segs,ids):
                out1 = np.expand_dims(out1, 0)
                seg1 = np.expand_dims(seg1, 0)
                if id1 not in sdice_for_id:
                    sdice_for_id[id1] = []
                curr_sdice = sdice(seg1,out1,ds.spacing_loader(id1))
                assert not np.any(np.isnan(curr_sdice))
                sdice_for_id[id1].append(curr_sdice)
    all_sdices = [np.mean(sdices) for sdices in sdice_for_id.values()]
    return float(np.mean(all_sdices))
def get_dice(model,ds,device,config,interp):
    model.eval()
    dices = []
    with torch.no_grad():
        for id1,images in tqdm(ds.patches_Allimages.items(),desc='running val or test loader'):
            segs = ds.patches_Allmasks[id1]
            images = Variable(torch.tensor(images)).to(device)
            _, output2 = model(images)
            output = interp(output2).cpu().data.numpy()
            output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            output = _connectivity_region_analysis(output)
            dices.append(dice(segs,output))
    return float(np.mean(dices))