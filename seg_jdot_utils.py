import torch
from scipy.spatial.distance import cdist
import ot
from torch.nn import BCEWithLogitsLoss

from metric_utils import dice_torch
from utils.loss import CrossEntropy2d


def dice_coefficient_loss(y_true, y_pred):
    '''
    Dice coefficient loss function.
    :param y_true:
    :param y_pred:
    :return:
    '''
    return 1-dice_torch(y_true, y_pred,smooth=1.)

def euclidean_dist(x,y):
    """
    Pairwise euclidean distance.
    :param x:
    :param y:
    :return: A matrix of size n_batch*n_batch where each entry represent the euclidean distance between one source sample and one target sample.
    """
    temp = torch.cdist(x.unsqueeze(0).contiguous(),y.unsqueeze(0).contiguous())[0]
    return temp
    # bs = x.shape[0]
    # dist = torch.reshape(torch.sum(torch.square(x), 1), (-1, 1)).repeat(1,bs)
    # dist += torch.reshape(torch.sum(torch.square(y), 1), (1, -1))
    # dist -= 2.0*torch.inner(x, y)

    return torch.sqrt(dist)

def deep_jdot_loss_euclidean(labels_source, prediction_source,prediction_target,gamma,jdot_beta,only_seg =False):
    # source_loss = dice_coefficient_loss(labels_source, prediction_source)
    source_loss = CrossEntropy2d()(prediction_source,labels_source.long())
    if only_seg:return source_loss
    target_loss = euclidean_dist(torch.flatten(labels_source,1), torch.flatten(prediction_target,1))
    return source_loss + jdot_beta * torch.sum(gamma * target_loss)

def distance_loss(pred_source,pred_target,gamma,jdot_alpha):
    '''
    Representation alignement loss function.
    Dif is the pairwise euclidean distance between the source samples and the target samples.

    :param y_true:
    :param y_pred:
    :return: \alpha * sum(\gamma*dif)
    The more two samples are connected by gamma (source and target) the more the representation should be similar.
    '''
    dif = euclidean_dist(torch.flatten(pred_source,1), torch.flatten(pred_target,1))
    return jdot_alpha * torch.sum(gamma*dif)


def compute_gamma(features_source,features_target,labels_source,pred_target,config,jdot_alpha,jdot_beta):
    '''
    Function to compute the OT between the target and source samples.
    :return:Gamma the OT matrix
    '''
    # Reshaping the samples into vectors of dimensions number of modalities * patch_dimension.
    # train_vecs are of shape (batch_size, d)

    train_vec_source = features_source.flatten(1).detach().cpu().numpy()

    train_vec_target = features_target.flatten(1).detach().cpu().numpy()
    # Same for the ground truth but the GT is the same for both modalities
    truth_vec_source = labels_source.flatten(1).detach().cpu().numpy()
    # We don't have information on target labels
    pred_vec_target = pred_target.flatten(1).detach().cpu().numpy()

    # Compute the distance between samples and between the source_truth and the target prediction.
    C0 = cdist(train_vec_source, train_vec_target, metric="sqeuclidean")
    C1 = cdist(truth_vec_source, pred_vec_target, metric='sqeuclidean')
    C = (jdot_alpha.detach().cpu().numpy()*C0+jdot_beta.detach().cpu().numpy()*C1)
    # Computing gamma using the OT library
    gamma = ot.emd(ot.unif(config.source_batch_size), ot.unif(config.source_batch_size), C,numItermax=10000000)
    return torch.tensor(gamma)