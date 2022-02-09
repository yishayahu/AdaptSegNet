import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BatchNorm2d


def freeze_model(model, exclude_layers=('inconv', )):
    for name, param in model.named_parameters():
        requires_grad = False
        for l in exclude_layers:
            if l in name:
                requires_grad = True
        param.requires_grad = requires_grad

def freeze_norm_stats(model, exclude_layers=('inconv', )):
    for name, m in model.named_modules():
        if isinstance(m, BatchNorm2d):
            m.eval()
            m.track_running_stats = False
            for l in exclude_layers:
                if l in name:
                    m.train()
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        assert self.size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        if target.dim() != 3:
            assert target.dim() == 4
            target = target.squeeze(1)
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight,reduction='mean')
        return loss
