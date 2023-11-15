# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):  # size_average=True, reduce=True
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
#         new_label = label.unsqueeze(0)
#         new_label = new_label.repeat(batch_size, 1)
        new_label = torch.eye(batch_size).half().to(device)
        probs2 = F.softmax(new_label, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss