import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):
    """
    Ordinal loss as defined in the paper "DORN for Monocular Depth Estimation".
    """

    def __init__(self):
        super(OrdinalLoss, self).__init__()

    def forward(self, pred_softmax, target_labels):
        """
        :param pred_softmax:    predicted softmax probabilities P
        :param target_labels:   ground truth ordinal labels
        :return:                ordinal loss
        """
        N, C, H, W = pred_softmax.size() # C - number of discrete sub-intervals (= number of channels)

        K = torch.zeros((N, C, H, W), dtype=torch.int).cuda()
        for i in range(C):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).cuda()

        mask = (K <= target_labels).detach()
        
        loss = pred_softmax[mask].clamp(1e-8, 1e8).log().sum() + (1 - pred_softmax[~mask]).clamp(1e-8, 1e8).log().sum()
        loss /= -N * H * W
        return loss