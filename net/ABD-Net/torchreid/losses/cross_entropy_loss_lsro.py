from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLossLSRO(nn.Module):
    """Cross entropy loss with label smoothing regularization for outliers.

    Reference:
    Zhang et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. CVPR 2017.

    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLossLSRO, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def apply_loss(self, inputs, targets):

        log_probs = self.logsoftmax(inputs)

        targets = torch.zeros(log_probs.size())
        for idx, label in enumerate(targets):
            print(idx, label)
            if label == -2:
                targets[idx, :] = torch.ones(log_probs.size()[1]) / self.num_classes
            else:
                targets[idx, label] = 1
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

    def _forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size)
        """

        if not isinstance(inputs, tuple):
            inputs_tuple = (inputs,)
        else:
            inputs_tuple = inputs

        return sum([self.apply_loss(x, targets) for x in inputs_tuple]) / len(inputs_tuple)

    def forward(self, inputs, targets):

        return self._forward(inputs[1], targets)
