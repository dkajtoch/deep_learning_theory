import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CatapultMSEClassificationLoss(nn.MSELoss):
    """MSE loss for classification tasks.

    It uses non-standard normalization by multiplying the loss by 0.5.
    That's why it's called CatapultMSE.

    See Also:
        https://arxiv.org/abs/2003.02218, Supplementary material A.
    """

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        one_hot = F.one_hot(targets, num_classes=inputs.shape[-1]).float()
        return super().forward(inputs, one_hot) * 0.5
