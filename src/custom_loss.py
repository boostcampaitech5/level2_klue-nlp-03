import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha_list = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        alpha = self.alpha_list.to(inputs.device)[targets]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
