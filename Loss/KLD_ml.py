__all__ = ['CriterionKD']

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriterionKD(nn.Module):
    '''
    Knowledge distillation loss with selective guidance
    Only apply KD loss on regions where teacher is correct but student is wrong
    '''

    def __init__(self, temperature=1):
        super(CriterionKD, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft, target):
        B, C, h, w = soft.size()

        # Reshape tensors for processing
        scale_pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        scale_soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
        scale_target = target.view(-1)  # Flatten target

        # Get predictions
        pred_softmax = F.softmax(scale_pred, dim=1)
        soft_softmax = F.softmax(scale_soft, dim=1)

        pred_argmax = torch.argmax(pred_softmax, dim=1)
        soft_argmax = torch.argmax(soft_softmax, dim=1)

        # Create masks for different regions
        soft_correct_mask = (soft_argmax == scale_target)
        pred_correct_mask = (pred_argmax == scale_target)

        # Target region: teacher is correct but student is wrong
        target_mask = (soft_correct_mask & ~pred_correct_mask)

        # If no target regions found, return zero loss
        if target_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        num_target_pixels = target_mask.sum().float()
        # Apply temperature scaling only to target regions
        p_s = F.log_softmax(scale_pred[target_mask] / self.temperature, dim=1)
        p_t = F.softmax(scale_soft[target_mask] / self.temperature, dim=1)

        # Calculate KL divergence only on target regions
        kd_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature ** 2)

        return kd_loss