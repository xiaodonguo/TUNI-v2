import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['CriterionKD']

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1, ignore_index=None):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
    def forward(self, pred, soft, gt=None):
        B, C, h, w = soft.size()
        scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        if gt is not None and self.ignore_index is not None:
            mask = (gt != self.ignore_index).view(-1)  # 展平为 1D mask
            p_s = p_s[mask]
            p_t = p_t[mask]
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return loss


class CriterionKD_weight(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1, ignore_index=None, class_weights=None):
        super(CriterionKD_weight, self).__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.class_weights = nn.Parameter(class_weights, requires_grad=False)


    def forward(self, pred, soft, gt=None):
        B, C, h, w = soft.size()
        scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)

        # 计算每个像素的KL散度 [B*h*w]
        kl_per_pixel = F.kl_div(p_s, p_t, reduction='none').sum(dim=1)

        # 应用类别平衡权重（仅在提供gt时）
        if gt is not None and self.class_weights is not None:
            # 创建mask处理ignore_index
            if self.ignore_index is not None:
                mask = (gt != self.ignore_index).view(-1)
                kl_per_pixel = kl_per_pixel[mask]
                valid_gt = gt.view(-1)[mask]
            else:
                valid_gt = gt.view(-1)

            if len(valid_gt) > 0:  # 确保有有效像素
                # 应用类别权重
                pixel_weights = self.class_weights[valid_gt]
                weighted_kl = (kl_per_pixel * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)
                loss = weighted_kl * (self.temperature ** 2)
            else:
                loss = torch.tensor(0.0, device=pred.device)

        else:
            # 回退到原始实现
            if gt is not None and self.ignore_index is not None:
                mask = (gt != self.ignore_index).view(-1)
                kl_per_pixel = kl_per_pixel[mask]
            if len(kl_per_pixel) > 0:
                loss = kl_per_pixel.mean() * (self.temperature ** 2)
            else:
                loss = torch.tensor(0.0, device=pred.device)

        return loss