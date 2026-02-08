from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F



class PixelContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07

        self.max_samples = 1024
        self.max_views = 10

    def _hard_anchor_sampling(self, X, y_hat, y):  #  x(特征), y_har(标签)，y(预测)
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        # 像素采样
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            # if ii <= kd_loss: continue   # 不采样背景类
            this_q = Q[ii, :cache_size, :]   # 采样第i个类别的所有样本

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q  # x_的前cache_size个样本都是第i类的
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii   # y_的前cache_size行为ii
            sample_ptr += cache_size

        # 最后得到的X_为((classes-kd_loss)*cache_size, feature_dim), y_为((classes-kd_loss) * cache_size)
        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]  # 采样的类别数，采样的个数
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        # logits = anchor_dot_contrast
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # 当从memory bank中采样时无需去掉对角线
        # logits_mask = torch.ones_like(mask).scatter_(kd_loss,
        #                                              torch.arange(anchor_num * anchor_count).view(-kd_loss, kd_loss).cuda(),
        #                                              0)
        # mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # exp_logits即自身, neg_logits即负样本之和
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-4)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_mean = loss.mean()

        return loss_mean

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.unsqueeze(1).float().clone()
        predict = predict.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)  # (B, 300)
        predict = predict.contiguous().view(batch_size, -1)  # (B, 300)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # (B, 300, 256)
        feats = nn.functional.normalize(feats, p=2, dim=2)
        # sample
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)  # 输入特征，标签和预测

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss


class ContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "sem" in preds
        assert "embed" in preds

        seg = preds['sem']
        embedding = preds['embed']
        pixel_queue = preds['pixel_queue']

        _, predict = torch.max(seg, 1)  # 获取预测图
        loss_contrast = self.contrast_criterion(embedding, target, predict, pixel_queue)

        return loss_contrast

if __name__ == '__main__':
    x = torch.randn(2, 128, 640, 480)
    embed = torch.randn(2, 128, 20, 15)
    pixel_queue = torch.randn(6, 300, 128)
    inputs = {'sem': x, 'embed': embed, 'pixel_queue': pixel_queue}
    y = torch.randn(2, 640, 480)
    model = ContrastLoss()
    loss = model(inputs, y)
    # x = torch.randn(2, 3)
    # y = F.normalize(x, p=2, dim=0)
    # z = F.normalize(x, p=2, dim=kd_loss)
    # print(x)
    # print(y)
    # print(z)



    # torch.set_printoptions(profile="full")
    # x = torch.randn(1800, kd_loss)
    # print(x)
    # print(loss)
    # # x = kd_loss
    # y = x
    # y = 2
    # print(x)
    # print(y)
