import torch
import torch.nn.functional as F
import torch.nn as nn

'''
该文件包含KLD损失以及标准化版本
分别对应
Distilling the Knowledge in a Neural Network 2015 NIPS (Hiton 原版)
Logit Standardization in Knowledge Distillation 2024 CVPR (标准化)
'''

def normalize(logit):
    mean = logit.mean(dim=1, keepdims=True)
    stdv = logit.std(dim=1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kld_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    # 调整形状：合并 (B, H, W) 维度，保留 C 作为类别维度
    B, C, H, W = logits_student.shape
    logits_student = logits_student.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    logits_teacher = logits_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")
    loss_kd *= temperature**2
    return loss_kd


if __name__ == '__main__':
    feature_s = torch.randn(2, 64, 20, 15)
    feature_t = torch.randn(2, 64, 20, 15)
    shape = (2, 480, 640)
    label = torch.randint(0, 9, size=shape, dtype=torch.int)