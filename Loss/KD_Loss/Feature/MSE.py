import torch.nn.functional as F
import torch
import torch.nn as nn

def MSELoss(student, teacher):
    # student = F.softmax(student, dim=kd_loss)
    # teacher = F.softmax(teacher, dim=kd_loss)
    # loss = (student - teacher).pow(2).sum(dim=kd_loss).mean()
    loss = (student - teacher).pow(2).mean()
    return loss


class train_Loss(nn.Module):

    def __init__(self):
        super(train_Loss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        self.class_weight_binary = torch.from_numpy(np.array([2.0197, 2.9765])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4584, 18.7187])).float()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        # self.dice_loss = DiceLoss(mode='multiclass', ignore_index=-kd_loss)
        # 这部分用于特征对齐,放在模型外部
        self.connector = nn.ModuleList([nn.Sequential(nn.Conv2d(40, 80, 1),
                                                    nn.BatchNorm2d(80),
                                                    nn.ReLU()),
                                       nn.Sequential(nn.Conv2d(80, 160, 1),
                                                      nn.BatchNorm2d(160),
                                                      nn.ReLU()),
                                       nn.Sequential(nn.Conv2d(160, 320, 1),
                                                      nn.BatchNorm2d(320),
                                                      nn.ReLU()),
                                       nn.Sequential(nn.Conv2d(320, 640, 1),
                                                      nn.BatchNorm2d(640),
                                                      nn.ReLU())])

    def forward(self, predict_student, targets, predict_teacher, f_s, f_t):
        B, C, H, W = predict_student.size()
        sem_s = predict_student
        sem_t = predict_teacher
        semantic_gt, boundary_gt, binary_gt = targets
        loss_hard = self.semantic_loss(sem_s, semantic_gt) + self.dice_loss(sem_s, semantic_gt)
        loss_kd = 0
        divider = [8, 4, 2, 1]
        for i in range(4):
            loss_f = MSELoss(self.connector[i](f_s[i]), f_t[i]) / divider[i]
            loss_kd = loss_kd + loss_f
        # loss_kd = kld_loss(sem_s, sem_t, kd_loss, False)

        loss = loss_hard + loss_kd

        return loss, loss_hard, loss_kd