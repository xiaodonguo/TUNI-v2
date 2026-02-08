import torch
import torch.nn as nn
from Semantic_Segmentation_Street_Scenes.toolbox import setup_seed
from torch.nn import functional as F
setup_seed(33)


def MSELoss(student, teacher):
    student = F.softmax(student, dim=1)
    teacher = F.softmax(teacher, dim=1)
    loss = (student - teacher).pow(2).sum(dim=1).mean()
    return loss

class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, f_s, f_t):
        b = f_s.shape[0]
        c = f_s.shape[1]
        f_s = f_s.view(b, c, -1)
        f_t = f_t.view(b, c, -1)
        G_s = F.normalize(f_s, dim=2)
        G_t = F.normalize(f_t, dim=2)
        G_s = torch.matmul(G_s, G_s.permute(0, 2, 1))
        print(G_s)
        G_t = torch.matmul(G_t, G_t.permute(0, 2, 1))
        print(G_t)
        G_diff = (G_t - G_s).pow(2).mean()
        return G_diff

if __name__ == '__main__':

    s = torch.randn(2, 9, 480, 640)
    s = torch.sigmoid(s)
    t = torch.randn(2, 9, 480, 640)
    t = torch.sigmoid(t)
    loss1 = Similarity()
    l2 = MSELoss(s, t)
    l1 = loss1(s, t)
    print(l1)
    print(l2)
