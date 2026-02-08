import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class MLPCAT(nn.Module):
    def __init__(self, in_channels, embed_dim, align_corners=False):
        super(MLPCAT, self).__init__()
        self.in_channels = in_channels
        in_channel1, in_channel2, in_channel3, in_channel4 = self.in_channels
        self.align_corners = align_corners
        embedding_dim = embed_dim
        self.linear1 = MLP(input_dim=in_channel1, embed_dim=embedding_dim)
        self.linear2 = MLP(input_dim=in_channel2, embed_dim=embedding_dim)
        self.linear3 = MLP(input_dim=in_channel3, embed_dim=embedding_dim)
        self.linear4 = MLP(input_dim=in_channel4, embed_dim=embedding_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )
    def forward(self, x):
        f1, f2, f3, f4 = x
        n, _, h, w = f4.shape
        _f4 = self.linear4(f4).permute(0, 2, 1).reshape(n, -1, f4.shape[2], f4.shape[3])
        _f4 = F.interpolate(_f4, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _f3 = self.linear3(f3).permute(0, 2, 1).reshape(n, -1, f3.shape[2], f3.shape[3])
        _f3 = F.interpolate(_f3, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _f2 = self.linear2(f2).permute(0, 2, 1).reshape(n, -1, f2.shape[2], f2.shape[3])
        _f2 = F.interpolate(_f2, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _f1 = self.linear1(f1).permute(0, 2, 1).reshape(n, -1, f1.shape[2], f1.shape[3])

        _c = self.linear_fuse(torch.cat([_f4, _f3, _f2, _f1], dim=1))

        return _c

class Decoder_DualMLP(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 embed_dim=512,
                 dropout_ratio=0.1,
                 num_classes=9,
                 ):

        super(Decoder_DualMLP, self).__init__()
        self.MLP1 = MLPCAT(in_channels=in_channels, embed_dim=embed_dim)
        self.linear_pred1 = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        if self.training:
            self.MLP2 = MLPCAT(in_channels=(channel // 2 for channel in in_channels),
                               embed_dim=embed_dim)

            self.linear_pred2 = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout1 = nn.Dropout2d(dropout_ratio)
            self.dropout2 = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, inputs1, inputs2):
        _c1 = self.MLP1(inputs1)
        _c1 = self.dropout1(_c1)
        sem1 = self.linear_pred1(_c1)
        if self.training:
            _c2 = self.MLP2(inputs2)
            _c2 = self.dropout2(_c2)
            sem2 = self.linear_pred2(_c2)
            return sem1, sem2
        else:
             return sem1

if __name__ == '__main__':
    input1 = [
            torch.rand(1, 32, 120, 160),
            torch.rand(1, 64, 60, 80),
            torch.rand(1, 128, 30, 40),
            torch.rand(1, 256, 15, 20)
    ]

    input2 = [
        torch.rand(1, 16, 120, 160),
        torch.rand(1, 32, 60, 80),
        torch.rand(1, 64, 30, 40),
        torch.rand(1, 128, 15, 20)
    ]

    decoder = Decoder_DualMLP(in_channels=[32, 64, 128, 256], embed_dim=256, num_classes=9).eval()
    out = decoder(input1, input2)
    # for i in out:
    #     print(i.shape)
    print(out.shape)


    # from thop import profile
    # flops, params = profile(decoder, inputs=(input, ))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")