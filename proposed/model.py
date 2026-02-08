import torch
import torch.nn as nn
from proposed.backbone_model.TUNI import *
from proposed.decoder.MLPHead import Decoder_MLP
# from proposed.decoder.DualMLPHead import Decoder_DualMLP
import torch.nn.functional as F


class Encoder_RGBX(nn.Module):
    def __init__(self, mode, input):
        super(Encoder_RGBX, self).__init__()

        if mode == 'tiny':
            self.enc = tiny()
            if input=='RGBT':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/TUNI-TCSVT-upload/pretrain/tiny.pth.tar" # CDML
            )
                print('load from Tiny_RGBT')



        if mode == 'small':
            self.enc = small()
            if input=='RGBT':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/TCSVT/pre-train/small/model_best.pth.tar"
            )
                print('load from small_RGBT')

        if mode == 'base':
            self.enc = base()
            if input=='RGBT':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/TUNI-TCSVT-upload/pretrain/base.pth.tar"
            )
                print('load from Base_RGBT')




    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        outs = self.enc(rgb, t)
        return outs


class Model(nn.Module):
    def __init__(self, mode, input='RGBT', n_class=9):
        super(Model, self).__init__()

        # channels = [32, 64, 128, 256]
        emb_c = 256
        self.encoder = Encoder_RGBX(mode=mode, input=input)
        channels = self.encoder.enc.dims
        # self.decoder = Decoder_DualMLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class, dropout_ratio=0.1)
        self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class, dropout_ratio=0.1)
    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        f_rgb, f_t = self.encoder(rgb, t)
        sem = self.decoder(f_rgb)
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        return sem
        # if self.training:
        #     sem1, sem2 = self.decoder(f_rgb, f_t)
        #     sem1 = F.interpolate(sem1, scale_factor=4, mode='bilinear', align_corners=False)
        #     sem2 = F.interpolate(sem2, scale_factor=4, mode='bilinear', align_corners=False)
        #     return sem1, sem2
        # else:
        #     sem1 = self.decoder(f_rgb, f_t)
        #     sem1 = F.interpolate(sem1, scale_factor=4, mode='bilinear', align_corners=False)
        #     return sem1


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 480, 640)
    t = torch.rand(1, 3, 480, 640)
    # text = torch.rand(5, 512)
    model = Model(mode='base', input='RGBT').eval()
    out = model(rgb, t)
    print(out.shape)

    # from ptflops import get_model_complexity_info
    #
    # flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    # print('Flops ' + flops)
    # print('Params ' + params)

    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    print(flop_count_table(FlopCountAnalysis(model, rgb)))
