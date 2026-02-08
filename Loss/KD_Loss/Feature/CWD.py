
import torch
import torch.nn as nn
import torch.nn.functional as F

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.contiguous().view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.contiguous().view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.contiguous().view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss


class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s, channel in
            zip(channels_s, channels_t)
        ]

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        # print('preds_S : ', preds_S.shape)
        masked_fea = torch.mul(preds_S, mat)
        # print('masked_fee : ', masked_fea.shape)
        # print(self.generation[idx])
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.align_module = nn.ModuleList([
        #     nn.Conv2d(channel, tea_channel, kernel_size=kd_loss, stride=kd_loss, padding=0).to(device)
        #     for channel, tea_channel in zip(channels_s, channels_t)
        # ])
        # self.norm = [
        #     nn.BatchNorm2d(tea_channel, affine=False).to(device)
        #     for tea_channel in channels_t
        # ]

        # print('channels_s_num : ', channels_s)
        # print('channels_T_num : ', channels_t)
        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        # tea_feats = []
        # stu_feats = []

        # for idx, (s, t) in enumerate(zip(y_s, y_t)):
        #     if self.distiller == 'cwd':
        #         s = self.align_module[idx](s)
        #         s = self.norm[idx](s)
        #         t = self.norm[idx](t)
        #     tea_feats.append(t)
        #     stu_feats.append(s)

        # loss = self.feature_loss(stu_feats, tea_feats)
        loss = self.feature_loss(y_s, y_t)
        return self.loss_weight * loss

def extract_module_channels(model, layers, role):
    layers = eval(layers)
    print('channels : ', layers)
    last_channels = []
    last_channel = None

    channel_model = model.model.model if role == 'teacher' else model.model
    for idx, module in enumerate(channel_model):
        if idx in layers:
            for name, layer in module.named_modules():
                if name == 'conv' or name == 'cv1.conv' or name == 'cv2.conv' or name == 'cv3.conv':
                    last_channel = layer.out_channels
            last_channels.append(last_channel)

    return last_channels

def extract_module_pairs(model, layers, role):
    layers = list(map(lambda x : str(x), eval(layers)))
    module_pairs = []
    for mname, ml in model.named_modules():
        if mname is not None:
            name = mname.split(".")
            # print(f'{role} : ', name)
            if name[0] == "module":
                name.pop(0)
            if len(name) == [4 if role == 'teacher' else 3][0] and name[2 if role == 'teacher' else 1] in layers:
                if "cv2" in mname:
                    module_pairs.append(ml)
    return module_pairs

class Distillation_loss:
    def __init__(self, modeln, modelL, distill_layers, distiller="mgd"):  # model must be de-paralleled

        self.distiller = distiller
        self.remove_handle = []
        self.distill_layers = distill_layers

        # Usage
        channels_t = extract_module_channels(modelL, self.distill_layers, role='teacher')
        channels_s = extract_module_channels(modeln, self.distill_layers, role='student')
        self.D_loss_fn = FeatureLoss(channels_s=channels_s, channels_t=channels_t, distiller=distiller)

        # Usage
        self.teacher_module_pairs = extract_module_pairs(modelL, self.distill_layers, role='teacher')
        self.student_module_pairs = extract_module_pairs(modeln, self.distill_layers, role='student')

        print('channels_t : ', channels_t)
        print('channels_s : ', channels_s)
        print('teacher module : ', len(self.teacher_module_pairs))
        print('student module : ', len(self.student_module_pairs))

        print('teacher module : ', self.teacher_module_pairs)
        print('student module : ', self.student_module_pairs)

    def register_hook(self):
        self.teacher_outputs = []
        self.origin_outputs = []

        def make_student_layer_forward_hook(output_list):
            def forward_hook(m, input, output):
                output_list.append(output)

            return forward_hook

        def make_teacher_layer_forward_hook(output_list):
            def forward_hook(m, input, output):
                output_list.append(output)

            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_layer_forward_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_layer_forward_hook(self.origin_outputs)))

    def get_loss(self):
        quant_loss = 0
        # for index, (mo, fo) in enumerate(zip(self.teacher_outputs, self.origin_outputs)):
        #     print(mo.shape,fo.shape)
        # quant_loss += self.D_loss_fn(mo, fo)

        # print('test1 : ',len(self.teacher_outputs))
        # print('test2 : ',len(self.origin_outputs))

        if not self.teacher_outputs or not self.origin_outputs:
            # print('없음')
            print(
                f"Warning: output not defined outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.origin_outputs)}")
            self.teacher_outputs.clear()
            self.origin_outputs.clear()
            return torch.tensor(0.0, requires_grad=True), False

        if len(self.teacher_outputs) != len(self.origin_outputs):
            print(
                f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.origin_outputs)}")
            self.teacher_outputs.clear()
            self.origin_outputs.clear()
            return torch.tensor(0.0, requires_grad=True), False

        # print(len(self.teacher_outputs), len(self.distill_layers))

        # if len(self.teacher_outputs) != len(self.distill_layers):
        #     diff_outputs = len(self.teacher_outputs) - len(self.distill_layers)

        if len(self.teacher_outputs) > len(self.distill_layers):
            self.teacher_outputs = self.teacher_outputs[len(self.distill_layers):]

        quant_loss += self.D_loss_fn(y_t=self.teacher_outputs, y_s=self.origin_outputs)

        self.teacher_outputs.clear()
        self.origin_outputs.clear()
        return quant_loss, True

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
