import os
import shutil
import json
import time
from torch.cuda import amp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from toolbox import get_dataset, get_metrics, get_logger, get_model, get_weights_semantic
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from Loss.dice import DiceLoss
from Loss.KLD_ml import CriterionKD
from toolbox.utils import parse_log_file, plot_training_curves

setup_seed(33)

class train_Loss(nn.Module):

    def __init__(self, cfg):
        super(train_Loss, self).__init__()
        self.class_weight_semantic = get_weights_semantic(cfg)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic, ignore_index=cfg['id_unlabel'])
        self.dice_loss = DiceLoss(mode='multiclass', ignore_index=cfg['id_unlabel'])
        self.ml = CriterionKD()
    def forward(self, out1, out2, targets):
        loss1 = self.semantic_loss(out1, targets) + self.dice_loss(out1, targets)
        loss_ml = self.ml(out1, out2.detach(), targets)
        loss = loss1 + loss_ml

        return loss, loss_ml

def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    ### multi-gpuss
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    ###
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # 如果有提供之前的log文件，解析它
    prev_train_losses, prev_test_losses, prev_test_mious = [], [], []
    if hasattr(args, 'previous_log') and args.previous_log:
        prev_log_path = args.previous_log
        logger.info(f'Loading previous training data from: {prev_log_path}')
        prev_train_losses, prev_test_losses, prev_test_mious = parse_log_file(prev_log_path)

        if prev_train_losses:
            logger.info(f'Previous training: {len(prev_train_losses)} epochs, '
                        f'final miou={prev_test_mious[-1]:.3f}')
        else:
            logger.warning('No previous training data found or failed to parse')

    # model
    model = get_model(cfg).to(gpu_ids[0])
    ## multi-gpus
    model = torch.nn.DataParallel(model, gpu_ids)

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)

    # dataloader
    dataset = get_dataset(cfg)
    train_loader = DataLoader(dataset['train'], batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    test_loader1 = DataLoader(dataset['test'], batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    Scaler = amp.GradScaler()
    train_criterion = train_Loss(cfg).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # # 指标 包含unlabel
    train_loss_meter1 = get_metrics(cfg)[0]
    train_loss_meter2 = get_metrics(cfg)[0]
    train_lossml_meter1 = get_metrics(cfg)[0]
    train_lossml_meter2 = get_metrics(cfg)[0]
    test_loss_meter = get_metrics(cfg)[0]
    running_metrics_test = get_metrics(cfg)[1]
    best_test = 0

    # 用于存储训练历史的列表
    train_losses1 = []
    train_losses2 = []
    train_lossesml1 = []
    train_lossesml2 = []
    test_losses = []
    test_mious = []

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter1.reset()
        train_loss_meter2.reset()
        train_lossml_meter1.reset()
        train_lossml_meter2.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            image = sample['image'].cuda()
            thermal = sample['thermal'].cuda()
            label = sample['label'].cuda()

            targets = label

            # learn from thermal branch
            with amp.autocast():
                if cfg['inputs'] == 't':
                    predict = model(thermal)
                else:
                    out1, out2 = model(image, thermal)
                loss1, lossml1 = train_criterion(out1, out2, targets)
                loss2, lossml2 = train_criterion(out2, out1, targets)
                loss = 0.8 * loss1 + 0.2 * loss2
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss_meter1.update(loss1.item())
            train_lossml_meter1.update(lossml1.item())
            train_loss_meter2.update(loss2.item())
            train_lossml_meter2.update(lossml2.item())

        scheduler.step()
        torch.cuda.empty_cache()

        # test
        with torch.no_grad():
            model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader1):

                image = sample['image'].cuda()
                thermal = sample['thermal'].cuda()
                label = sample['label'].cuda()
                if cfg['inputs'] == 't':
                    predict = model(thermal)
                else:
                    predict = model(image, thermal)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [b,c,h,w] to [c, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)


        train_loss1 = train_loss_meter1.avg
        train_loss2 = train_loss_meter2.avg
        train_lossml1 = train_lossml_meter1.avg
        train_lossml2 = train_lossml_meter2.avg
        test_loss = test_loss_meter.avg
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]

        train_losses1.append(train_loss1)
        train_losses2.append(train_loss2)
        train_lossesml1.append(train_lossml1)
        train_lossesml2.append(train_lossml2)
        test_losses.append(test_loss)
        test_mious.append(test_miou)

        # 每轮训练结束后打印结果
        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss1:.3f}/{train_loss2:.3f}/{test_loss:.3f}, '
                    f'lossml={train_lossml1:.3f}/{train_lossml2:.3f}, '
                    f'miou={test_miou:.3f}, '
                    )
        if test_miou > best_test:
            best_test = test_miou
            save_ckpt(logdir, model)

        # if ep >=0.8 * cfg["epochs"]:
        #     name = f"{ep+1}" + "_"
        #     save_ckpt(logdir, model, name)

        # if (ep + kd_loss) % 50 == 0:
        #     name = f"{ep + kd_loss}" + "_"
        #     save_ckpt(logdir, model, name)

        _ = plot_training_curves(
                logdir, train_losses1, test_losses, test_mious, ep + 1,
                prev_train_losses, prev_test_losses, prev_test_mious,
                "Previous"
            )
        # logger.info(f'Training curves saved to {plot_path}')

    # 训练结束后绘制最终图像
    final_plot_path = plot_training_curves(
        logdir, train_losses1, test_losses, test_mious, cfg['epochs'],
        prev_train_losses, prev_test_losses, prev_test_mious,
        "Previous"
    )
    logger.info(f'Final training curves saved to {final_plot_path}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/MSRS.json", help="Configuration file to use")
    parser.add_argument("--gpu_ids", type=str, default='0', help="set cuda device id")
    parser.add_argument("--previous_log", type=str,
                        default='',
                        help="path to previous training log file for comparison")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
