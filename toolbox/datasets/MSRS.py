import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation
import glob

class MSRS(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre_feature_all-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")
        if self.mode in ['train', 'test']:
            rgb_folder = os.path.join(self.root, self.mode, 'vi')
            t_folder = os.path.join(self.root, self.mode, 'ir')
            label_folder = os.path.join(self.root, self.mode, 'Segmentation_labels')
            self.rgb_infos = glob.glob(os.path.join(rgb_folder, '*.*'))
            self.t_infos = glob.glob(os.path.join(t_folder, '*.*'))
            self.label_infos = glob.glob(os.path.join(label_folder, '*.*'))
        if self.mode in ['test_day', 'test_night']:
            # 过滤白天图像 (以D.png结尾)
            rgb_folder = os.path.join(self.root, 'test', 'vi')
            t_folder = os.path.join(self.root, 'test', 'ir')
            label_folder = os.path.join(self.root, 'test', 'Segmentation_labels')
            rgb_files = glob.glob(os.path.join(rgb_folder, '*.*'))
            t_files = glob.glob(os.path.join(t_folder, '*.*'))
            label_files = glob.glob(os.path.join(label_folder, '*.*'))
            if self.mode == 'test_day':
                self.rgb_infos = [f for f in rgb_files if f.endswith('D.png')]
                self.t_infos = [f for f in rgb_files if f.endswith('D.png')]
                self.label_infos = [f for f in label_files if f.endswith('D.png')]
            else:
                self.rgb_infos = [f for f in rgb_files if f.endswith('N.png')]
                self.t_infos = [f for f in t_files if f.endswith('N.png')]
                self.label_infos = [f for f in label_files if f.endswith('N.png')]


    def __len__(self):
        return len(self.rgb_infos)

    def __getitem__(self, index):
        rgb_path = self.rgb_infos[index].strip()
        t_path = self.t_infos[index].strip()
        label_path = self.label_infos[index].strip()

        image = Image.open(rgb_path)
        thermal = Image.open(t_path).convert('RGB')
        label = Image.open(label_path)

        sample = {
            'image': image,
            'thermal': thermal,
            'label': label,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.dp_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (64, 0, 128),  # car
            (64, 64, 0),  # person
            (0, 128, 192),  # bike
            (0, 0, 192),  # curve
            (128, 128, 0),  # car_stop
            (64, 64, 128),  # guardrail
            (192, 128, 128),  # color_cone
            (192, 64, 0),  # bump
        ]


if __name__ == '__main__':
    import json

    path = '/home/ubuntu/code/ICRA/configs/MSRS.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    # cfg['root'] = '/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/irseg'

    dataset = MSRS(cfg, mode='train', do_aug=True)
    print(len(dataset))
    from toolbox.utils import ClassWeight

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
                                               num_workers=cfg['num_workers'], pin_memory=True)
    classweight = ClassWeight('enet')  # enet, median_freq_balancing
    class_weight = classweight.get_weight(train_loader, 9)
    class_weight = torch.from_numpy(class_weight).float()

    print(class_weight)

    # # enet
    # [ kd_loss.5070, 17.0317, 30.0930, 34.9505, 40.5934, 40.5476, 48.0715, 46.1221, 45.6460]
