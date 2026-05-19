import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
from torch.utils.data import DataLoader


# 保留原有的类别映射
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

def voc_label_indices(colormap):
    """
    Convert colormap (PIL image) to binary label indices (uint8 tensor).
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    label = colormap2label[idx]
    
    # 将所有非背景类别的索引值设为 1，背景类别的索引值设为 0
    binary_label = np.where(label > 0, 1, 0).astype(np.uint8)
    
    # 扩展为 3 个通道
    binary_label = np.stack([binary_label, binary_label, binary_label], axis=-1)
    
    return binary_label

class VOCSegDataset(Dataset):
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["train", "val", "test"], "Mode must be one of train, val, or test"
        self.data_paths = self._parse_data_list()
        self.size = size
        self.interpolation = dict(nearest=Image.ANTIALIAS, bilinear=Image.BILINEAR)[interpolation]
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        print(f"[Dataset]: VOC with {num_classes} classes, in {self.mode} mode")

    def __getitem__(self, idx):
        image_path = self.data_paths['image'][idx]
        label_path = self.data_paths['label'][idx]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')  # 保持彩色以用于颜色映射

        if self.size is not None:
            label = label.resize((self.size, self.size), resample=self.interpolation)
            image = image.resize((self.size, self.size), resample=Image.BILINEAR)

        if self.mode == "train":
            label, image = self._utilize_transformation(label, image, self.transform)

        # 将标签转换为二值图
        label = voc_label_indices(label)
        # 打印标签数据
        # print(f"Label shape: {label.shape}")
        # print(f"Label unique values: {torch.unique(label)}")
        # if self.mode == "test":
        #     example["label"] = label   
        # else:
        #     example["label"] = ((label * 2) - 1)   # range: binary -1 and 1

        # 归一化图像
        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.

        example = {
            "image": image,
            "segmentation": label,
            "class_id": np.array([-1]),  # 不影响二分类
            "file_path_": image_path  # 添加原始图像的文件路径
        }

        return example

    def __len__(self):
        return len(self.data_paths['image'])

    def _parse_data_list(self):
        txt_fname = os.path.join(self.data_root, 'ImageSets/Segmentation', f'{self.mode}.txt')
        with open(txt_fname, 'r') as f:
            filenames = f.read().splitlines()

        images = [os.path.join(self.data_root, 'JPEGImages', f + '.jpg') for f in filenames]
        labels = [os.path.join(self.data_root, 'SegmentationClass', f + '.png') for f in filenames]

        return {'image': images, 'label': labels}

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image

class VOCTrain(VOCSegDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/VOCdevkit/VOC2012", mode="train", **kwargs)


class VOCValidation(VOCSegDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/VOCdevkit/VOC2012", mode="val", **kwargs)


class VOCTest(VOCSegDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/VOCdevkit/VOC2012", mode="test", **kwargs)
