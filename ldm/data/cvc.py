import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


class CVCBase(Dataset):
    """CVC Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["train", "val", "test","all"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])
        # TODO: more data transformation

        print(f"[Dataset]: CVC with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # segmentation = Image.open(example["file_path_"].replace("Original", "GroundTruth")).convert("RGB")
        # image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"].replace("Original", "Ground Truth")),cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]),cv2.COLOR_BGR2RGB))
        # print(f"Original image shape: {np.array(image).shape}")  # 调试原始图像形状
        # print(f"Original segmentation shape: {np.array(segmentation).shape}")  # 调试原始分割掩码形状

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        segmentation = (np.array(segmentation) > 128).astype(np.float32)

        # 打印分割图中的唯一值
        unique_values = np.unique(segmentation)
        # print(f"Unique values in segmentation before scaling: {unique_values}")

        # 检查每个通道的唯一值
        for channel in range(3):
            unique_channel_values = np.unique(segmentation[:, :, channel])
            # print(f"Unique values in channel {channel} before scaling: {unique_channel_values}")

        if self.mode == "test" and "all":
            example["segmentation"] = segmentation   
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1
        
        # 再次打印分割图中的唯一值
        unique_values_after_scaling = np.unique(example["segmentation"])
        # print(f"Unique values in segmentation after scaling: {unique_values_after_scaling}")

        # 检查每个通道的唯一值
        for channel in range(3):
            unique_channel_values = np.unique(example["segmentation"][:, :, channel])
            # print(f"Unique values in channel {channel} after scaling: {unique_channel_values}")

        # 打印一些具体的像素值
        # print(f"Pixel values at position (0, 0): {example['segmentation'][0, 0, :]}")
        # print(f"Pixel values at position (128, 128): {example['segmentation'][128, 128, :]}")
        # print(f"Pixel values at position (255, 255): {example['segmentation'][255, 255, :]}")

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        # print(f"Final image shape: {image.shape}")  # 调试最终图像形状
        # print(f"Final segmentation shape: {segmentation.shape}")  # 调试最终分割掩码形状
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self): # 80% / 10% / 10%
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        train_imgs, val_imgs, test_imgs = all_imgs[:492], all_imgs[492:492+60], all_imgs[492+60:]

        if self.mode == "train":
            return train_imgs
        elif self.mode == "val":
            return val_imgs
        elif self.mode == "test":
            return test_imgs
        elif self.mode == "all":
            return all_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test !")

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class CVCTrain(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="train", **kwargs)


class CVCValidation(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="val", **kwargs)


class CVCTest(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="test", **kwargs)

class CVCAll(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="all", **kwargs)


