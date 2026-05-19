import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class REFUGE2Base(Dataset):
    """REFUGE2 Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["train", "val", "test"]
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

        print(f"[Dataset]: REFUGE-2 with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = Image.open(example["file_path_"].replace("images", "mask")).convert("RGB")
        image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        # 修改后的二值化处理：0和128视为前景(1)，255视为背景(0)
        segmentation = np.array(segmentation).astype(np.float32)
        
        # 提取绿色通道（通常分割信息在G通道）
        seg_channel = segmentation[:, :, 1]  # 使用绿色通道
        
        # 创建二值掩码：0和128的像素变为1（前景），255变为0（背景）
        binary_mask = np.zeros_like(seg_channel)
        binary_mask[(seg_channel == 0) | (seg_channel == 128)] = 1.0  # 前景
        # binary_mask[(seg_channel == 0)] = 1.0  # 前景
        binary_mask[seg_channel == 255] = 0.0  # 背景
        # binary_mask[seg_channel == 255 | (seg_channel == 128)] = 0.0  # 背景
        
        # 将单通道掩码转换为三通道
        segmentation = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
        
        if self.mode == "test":
            example["segmentation"] = segmentation   
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        return all_imgs

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class REFUGE2Train(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/refuge2/REFUGE2/train/images", mode="train", **kwargs)


class REFUGE2Validation(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/refuge2/REFUGE2/val/images", mode="val", **kwargs)


class REFUGE2Test(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/refuge2/REFUGE2/test/images", mode="test", **kwargs)