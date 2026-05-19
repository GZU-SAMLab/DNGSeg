import os
import glob
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import nibabel as nib

COLOR_MAP = np.array([
    [0., 0., 0.],
    [255., 0., 0.],
    [0., 255., 0.],
    [0., 0., 255.],
    [255., 255., 0.],
    [0., 255., 255.],
    [255., 0., 255.],
    [255., 239., 213.],
    [0., 0., 205.],
    [205., 133., 63.],
    [210., 180., 140.],
    [102., 205., 170.],
    [0., 0., 128.],
    [0., 139., 139.],
])

def colorize(seg, num_classes=14):
    """ seg (H W C)"""
    if num_classes == 2:
        return seg * 255
    for idx in range(1, 14):
        seg[seg[:, :, 0] == idx] = COLOR_MAP[idx]
    return seg

class SynapseBase(Dataset):
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.mode = mode
        self.num_classes = num_classes
        print(f"[Dataset]: Synapse with {self.num_classes} classes, in {self.mode} mode")
        assert mode in ["train", "val", "test_vol"]

        self.data_root = data_root
        if mode == "test_vol":
            self.data_paths = glob.glob(os.path.join(self.data_root, "img*"))
        else:
            self.data_paths = glob.glob(os.path.join(self.data_root, "*.png"))
        self._length = len(self.data_paths)

        self.labels = dict(
            file_path_=[path for path in self.data_paths],
        )
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        if self.mode == "test_vol":     # 3-D volume
            image = nib.load(example["file_path_"]).get_fdata()
            segmentation = nib.load(example["file_path_"].replace("img", "label")).get_fdata()

            image[image < -125] = -125
            image[image > 275] = 275
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 2) - 1

            if self.num_classes == 2:
                segmentation = self.transfer_to_9(segmentation)
                segmentation = np.where(segmentation > 0, 1, 0)
            elif self.num_classes == 9:
                segmentation = self.transfer_to_9(segmentation)
            elif self.num_classes == 14:
                pass  # Do nothing, keep original segmentation
            else:
                raise ValueError(f"Unsupported num_classes: {self.num_classes}")

            example["image"] = image
            example["segmentation"] = segmentation
            return example

        segmentation = np.array(Image.open(example["file_path_"]))
        image = np.load(example["file_path_"].replace("png", "npy"))

        segmentation = torch.tensor(segmentation.transpose([2, 0, 1]))
        image = torch.tensor(image.transpose([2, 0, 1]))

        if self.mode == "train":
            state = torch.get_rng_state()
            segmentation = self.transform(segmentation)
            torch.set_rng_state(state)
            image = self.transform(image)

        segmentation = np.array(segmentation.permute(1, 2, 0))
        image = np.array(image.permute(1, 2, 0))

        class_id = np.array([-1])  # Default value for binary segmentation

        if self.num_classes == 2:
            segmentation = self.transfer_to_9(segmentation)
            segmentation = np.where(segmentation > 0, 1, 0)
        elif self.num_classes == 9:
            segmentation = self.transfer_to_9(segmentation)
            exist_class = sorted(list(set(segmentation.flatten())))
            class_id = np.random.choice(np.array(exist_class), size=1, p=None).astype(np.int64)

            if class_id != 0:
                segmentation = (segmentation == class_id)
            else:
                segmentation = (segmentation != class_id)
        elif self.num_classes == 14:
            exist_class = sorted(list(set(segmentation.flatten())))
            class_id = np.random.choice(np.array(exist_class), size=1, p=None).astype(np.int64)

            if class_id != 0:
                segmentation = (segmentation == class_id)
            else:
                segmentation = (segmentation != class_id)

        example["class_id"] = class_id
        example["segmentation"] = ((segmentation.astype(np.float32) * 2) - 1)
        example["image"] = image
        assert (-1 <= example["image"].all() <= 1), (example["image"].min(), example["image"].max())
        assert (-1 <= example["segmentation"].all() <= 1), (example["segmentation"].min(), example["segmentation"].max())
        return example

    @staticmethod
    def transfer_to_9(gts):
        gts[gts == 5] = 0
        gts[gts == 6] = 5
        gts[gts == 7] = 6
        gts[gts == 8] = 7
        gts[gts == 9] = 0
        gts[gts == 10] = 0
        gts[gts == 11] = 8
        gts[gts == 12] = 0
        gts[gts == 13] = 0
        return gts

class SynapseTrain(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/synapse/train", mode="train", **kwargs)

class SynapseValidation(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/synapse/test", mode="val", **kwargs)

class SynapseValidationVolume(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/home/danyujiao/Stable-Diffusion-Seg-main/data/synapse/abdomen/imagesTr", mode="test_vol", **kwargs)

class SynapseValidationVolume4test(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# from Swin-UNETR:

# "validation": [
#         {
#             "image": "imagesTr/img0035.nii.gz",
#             "label": "labelsTr/label0035.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0036.nii.gz",
#             "label": "labelsTr/label0036.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0037.nii.gz",
#             "label": "labelsTr/label0037.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0038.nii.gz",
#             "label": "labelsTr/label0038.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0039.nii.gz",
#             "label": "labelsTr/label0039.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0040.nii.gz",
#             "label": "labelsTr/label0040.nii.gz"
#         }