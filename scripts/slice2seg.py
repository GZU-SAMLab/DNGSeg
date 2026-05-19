import argparse, os, sys, glob

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from ldm.data.synapse import SynapseValidation, SynapseValidationVolume
from ldm.data.refuge2 import REFUGE2Validation, REFUGE2Test
from ldm.data.refuge2 import REFUGE2Validation, REFUGE2All
from ldm.data.sts3d import STS3DValidation, STS3DTest
from ldm.data.cvc import CVCValidation, CVCTest
from ldm.data.cvc import CVCValidation, CVCAll
from ldm.data.kseg import KSEGValidation, KSEGTest
from ldm.data.kseg import KSEGValidation, KSEGAll
from ldm.data.voc import VOCValidation, VOCTest
from ldm.data.plant import PlantValidation, PlantTest
from ldm.data.isic import ISICValidation, ISICTest
from ldm.data.isic import ISICValidation, ISICAll
from medpy.metric.binary import hd95 as medpy_hd95


from scipy.ndimage import zoom

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor

def hd95_score(pred, targs, spacing=(1.0, 1.0)):
    """
    Compute 95% Hausdorff Distance in millimeters.
    - pred, targs: numpy arrays of shape (H, W) or (D, H, W)
    - spacing: voxel spacing in mm, e.g., (1.0, 1.0) for 2D isotropic, or (1.5, 0.8, 0.8) for 3D
    Returns:
        hd95 in mm, or 0 if both empty, or max possible distance if one is empty and the other not.
    """
    pred[pred > 0] = 1
    targs[targs > 0] = 1

    # Handle edge cases
    if pred.sum() == 0 and targs.sum() == 0:
        return 0.0
    elif pred.sum() == 0 or targs.sum() == 0:
        # One is empty → HD95 is undefined; common practice: return max possible distance
        # Approximate by diagonal of image in physical space
        shape = np.array(pred.shape)
        physical_diag = np.sqrt(np.sum((shape * np.array(spacing)) ** 2))
        return float(physical_diag)
    else:
        try:
            return float(medpy_hd95(pred, targs, voxelspacing=spacing))
        except Exception as e:
            # In rare cases (e.g., disconnected components), medpy may fail
            print(f"HD95 computation failed: {e}. Returning large value.")
            shape = np.array(pred.shape)
            physical_diag = np.sqrt(np.sum((shape * np.array(spacing)) ** 2))
            return float(physical_diag)


def prepare_for_first_stage(x, gpu=True):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    if gpu:
        x = x.to(memory_format=torch.contiguous_format).float().cuda()
    else:
        x = x.float()
    return x


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # if targs is None:
    #     return None
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() > 0 and targs.sum() == 0:
        return 1
    elif pred.sum() > 0 and targs.sum() > 0:
        # intersection = (pred * targs).sum()
        # union = pred.sum() + targs.sum() - intersection
        # return (2. * intersection) / (union + 10e-6)
        return (2. * (pred * targs).sum()) / (pred.sum() + targs.sum() + 1e-10)
    else:
        return 0


def iou_score(pred, targs):
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)

    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    # return intersection, union
    return intersection / (union + 1e-10)

def precision_score(pred, targs):
    """
    Compute Precision = TP / (TP + FP)
    Handles edge case: if no positive predictions, precision = 1 if target is also empty, else 0.
    """
    pred = (pred > 0).astype(np.float32)
    targs = (targs > 0).astype(np.float32)

    tp = (pred * targs).sum()
    fp = (pred * (1 - targs)).sum()

    if tp + fp == 0:
        # No positive predictions → if target is also empty, perfect; else, no recall but precision undefined → conventionally 1?
        # However, in segmentation, if model predicts nothing and there's no target, it's correct.
        # But if there IS target, then predicting nothing gives precision = 0? Actually, precision is not defined.
        # Common practice: return 1.0 when both are empty, else 0.0 when pred empty but target exists.
        if targs.sum() == 0:
            return 1.0
        else:
            return 0.0
    return tp / (tp + fp + 1e-10)


def recall_score(pred, targs):
    """
    Compute Recall = TP / (TP + FN)
    Handles edge case: if no ground truth positives, recall = 1 if prediction is also empty, else 0.
    """
    pred = (pred > 0).astype(np.float32)
    targs = (targs > 0).astype(np.float32)

    tp = (pred * targs).sum()
    fn = ((1 - pred) * targs).sum()

    if tp + fn == 0:
        # No ground truth positives → if prediction is also empty, it's correct
        if pred.sum() == 0:
            return 1.0
        else:
            return 0.0
    return tp / (tp + fn + 1e-10)



def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    # print(set(key.split(".")[0] for key in sd.keys()))
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    print("\033[31mmissing keys:\033[0m")
    print(m)
    # if len(u) > 0 and verbose:
    print("\033[31munexpected keys:\033[0m")
    print(u)
    # model.cuda()
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["pred_sum"] > 0 and kwargs["gt_sum"] > 0:
        return 2 * inter / (union + 1e-10)
    elif kwargs["pred_sum"] > 0 and kwargs["gt_sum"] == 0:
        return 1
    else:
        return 0


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["raw", "direct", "ddim", "plms", "dpm_solver"],
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument("--dataset", type=str,  # '-b' for binary, '-m' for multi
                        help="uses the model trained for given dataset", )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--times", type=int, default=1,
                        help="times of testing for stability evaluation", )
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the predictions for the whole test set.", )
    opt = parser.parse_args()

    
    if opt.dataset == "synapse-b":
        run = "2024-08-09T14-10-54_experiment_name"      # for example: 2024-02-13T17-09-00_binary
        print("Evaluate on synapse dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"      # name of the trained model
        opt.outdir = "outputs/slice2seg-samples-synapse-b"
        dataset = SynapseValidationVolume(num_classes=2)
    elif opt.dataset == "synapse-m":
        run = "2024-08-09T14-10-54_experiment_name" 
        print("Evaluate on synapse dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=107-step=14999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-synapse-m"
        dataset = SynapseValidationVolume(num_classes=9)
    elif opt.dataset == "synapse-14":
        run = "2026-01-19T16-26-35_experiment_syn_ori" 
        print("Evaluate on synapse dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"
        opt.outdir = "outputs/slice2seg-samples-synapse-14"
        dataset = SynapseValidationVolume(num_classes=14)
    elif opt.dataset == "refuge2-b":
        run = "2025-11-12T15-17-38_experiment_ref—disc_ori" 
        print("Evaluate on refuge2 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        # opt.ckpt = "models/ldm/synapse_binary/model.ckpt"
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"
        opt.outdir = "outputs/slice2seg-samples-refuge2-b"
        dataset = REFUGE2Test()
    elif opt.dataset == "refuge2-b-all":
        run = "2025-11-12T15-17-38_experiment_ref—disc_ori" 
        print("Evaluate on refuge2 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        # opt.ckpt = "models/ldm/synapse_binary/model.ckpt"
        opt.ckpt = f"logs/{run}/checkpoints/epoch=999-step=99999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-refuge2-b-ori"
        dataset = REFUGE2All()
    elif opt.dataset == "sts-3d": 
        run = "2024-08-09T14-10-54_experiment_name" 
        print("Evaluate on sts-3d dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        # opt.ckpt = "models/ldm/synapse_binary/model.ckpt"
        opt.ckpt = f"logs/{run}/checkpoints/epoch=199-step=124999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-sts-3d"
        dataset = STS3DTest()
    elif opt.dataset == "cvc":
        run = "2024-11-13T20-05-48_experiment_lstm_cvc" 
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"
        opt.outdir = "outputs/slice2seg-samples-cvc-lstmandloss"
        dataset = CVCTest()
    elif opt.dataset == "cvc_all":
        run = "2025-07-17T15-30-04_experiment_cvc_vae2_0m_47" 
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=813-step=99999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-cvc-vae-1m"
        dataset = CVCAll()
    elif opt.dataset == "cvc_all_1":
        run = "2025-07-17T15-30-04_experiment_cvc_vae2_0m_47" 
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=813-step=99999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-cvc-vae-1m"
        dataset = CVCAll()
    elif opt.dataset == "kseg":
        run = "2025-11-06T07-23-22_experiment_kseg_vae_100card" 
        print("Evaluate on kseg dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=639-step=127999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-kseg-vae-bianjie"
        dataset = KSEGTest()
    elif opt.dataset == "kseg_all":
        run = "2025-11-06T07-23-22_experiment_kseg_vae_100card" 
        print("Evaluate on kseg dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=639-step=127999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-kseg-vae-bianjie"
        dataset = KSEGAll()
    elif opt.dataset == "voc":
        run = "2024-08-09T14-10-54_experiment_cvc_ori" 
        print("Evaluate on synapse dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"
        opt.outdir = "outputs/slice2seg-samples-voc-ori"
        dataset = VOCTest()
    elif opt.dataset == "plant":
        run = "2025-02-15T18-18-08_experiment_plant_ori" 
        print("Evaluate on synapse dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=969-step=193999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-voc-ori"
        dataset = PlantTest()
    elif opt.dataset == "isic":
        run = "2025-11-12T22-49-06_experiment_isic_ori_l20" 
        print("Evaluate on isic dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=414-step=105999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-isic-ori"
        dataset = ISICTest()
    elif opt.dataset == "isic_all":
        run = "2025-11-17T01-27-44_experiment_isic_vae_l20_bianjie" 
        print("Evaluate on isic dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=687-step=175999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-isic-all-vae"
        dataset = ISICAll()
    else:
        raise NotImplementedError(f"Not implement for dataset {opt.dataset}")

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    config = OmegaConf.load(f"{opt.config}")
    config["model"]["params"].pop("ckpt_path", None)  # 如果不存在，返回 None，不报错
    config["model"]["params"]["cond_stage_config"]["params"].pop("ckpt_path")
    config["model"]["params"]["first_stage_config"]["params"].pop("ckpt_path")

    model, pl_sd = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)


    for idx in range(opt.times):
        if opt.times > 1:   # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")
        
        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        metrics_dict, _ = model.log_dice(data=data, save_dir=outpath if opt.save_results else None) 

        # # 在这里添加调试信息：
        # print("=== DEBUG INFO ===")
        # print("Available keys in metrics_dict:", list(metrics_dict.keys()))
        # print("DEBUG: Checking HD95 key...")

        # 检查 HD95 相关的键
        # hd95_keys = [key for key in metrics_dict.keys() if 'hd95' in key.lower()]
        # print("HD95 related keys:", hd95_keys)

        # 打印所有键值对以便调试
        # print("\nAll metrics_dict items:")
        # for key, value in metrics_dict.items():
        #     if 'hd95' in key.lower() or 'dice' in key.lower() or 'iou' in key.lower():
        #         print(f"  {key}: {value} (type: {type(value)}, len: {len(value) if hasattr(value, '__len__') else 'N/A'})")

        print("=== END DEBUG ===")

        dice_list = metrics_dict["val_avg_dice"]
        iou_list = metrics_dict["val_avg_iou"]
        precision_list = metrics_dict["val_avg_precision"]
        recall_list = metrics_dict["val_avg_recall"]
        # hd95_list = metrics_dict["val_avg_hd95"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][direct]: {sum(dice_list) / len(dice_list)}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][direct]: {sum(iou_list) / len(iou_list)}\033[0m")
        print(f"\033[31m[Mean precision][{opt.dataset}][direct]: {sum(precision_list) / len(precision_list)}\033[0m")
        print(f"\033[31m[Mean  recall][{opt.dataset}][direct]: {sum(recall_list) / len(recall_list)}\033[0m")
        # print(f"\033[31m[HD95][{opt.dataset}][direct]: {sum(hd95_list) / len(hd95_list)}\033[0m")

        if opt.times > 1:
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")
        
        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        metrics_dict, _ = model.log_dice(data=data, save_dir=outpath if opt.save_results else None) 

        dice_list = metrics_dict["val_avg_dice"]
        iou_list = metrics_dict["val_avg_iou"]
        precision_list = metrics_dict["val_avg_precision"]
        recall_list = metrics_dict["val_avg_recall"]
        # hd95_list = metrics_dict["val_avg_hd95"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][direct]: {sum(dice_list) / len(dice_list)}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][direct]: {sum(iou_list) / len(iou_list)}\033[0m")
        print(f"\033[31m[Mean precision][{opt.dataset}][direct]: {sum(precision_list) / len(precision_list)}\033[0m")
        print(f"\033[31m[Mean  recall][{opt.dataset}][direct]: {sum(recall_list) / len(recall_list)}\033[0m")
        # print(f"\033[31m[HD95][{opt.dataset}][direct]: {sum(hd95_list) / len(hd95_list)}\033[0m")

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()
   