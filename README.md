# DNGSeg: Dynamic Noise-Guided Diffusion with Adaptive Hypothesis Integration for Medical Image Segmentation

This repository contains the official implementation of **DNGSeg**, a medical image segmentation framework.

DNGSeg introduces **sample-adaptive dynamic noise modeling** in latent diffusion, generates multiple segmentation hypotheses, and integrates them with a **boundary-aware fusion strategy**. The main implementation is in [ldm/models/diffusion/ddpm_vae_bianjie.py](ldm/models/diffusion/ddpm_vae_bianjie.py).

## Overview

DNGSeg is designed around several key ideas:

- **Dynamic noise generation** from latent Gaussian statistics (`mu`, `std`) estimated by the VAE encoder.
- **Multi-hypothesis diffusion sampling** by generating multiple noise realizations with different modulation factors.
- **Adaptive local noise modulation** through optional local variance adaptation.
- **Boundary-aware hypothesis integration** to combine multiple segmentation predictions using boundary clarity as weights.

DNGSeg is designed to better capture uncertainty in the latent diffusion process and improve boundary quality in medical image segmentation.

## Key Modifications in DNGSeg

The main changes are implemented in [ddpm_vae_bianjie.py](ldm/models/diffusion/ddpm_vae_bianjie.py):

### 1. VAE-guided dynamic noise
DNGSeg extracts latent distribution statistics from the first-stage VAE:

- `encode_with_vae(...)` obtains latent `mu` and `std`
- `generate_dynamic_noise(...)` constructs diffusion noise as:

```python
noise = mu + alpha * std * base_noise
```

where `alpha` is modulated by `noise_id`, producing diverse but structured noise samples.

### 2. Multi-hypothesis generation
Instead of using a single fixed noise realization, DNGSeg samples multiple dynamic noises:

- each hypothesis corresponds to one `noise_id`
- each hypothesis leads to one noised latent and one segmentation prediction
- training and visualization scripts support repeated hypothesis generation and comparison

### 3. Adaptive hypothesis integration
Multiple segmentation hypotheses are integrated according to their boundary clarity:

- gradient-based boundary strength is computed from each prediction
- hypotheses with clearer boundaries receive larger fusion weights
- the fused prediction serves as the final segmentation output

### 4. Optional local adaptation
When enabled, local latent variance is used to further modulate dynamic noise spatially, allowing DNGSeg to adapt noise magnitude to local uncertainty patterns.

## Repository Structure

Important files and folders:

- [ldm/models/diffusion/ddpm_vae_bianjie.py](ldm/models/diffusion/ddpm_vae_bianjie.py) — core DNGSeg model implementation
- [configs/latent-diffusion/](configs/latent-diffusion/) — experiment configs
- [ldm/data/](ldm/data/) — dataset loaders
- [scripts/slice2seg.py](scripts/slice2seg.py) — inference / evaluation entry script

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate sdseg
```

Then install dependencies:

```bash
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

If GitHub access is limited, you can manually download and install `taming-transformers` and `CLIP` first, then run `pip install -e .` in this repository.

## Datasets

The project supports several medical image segmentation datasets. Dataset files are stored under [data/](data/) and dataloaders are implemented in [ldm/data/](ldm/data/).

Examples already used in this repository include:

- CVC-ClinicDB
- Kvasir-SEG
- ISIC
- REFUGE2

Please check the corresponding dataset loader under [ldm/data/](ldm/data/) for exact file organization.

## Pretrained Weights

DNGSeg relies on Stable Diffusion style pretrained initialization.

Download first-stage autoencoder and conditioning-stage weights:

```bash
bash scripts/download_first_stages_f8.sh
```

Download the diffusion UNet initialization:

```bash
bash scripts/download_models_lsun_churches.sh
```

## Training

Train DNGSeg with a config whose model target points to:

```text
ldm.models.diffusion.ddpm_vae_bianjie.LatentDiffusion
```

Example:

```bash
python -u main.py --base configs/latent-diffusion/cvc-ldm-kl-8.yaml -t --gpus 0 --name experiment_dngseg_cvc
```

You can monitor logs with:

```bash
tail -f nohup/experiment_dngseg_cvc.log
```

## Inference

After training, run segmentation inference with the evaluation script:

```bash
python -u scripts/slice2seg.py --dataset cvc
```

Depending on the dataset, you may need to adjust run paths or checkpoint paths inside the evaluation scripts.

## Acknowledgement

This project is implemented on top of the latent diffusion framework. We thank the related open-source contributors for making their code available.
