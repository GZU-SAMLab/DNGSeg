#!/bin/bash

# 创建必要的目录
mkdir -p models/first_stage_models/kl-f4
mkdir -p models/first_stage_models/kl-f16
mkdir -p models/first_stage_models/kl-f32
mkdir -p models/first_stage_models/vq-f4
mkdir -p models/first_stage_models/vq-f4-noattn
mkdir -p models/first_stage_models/vq-f8
mkdir -p models/first_stage_models/vq-f8-n256
mkdir -p models/first_stage_models/vq-f16

# 下载模型文件
wget -O models/first_stage_models/kl-f4/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f4.zip
wget -O models/first_stage_models/kl-f16/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f16.zip
wget -O models/first_stage_models/kl-f32/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f32.zip
wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
wget -O models/first_stage_models/vq-f4-noattn/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4-noattn.zip
wget -O models/first_stage_models/vq-f8/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
wget -O models/first_stage_models/vq-f8-n256/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip
wget -O models/first_stage_models/vq-f16/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f16.zip

# 解压文件
for dir in models/first_stage_models/*; do
    cd "$dir"
    unzip model.zip
done