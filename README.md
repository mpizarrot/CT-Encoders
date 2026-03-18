# CT Encoders

This repository contains the code used and modified from the [MAE](https://github.com/facebookresearch/mae) and [iBOT](https://github.com/bytedance/ibot) repositories to pre-train encoders with ViT architecture using computed tomography images.

## Installation

1. Create a virtual environment: 
   ```bash
   conda create -n ctencoders python=3.10 -y
   ```
   and activate it: 
   ```bash
   conda activate ctencoders
   ```
2. Install [PyTorch 2.5](https://pytorch.org/get-started/locally/)
3. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### Using a CT Encoder

The following example shows how to instantiate a `SwinTransformerCatIn` encoder and run it on the CT volume [`CRLM-CT-1001.nii.gz`](/home/mpizarro/CT-Encoders/CRLM-CT-1001.nii.gz) using `SimpleITK`. You can download the checkpoint from [this link](http://201.238.213.114:2280/sketchapp/get_files/swin_g2_enc_pretrained.pth).

Example:

```python
import SimpleITK as sitk
import numpy as np
import torch

from SwinTransformer import SwinTransformerCatIn

encoder = SwinTransformerCatIn(
    in_chans=1,
    embed_dim=48,
    window_size=(7, 7, 7),
    patch_size=(4, 4, 4),
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    return_all_tokens=False,
    masked_im_modeling=True,
    swin_weights_path="swin_g2_enc_pretrained.pth",
)

ct = sitk.ReadImage("CRLM-CT-1001.nii.gz")
ct_array = sitk.GetArrayFromImage(ct).astype(np.float32)

# Add batch and channel dimensions to get [B, C, D, H, W]
x = torch.from_numpy(ct_array).unsqueeze(0).unsqueeze(0)

out = encoder(x)
for i in out:
    print(i.shape)
```

If your CT volume does not already match the input size expected by the encoder, you may need to resample or crop it before inference.

### iBOT Training

For a glimpse at the full documentation of iBOT pre-training, please run:
```bash
python ibot/main_ibot.py --help
   ```
To start the iBOT pre-training with Vision Transformer (ViT), simply run the following command:
```bash
python ibot/main_ibot.py --data_path /../your_dataset --output_dir ./output_dir
   ```
To start the iBOT pre-training with Swin3D, simply run the following command:
```bash
python ibot/main_ibot.py \
--arch swin3D \
--batch_size_per_gpu 2 \
--epochs 400 \
--data_path /../your_dataset \
--saveckp_freq 40 \
--clip_grad 3.0 \
--global_crops_scale 0.4 1.0 \
--local_crops_scale 0.05 0.4 \
--lr 0.0005 \
--min_lr 1e-06 \
--norm_last_layer False \
--patch_size 4 \
--pred_ratio 0.0 0.3 \
--pred_ratio_var 0.0 0.2 \
--pred_shape block \
--pred_start_epoch 50 \
--warmup_teacher_temp_epochs 30 \
--window_size 7 \
--output_dir ./output_dir
   ```

### MAE Training

For a glimpse at the full documentation of MAE pre-training, please run:
```bash
python mae/main_pretrain.py --help
   ```
To start the MAE pre-training, simply run the following command:
```bash
python mae/main_pretrain.py --data_path /../your_dataset --output_dir ./output_dir
   ```




