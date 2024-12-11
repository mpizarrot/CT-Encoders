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

### iBOT Training

For a glimpse at the full documentation of iBOT pre-training, please run:
```bash
python ibot/main_ibot.py --help
   ```
To start the iBOT pre-training with Vision Transformer (ViT), simply run the following command:
```bash
python ibot/main_ibot.py --data_path /../your_dataset --output_dir ./output_dir
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

   





