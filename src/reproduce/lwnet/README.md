# LWNet Reproduction Environment

This directory contains the setup files and instructions to reproduce the results from the [LWNet project](../../references/lwnet).

## Overview

LWNet (The Little W-Net That Could) is a retinal vessel segmentation model with minimalistic architecture (~70k parameters). This reproduction environment allows you to:
- Set up the required dependencies
- Download and prepare datasets
- Train models locally or on Google Colab
- Generate predictions and evaluate performance

## Quick Start

### Option 1: Local Setup with Conda (Recommended)

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate lwnet_reproduce
   ```

2. **Navigate to the reference directory:**
   ```bash
   cd ../../references/lwnet
   ```

3. **Download public datasets:**
   ```bash
   python get_public_data.py
   ```

4. **Train a model (requires GPU):**
   ```bash
   # Train on DRIVE dataset
   python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50 \
                            --model_name wnet --save_path wnet_drive --device cuda:0
   ```

### Option 2: Local Setup with pip

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Follow steps 2-4 from Option 1**

### Option 3: Google Colab (For GPU Training)

If you don't have a local GPU, use the provided Colab notebook:

1. Open `lwnet_colab_training.ipynb` in Google Colab
2. Follow the instructions in the notebook to:
   - Clone the repository
   - Install dependencies
   - Download datasets
   - Train models using Colab's free GPU

## System Requirements

### Minimum Requirements
- **Python:** 3.7
- **RAM:** 8GB minimum
- **Storage:** 5GB for datasets

### For GPU Training
- **GPU:** NVIDIA GPU with CUDA 10.0+ support
- **VRAM:** 4GB minimum (6GB+ recommended for HRF at 1024x1024)

### CPU-Only Training
While possible, CPU training is significantly slower. The authors note that training is feasible on CPU due to the small model size, but GPU is highly recommended.

## Reproduction Workflow

### 1. Data Preparation
```bash
cd ../../references/lwnet
python get_public_data.py
```

This downloads 7 public datasets: DRIVE, CHASE-DB, HRF, STARE, IOSTAR, ARIA, and RC-SLO.

### 2. Training Models

**DRIVE Dataset:**
```bash
python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50 \
                         --model_name wnet --save_path wnet_drive --device cuda:0
```

**CHASE-DB Dataset:**
```bash
python train_cyclical.py --csv_train data/CHASEDB/train.csv --cycle_lens 40/50 \
                         --model_name wnet --save_path wnet_chasedb --device cuda:0
```

**HRF Dataset (requires more VRAM):**
```bash
python train_cyclical.py --csv_train data/HRF/train.csv --cycle_lens 30/50 \
                         --model_name wnet --save_path wnet_hrf_1024 \
                         --im_size 1024 --batch_size 2 --grad_acc_steps 1 --device cuda:0
```

### 3. Generating Predictions

```bash
# DRIVE
python generate_results.py --config_file experiments/wnet_drive/config.cfg \
                           --dataset DRIVE --device cuda:0

# CHASE-DB
python generate_results.py --config_file experiments/wnet_chasedb/config.cfg \
                           --dataset CHASEDB --device cuda:0

# HRF
python generate_results.py --config_file experiments/wnet_hrf_1024/config.cfg \
                           --dataset HRF --im_size 1024 --device cuda:0
```

### 4. Evaluating Performance

```bash
# DRIVE
python analyze_results.py --path_train_preds results/DRIVE/experiments/wnet_drive \
                          --path_test_preds results/DRIVE/experiments/wnet_drive \
                          --train_dataset DRIVE --test_dataset DRIVE

# CHASE-DB
python analyze_results.py --path_train_preds results/CHASEDB/experiments/wnet_chasedb \
                          --path_test_preds results/CHASEDB/experiments/wnet_chasedb \
                          --train_dataset CHASEDB --test_dataset CHASEDB

# HRF
python analyze_results.py --path_train_preds results/HRF/experiments/wnet_hrf_1024 \
                          --path_test_preds results/HRF/experiments/wnet_hrf_1024 \
                          --train_dataset HRF --test_dataset HRF
```

## Cross-Dataset Evaluation

Train on one dataset and test on another:

```bash
# Train on DRIVE
python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50 \
                         --model_name wnet --save_path wnet_drive --device cuda:0

# Generate predictions on both DRIVE and CHASE-DB
python generate_results.py --config_file experiments/wnet_drive/config.cfg \
                           --dataset DRIVE --device cuda:0
python generate_results.py --config_file experiments/wnet_drive/config.cfg \
                           --dataset CHASEDB --device cuda:0

# Evaluate cross-dataset performance
python analyze_results.py --path_train_preds results/DRIVE/experiments/wnet_drive \
                          --path_test_preds results/CHASEDB/experiments/wnet_drive \
                          --train_dataset DRIVE --test_dataset CHASEDB
```

## Artery/Vein Segmentation

Train and evaluate A/V segmentation models:

```bash
# Train on DRIVE A/V
python train_cyclical.py --csv_train data/DRIVE/train_av.csv --model_name big_wnet \
                         --cycle_len 40/50 --do_not_save False --save_path big_wnet_drive_av \
                         --device cuda:0

# Generate A/V predictions
python generate_av_results.py --config_file experiments/big_wnet_drive_av/config.cfg \
                               --dataset DRIVE --device cuda:0
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 2` or `--batch_size 1`
- Use gradient accumulation: `--grad_acc_steps 2`
- Reduce image size (for HRF): `--im_size 512`

### Missing Datasets
- Ensure `get_public_data.py` completed successfully
- Check internet connection
- Some datasets may require manual download (e.g., LES-AV)

### Conda Environment Issues
- Clear conda cache: `conda clean --all`
- Use mamba for faster installation: `conda install mamba -c conda-forge`
- Create environment with mamba: `mamba env create -f environment.yml`

## Reference

**Paper:**
```
The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation with Minimalistic Models
Adrian Galdran, André Anjos, Jose Dolz, Hadi Chakor, Hervé Lombaert, Ismail Ben Ayed
https://arxiv.org/abs/2009.01907, Sep. 2020
```

## Directory Structure

```
src/
├── references/lwnet/          # Original LWNet repository (git submodule)
│   ├── data/                  # Downloaded datasets
│   ├── experiments/           # Trained model weights
│   ├── results/               # Generated predictions
│   └── ...
└── reproduce/lwnet/           # This directory
    ├── README.md              # This file
    ├── requirements.txt       # pip dependencies
    ├── environment.yml        # conda environment
    └── lwnet_colab_training.ipynb  # Google Colab notebook
```

## Next Steps

1. Choose your setup method (Conda, pip, or Colab)
2. Install dependencies
3. Download datasets with `get_public_data.py`
4. Start training models or use pretrained weights
5. Generate predictions and evaluate performance

For detailed information about the model architecture and experimental setup, refer to the [original README](../../references/lwnet/README.md) and the paper.
