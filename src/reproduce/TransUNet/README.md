# TransUNet Reproduction on Modal

This directory contains scripts to reproduce TransUNet training results using Modal.

## Prerequisites

1.  **Modal Account**: You need a Modal account.
2.  **Data**: The BTCV dataset should be downloaded and unzipped in `data/BTCV`.
    *   The script `download_data.py` downloads the data.
    *   The zip file `project_TransUNet.zip` should be unzipped. (This has been done).

## Setup

1.  **Secrets**:
    *   You need to set up your Modal credentials.
    *   Copy `.env.example` to `.env` (if you haven't already).
    *   Run the following command to authenticate (uses the local virtual environment):
        ```bash
        ../../../.venv/bin/modal setup
        ```
    *   Alternatively, manually populate `.env` with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from your [Modal Dashboard](https://modal.com/settings/tokens).

## Running the Reproduction

To start the training on Modal using the local virtual environment:

```bash
../../../.venv/bin/modal run reproduce_transunet.py
```

This will:
1.  Build a Docker image with Python 3.7 and dependencies.
2.  Download the pre-trained ViT model.
3.  Mount the code and data.
4.  Run the training script (`train.py`) on a GPU.
5.  Save checkpoints to a Modal Volume named `transunet-models`.

## Results

- The training logs will be streamed to your terminal.
Checkpoints are saved in the `transunet-models` volume.
You can manage volumes via the Modal dashboard or CLI.

- The detached App will keep running. You can track its progress on the Dashboard: 
https://modal.com/apps/hcmut-edu-vn/main/ap-vOWKSDoOH16OIhnK8mJM3j

Stream App logs:
```
modal app logs ap-vOWKSDoOH16OIhnK8mJM3j
```

Stop the App:
```
modal app stop ap-vOWKSDoOH16OIhnK8mJM3j
```
