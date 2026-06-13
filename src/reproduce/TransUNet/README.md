# TransUNet Reproduction on Modal

This directory contains scripts to reproduce TransUNet training results using Modal.

## Prerequisites

1. **Modal Version**: must be compatible with `Python 3.9`, version `1.2.6` recommended. 
2. **Modal Account**: You need a Modal account.
3. **Data**: The BTCV dataset should be downloaded and unzipped in `data/BTCV`.
    *   The script `download_data.py` downloads the data.
    *   The zip file `project_TransUNet.zip` should be unzipped. (This has been done).

## Setup

1.  **Secrets**:
    *   You need to set up your Modal credentials.
    *   Copy `.env.example` to `.env` (if you haven't already).
    *   Run the following command to authenticate (uses the local virtual environment):
        ```bash
        modal setup
        ```
    *   Alternatively, manually populate `.env` with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from your [Modal Dashboard](https://modal.com/settings/tokens).

## Running the Reproduction

To start the training/test on Modal using the local virtual environment:

```bash
modal run --detach reproduce_*.py::train_transunet
```
or
```bash
modal run reproduce_*.py::test_transunet
```

This will:
1.  Build a Docker image with Python 3.9 and dependencies.
2.  Download the pre-trained ViT model and data.
3.  Mount the code.
4.  Run the training script (`train.py`), or test script (`test.py`) on a GPU.
5.  Save training or test results (checkpoints, logs, etc.) to a Modal Volume named `transunet-models`.

## Results

- The training logs will be streamed to your terminal.
Checkpoints are saved in the `transunet-models` volume.
You can manage volumes via the Modal dashboard or CLI.

- The detached App will keep running, i.e. run with `--detach` flag. 
You can track its progress on the Dashboard. For example:
`https://modal.com/apps/{project}/main/{long_id}`

Stream App logs:
```
modal app logs {long_id}
```

Stop the App:
```
modal app stop {long_id}
```
