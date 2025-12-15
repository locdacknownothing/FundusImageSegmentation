
import modal
import os
import subprocess
import time
import random

# Define the Modal application
app = modal.App("nnunet-hippocampus")

# Define the remote volumes
# 'nnunet-data': Stores raw, downloaded, and preprocessed data
data_vol = modal.Volume.from_name("nnunet-data", create_if_missing=True)
# 'nnunet-models': Stores training checkpoints and inference results
models_vol = modal.Volume.from_name("nnunet-models", create_if_missing=True)

# Define paths (mapped to the volumes)
NNUNET_RAW = "/root/nnUNet_data/nnUNet_raw"
NNUNET_PREPROCESSED = "/root/nnUNet_data/nnUNet_preprocessed"
NNUNET_RESULTS = "/root/nnUNet_models/nnUNet_results"

# Image definition with constrained dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "tar")
    .pip_install(
        "torch<=2.8.0",
        "torchaudio<=2.8.0",
        "torchvision",
        "nnunetv2"
    )
    .env({
        "nnUNet_raw": NNUNET_RAW,
        "nnUNet_preprocessed": NNUNET_PREPROCESSED,
        "nnUNet_results": NNUNET_RESULTS,
    })
)

# Common configuration for all functions
common_mounts = {
    "/root/nnUNet_data": data_vol,
    "/root/nnUNet_models": models_vol,
}

@app.function(image=image, volumes=common_mounts, timeout=3600)
def prepare_data():
    """
    Downloads, Converts, and Preprocesses data.
    This runs once sequentially.
    """
    print(">>> [Step 1] Preparing Data...")
    
    # Ensure directories exist
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        os.makedirs(d, exist_ok=True)
    
    downloads_dir = "/root/nnUNet_data/downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    
    dataset_name = "Dataset004_Hippocampus"
    converted_dataset_path = os.path.join(NNUNET_RAW, dataset_name)
    
    # 1. Download & Convert
    if not os.path.exists(converted_dataset_path):
        print(f"Downloading {dataset_name}...")
        task_tar = os.path.join(downloads_dir, "Task04_Hippocampus.tar")
        if not os.path.exists(task_tar):
            subprocess.run(
                ["wget", "-q", "-O", task_tar, "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"],
                check=True
            )
        subprocess.run(["tar", "-xf", task_tar, "-C", downloads_dir], check=True)
        
        print("Converting to nnU-Net format...")
        subprocess.run([
            "nnUNetv2_convert_MSD_dataset",
            "-i", os.path.join(downloads_dir, "Task04_Hippocampus"),
            "-overwrite_id", "004"
        ], check=True)
    else:
        print("Data already converted.")

    # 2. Plan & Preprocess
    plans_file = os.path.join(NNUNET_PREPROCESSED, dataset_name, "nnUNetPlans.json")
    if not os.path.exists(plans_file):
        print("Running Planning and Preprocessing...")
        subprocess.run([
            "nnUNetv2_plan_and_preprocess",
            "-d", "004",
            "--verify_dataset_integrity"
        ], check=True)
    else:
        print("Preprocessing already complete.")
        
    # Commit changes to volumes so other workers can see them
    data_vol.commit()

@app.function(image=image, volumes=common_mounts, gpu="T4", timeout=86400)
def train_config_fold(config: str, fold: int, max_epochs: int):
    """
    Trains a specific configuration and fold.
    """
    # Reload volumes to ensure we see the preprocessed data
    data_vol.reload()
    models_vol.reload()
    
    print(f">>> [Step 2] Training Config: {config}, Fold: {fold}...")
    
    # --npz is REQUIRED for find_best_configuration (saves validation softmax)
    subprocess.run([
        "nnUNetv2_train",
        "004",
        config,
        str(fold),
        "-tr", f"nnUNetTrainer_{max_epochs}epochs", # Use max_epochs instead of 1000
        # "--npz" 
    ], check=True)
    
    # Commit the trained weights and .npz files
    models_vol.commit()
    print(f"Finished Training Config: {config}, Fold: {fold}")

@app.function(image=image, volumes=common_mounts, timeout=3600)
def find_best_configuration_and_postprocessing(config: str, fold: int, max_epochs: int):
    """
    Runs nnUNetv2_find_best_configuration to:
    1. Determine the best postprocessing (based on the validation folds available).
    2. Report validation metrics (Dice scores).
    3. Generate inference instructions.
    """
    models_vol.reload()
    print(">>> [Step 3] Finding Best Configuration & Postprocessing...")
    
    # We restrict the search to the specific config and fold we trained.
    # This effectively just evaluates that specific run and optimizes postprocessing for it.
    subprocess.run([
        "nnUNetv2_find_best_configuration",
        "004",
        "-c", config,
        "-f", str(fold), 
        "-tr", f"nnUNetTrainer_{max_epochs}epochs", 
        "--disable_ensembling" # No ensemble since we only have 1 config/fold
    ], check=True)
    
    models_vol.commit()

@app.function(image=image, volumes=common_mounts, gpu="T4", timeout=3600)
def run_inference(config: str, fold: int, max_epochs: int):
    """
    Runs inference using the determined best configuration.
    """
    models_vol.reload()
    data_vol.reload()
    print(">>> [Step 4] Running Inference...")

    dataset_name = "Dataset004_Hippocampus"

    input_folder = os.path.join(NNUNET_RAW, dataset_name, "imagesTs")
    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print("imagesTs missing/empty, using imagesTr for demo.")
        input_folder = os.path.join(NNUNET_RAW, dataset_name, "imagesTr")
        
    output_folder = os.path.join(NNUNET_RESULTS, dataset_name, "final_predictions")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Predicting from {input_folder} to {output_folder} using 3d_fullres (fold 0 only)...")
    
    # Run prediction for the specific fold and config
    subprocess.run([
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", "004",
        "-c", config,
        "-f", str(fold),
        "-tr", f"nnUNetTrainer_{max_epochs}epochs", 
        # "-chk", "checkpoint_best.pth",  ## Optional: only use when training is not done
    ], check=True)
    
    models_vol.commit()
    print(f"Inference complete. Results in {output_folder}")

@app.function(image=image, volumes=common_mounts, timeout=86400) # Long timeout for the orchestrator
def run_full_pipeline(config: str, fold: int, max_epochs: int):
    """
    Orchestrates the entire pipeline remotely. 
    This ensures the sequence continues even if the local client disconnects (detached mode).
    """
    print(">>> Triggering Data Preparation...")
    prepare_data.remote()
    
    print(f">>> Starting Training for Config: {config}, Fold: {fold}...")
    train_config_fold.remote(config, fold, max_epochs)

    print(f">>> Evaluating Validation Set (Metrics & Postprocessing)...")
    find_best_configuration_and_postprocessing.remote(config, fold, max_epochs)

    print(f">>> Starting Inference...")
    run_inference.remote(config, fold, max_epochs)

@app.local_entrypoint()
def main():
    config = "3d_fullres" # ONE best config
    fold = 2  # random.randint(0, 4) # ONE random fold
    max_epochs = 50 # Reduced epochs
    
    run_full_pipeline.remote(config, fold, max_epochs)
