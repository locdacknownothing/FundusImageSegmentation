
import modal
import os
import subprocess
import time
import random

# Define the Modal application
app = modal.App("nnunet-lung")

# Define the remote volumes
data_vol = modal.Volume.from_name("nnunet-data", create_if_missing=True)
models_vol = modal.Volume.from_name("nnunet-models", create_if_missing=True)

# Define paths
NNUNET_RAW = "/root/nnUNet_data/nnUNet_raw"
NNUNET_PREPROCESSED = "/root/nnUNet_data/nnUNet_preprocessed"
NNUNET_RESULTS = "/root/nnUNet_models/nnUNet_results"

image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "tar")
    .pip_install("torch<=2.8.0", "torchaudio<=2.8.0", "torchvision", "nnunetv2")
    .env({
        "nnUNet_raw": NNUNET_RAW,
        "nnUNet_preprocessed": NNUNET_PREPROCESSED,
        "nnUNet_results": NNUNET_RESULTS,
    })
)

common_mounts = {
    "/root/nnUNet_data": data_vol,
    "/root/nnUNet_models": models_vol,
}

@app.function(image=image, volumes=common_mounts, timeout=3600)
def prepare_data():
    """
    Downloads, Converts, and Preprocesses data.
    """
    print(">>> [Step 1] Preparing Data...")
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        os.makedirs(d, exist_ok=True)
    
    downloads_dir = "/root/nnUNet_data/downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    
    dataset_name = "Dataset006_Lung"
    converted_dataset_path = os.path.join(NNUNET_RAW, dataset_name)
    
    if not os.path.exists(converted_dataset_path):
        print(f"Downloading {dataset_name}...")
        task_tar = os.path.join(downloads_dir, "Task06_Lung.tar")
        if not os.path.exists(task_tar):
            subprocess.run(["wget", "-q", "-O", task_tar, "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar"], check=True)
        subprocess.run(["tar", "-xf", task_tar, "-C", downloads_dir], check=True)
        
        print("Converting to nnU-Net format...")
        subprocess.run(["nnUNetv2_convert_MSD_dataset", "-i", os.path.join(downloads_dir, "Task06_Lung"), "-overwrite_id", "006"], check=True)
    else:
        print("Data already converted.")

    plans_file = os.path.join(NNUNET_PREPROCESSED, dataset_name, "nnUNetPlans.json")
    if not os.path.exists(plans_file):
        print("Running Planning and Preprocessing...")
        subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "006", "--verify_dataset_integrity"], check=True)
    else:
        print("Preprocessing already complete.")
    data_vol.commit()

@app.function(image=image, volumes=common_mounts, gpu="T4", timeout=86400)
def train_config_fold(config: str, fold: int, max_epochs: int):
    """
    Trains a specific configuration and fold.
    For cascade, we typically train 3d_lowres then 3d_cascade_fullres.
    """
    data_vol.reload()
    models_vol.reload()
    
    print(f">>> [Step 2] Training Config: {config}, Fold: {fold}...")
    
    cmd = [
        "nnUNetv2_train",
        "006",
        config,
        str(fold),
        "-tr", f"nnUNetTrainer_{max_epochs}epochs",
        "--npz"
    ]
    subprocess.run(cmd, check=True)
    
    models_vol.commit()
    print(f"Finished Training Config: {config}, Fold: {fold}")

@app.function(image=image, volumes=common_mounts, timeout=3600)
def find_best_configuration_and_postprocessing(fold: int, max_epochs: int):
    models_vol.reload()
    print(">>> [Step 3] Finding Best Configuration & Postprocessing...")
    
    subprocess.run([
        "nnUNetv2_find_best_configuration",
        "006",
        "-c", "3d_lowres", 
        "-f", str(fold), 
        "-tr", f"nnUNetTrainer_{max_epochs}epochs", 
    ], check=True)
    models_vol.commit()

@app.function(image=image, volumes=common_mounts, gpu="T4", timeout=3600)
def run_cascade_inference(fold: int, max_epochs: int):
    """
    Runs inference for 3D Cascade.
    1. Predict LowRes.
    2. Predict Cascade FullRes (using LowRes output).
    """
    models_vol.reload()
    data_vol.reload()
    print(">>> [Step 4] Running Cascade Inference...")

    dataset_name = "Dataset006_Lung"
    input_folder = os.path.join(NNUNET_RAW, dataset_name, "imagesTs")
    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print("imagesTs missing/empty, using imagesTr for demo.")
        input_folder = os.path.join(NNUNET_RAW, dataset_name, "imagesTr")
    
    out_lowres = os.path.join(NNUNET_RESULTS, dataset_name, f"pred_lowres_fold{fold}")
    out_cascade = os.path.join(NNUNET_RESULTS, dataset_name, f"final_predictions_cascade_fold{fold}")
    
    for d in [out_lowres, out_cascade]:
        os.makedirs(d, exist_ok=True)

    # 1. Predict LowRes
    print("Predicting LowRes...")
    subprocess.run([
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", out_lowres,
        "-d", "006",
        "-c", "3d_lowres",
        "-f", str(fold),
        "-tr", f"nnUNetTrainer_{max_epochs}epochs",
        # "-chk", "checkpoint_best.pth",
    ], check=True)
    
    # 2. Predict Cascade
    print("Predicting Cascade FullRes...")
    subprocess.run([
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", out_cascade,
        "-d", "006",
        "-c", "3d_cascade_fullres",
        "-f", str(fold),
        "-chk", "checkpoint_best.pth",
        "-prev_stage_predictions", out_lowres
    ], check=True)
    
    models_vol.commit()
    print(f"Cascade Inference complete. Results in {out_cascade}")

@app.function(image=image, volumes=common_mounts, gpu="T4", timeout=86400)
def run_full_pipeline(fold: int, max_epochs: int):
    print(">>> Triggering Data Preparation...")
    prepare_data.remote()
    
    # 2. Train LowRes
    print(f"Training 3d_lowres Fold {fold}...")
    train_config_fold.remote("3d_lowres", fold, max_epochs)
    
    # 3. Train Cascade FullRes
    print(f"Training 3d_cascade_fullres Fold {fold}...")
    try:
        train_config_fold.remote("3d_cascade_fullres", fold, max_epochs)
    except Exception as e:
        print(f"Warning: Cascade training failed. Error: {e}")
        return

    # 4. Evaluate
    print(f">>> Evaluating Validation Set...")
    find_best_configuration_and_postprocessing.remote(fold, max_epochs)

    # 5. Inference
    # run_cascade_inference.remote(fold)

@app.local_entrypoint()
def main():
    fold = random.randint(0, 4) 
    max_epochs = 50 
    
    run_full_pipeline.remote(fold, max_epochs)
