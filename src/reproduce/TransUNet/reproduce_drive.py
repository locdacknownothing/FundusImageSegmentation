import modal
import os

app = modal.App("transunet-drive")

# Volume for checkpoints to persist results
model_volume = modal.Volume.from_name("transunet-models", create_if_missing=True)

# Define the image with dependencies and local files
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("wget", "git", "unzip", "libgl1", "libglib2.0-0", "curl")
    .pip_install(
        "torch", 
        "torchvision", 
        "numpy", 
        "tqdm", 
        "tensorboard", 
        "tensorboardX", 
        "ml-collections", 
        "medpy", 
        "SimpleITK", 
        "scipy", 
        "h5py",
        "gdown",
        "opencv-python",
        "scikit-image",
        "pandas",
        "imagecodecs"
    )
    # Download weights
    .run_commands(
        "wget -q https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz",
        "mkdir -p /assets/vit_checkpoint/imagenet21k",
        "mv R50+ViT-B_16.npz /assets/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    )
    # Mount TransUNet source code
    .add_local_dir(
        "/home/locdac/Documents/DATN_ThS/FundusImageSegmentation/src/references/TransUNet",
        remote_path="/app/transunet"
    )
)

# Volume for data
data_volume = modal.Volume.from_name("transunet-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="t4",
    volumes={
        "/app/model": model_volume,
        "/app/data": data_volume
    }, 
    timeout=86400 # 24 hours
)
def train_transunet():
    import os
    import shutil
    import pandas as pd
    import numpy as np
    from skimage import io
    
    # Change working directory to the code
    os.chdir("/app/transunet")
    
    # 1. Prepare Pre-trained Weights
    pretrained_target_dir = "/app/model/vit_checkpoint/imagenet21k"
    pretrained_target_file = os.path.join(pretrained_target_dir, "R50+ViT-B_16.npz")
    
    if not os.path.exists(pretrained_target_file):
        print(f"Pre-trained weights not found at {pretrained_target_file}. Copying from image assets...")
        os.makedirs(pretrained_target_dir, exist_ok=True)
        shutil.copy("/assets/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz", pretrained_target_file)
        print("Pre-trained weights copied.")
        model_volume.commit()
    else:
        print("Pre-trained weights found in volume.")

    # 2. Check and Prepare DRIVE Data
    drive_data_path = "/app/data/DRIVE"

    # # DEBUG: Remove existing data
    # if os.path.exists(drive_data_path):
    #     shutil.rmtree(drive_data_path)
    #     data_volume.commit()

    if not os.path.exists(drive_data_path) or not os.listdir(drive_data_path):
        print("DRIVE data not found. Downloading and preparing...")
        os.makedirs(drive_data_path, exist_ok=True)
        
        # Download commands adapted from get_public_data.py
        # URL for DRIVE itself is usually requiring login or similar, but the script uses AV_groundTruth.zip for AV labels.
        # Wait, get_public_data.py downloads AV_groundTruth.zip AND has existing data?
        # Line 17: downloads deepdyn-master/data which has DRIVE images?
        # "curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data"
        # This seems to download the base DRIVE images and structure from deepdyn repo.
        
        print("Downloading base DRIVE data from deepdyn repo...")
        # We need to run this in /app/data to extract into /app/data/DRIVE (if structure matches)
        # deepdyn-master/data contains DRIVE folder.
        
        os.chdir("/app/data")
        cmd_download_base = "curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data"
        os.system(cmd_download_base)
        
        # Clean up other datasets downloaded by deepdyn command if we only want DRIVE
        for d in ["VEVIO", "STARE", "CHASEDB", "AV-WIDE", "HRF", "DR-HAGIS", "LES-AV"]:
             if os.path.exists(d):
                 shutil.rmtree(d)
        
        # Now we have /app/data/DRIVE with images, mask, manual
        
        print("Downloading and preparing AV ground-truth...")
        # Commands from get_public_data.py lines 29-33
        cmd_av = 'wget http://webeye.ophth.uiowa.edu/abramoff/AV_groundTruth.zip ' \
              '&& unzip AV_groundTruth.zip -d DRIVE && rm AV_groundTruth.zip ' \
              '&& mkdir -p DRIVE/manual_av && mv DRIVE/AV_groundTruth/training/av/* DRIVE/manual_av ' \
              '&& mv DRIVE/AV_groundTruth/test/av/* DRIVE/manual_av && rm -r DRIVE/AV_groundTruth'
        os.system(cmd_av)
        
        # Now prepare CSVs
        print("Generating CSV splits...")
        import os.path as osp
        
        path_ims = 'DRIVE/images'
        path_masks = 'DRIVE/mask'
        path_gts = 'DRIVE/manual'
        
        all_im_names = sorted(os.listdir(path_ims))
        all_mask_names = sorted(os.listdir(path_masks))
        all_gt_names = sorted(os.listdir(path_gts))
        
        # append paths
        num_ims = len(all_im_names)
        # We want paths relative to /app/data, so "DRIVE/images/..." works if we set root_path to /app/data
        # OR we use absolute paths: /app/data/DRIVE/images/...
        # dataset_drive.py appends base_dir.
        # If base_dir is /app/data/DRIVE, then paths in CSV should be "images/..."
        # But get_public_data.py used "data/DRIVE/images/..." relative to repo root.
        # Let's use logic compatible with our dataset_drive.py in previous step.
        # dataset_drive.py tries to strip 'data/DRIVE/'.
        # Let's write CSV with "DRIVE/images/..." and adhere to that.
        
        all_im_names = [osp.join(osp.abspath(path_ims), n) for n in all_im_names]
        all_mask_names = [osp.join(osp.abspath(path_masks), n) for n in all_mask_names]
        all_gt_names = [osp.join(osp.abspath(path_gts), n) for n in all_gt_names]
        
        test_im_names = all_im_names[:num_ims//2]
        train_im_names = all_im_names[num_ims//2:]
        
        test_mask_names = all_mask_names[:num_ims//2]
        train_mask_names = all_mask_names[num_ims//2:]
        
        test_gt_names = all_gt_names[:num_ims//2]
        train_gt_names = all_gt_names[num_ims//2:]
        
        # df_drive_all = pd.DataFrame({'im_paths': all_im_names, 'gt_paths': all_gt_names, 'mask_paths': all_mask_names})
        df_drive_train = pd.DataFrame({'im_paths': train_im_names, 'gt_paths': train_gt_names, 'mask_paths': train_mask_names})
        df_drive_test = pd.DataFrame({'im_paths': test_im_names, 'gt_paths': test_gt_names, 'mask_paths': test_mask_names})
        
        # Split train into train/val
        df_drive_train, df_drive_val = df_drive_train[:16], df_drive_train[16:]
        
        df_drive_train.to_csv('DRIVE/train.csv', index=False)
        df_drive_val.to_csv('DRIVE/val.csv', index=False)
        df_drive_test.to_csv('DRIVE/test.csv', index=False)
        
        # AV Split logic (if needed, but TransUNet normally does Segmentation not AV classification unless specified)
        # But let's generate them just in case or if we want to use them.
        # The prompt implies we want to reproduce TransUNet on DRIVE, usually vessel segmentation.
        # But AV was in get_public_data.py.
        # We'll stick to vessel segmentation (manual) for now unless specified otherwise.
        # If we need AV, we swap manual with manual_av in paths.
        
        print("Data preparation complete.")
        data_volume.commit()
    else:
        print("DRIVE data found in volume.")
        
    os.chdir("/app/transunet")

    print("Starting training for DRIVE...")
    print("Current working directory:", os.getcwd())
    print("Data directory content (/app/data/DRIVE):", os.listdir("/app/data/DRIVE"))
    
    # Run the training command
    # DRIVE training
    # batch size 12, lr 0.005
    cmd = "python train.py --dataset DRIVE --vit_name R50-ViT-B_16 --batch_size 1 --base_lr 0.005 --max_epochs 150 --img_size 224"
    
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        raise Exception("Training failed. Check logs for details.")
    
    print("Training finished successfully.")
    
    # Commit the volume to ensure everything is saved
    model_volume.commit()
    

@app.function(
    image=image,
    gpu="t4",
    volumes={
        "/app/model": model_volume,
        "/app/data": data_volume
    }, 
    timeout=86400 # 24 hours
)
def test_transunet():
    import shutil
    import os

    # NOTE: delete redundant dir if exists
    if os.path.exists("/app/model/predictions_drive"):
        shutil.rmtree("/app/model/predictions_drive")
    
    os.chdir("/app/transunet")

    # Run the test command
    # For test, we need to ensure correct path for test dataset is used (handled by Drive_dataset logic using test.csv)
    # We add --is_savenii (though we modified utils to save pngs too)
    test_cmd = "python test.py --dataset DRIVE --vit_name R50-ViT-B_16 --batch_size 1 --base_lr 0.005 --is_savenii --test_save_dir /app/model/predictions --img_size 224"
    
    print(f"Executing Testing: {test_cmd}")
    ret = os.system(test_cmd)
    
    if ret != 0:
        raise Exception("Testing failed. Check logs for details.")
        
    print("Testing finished successfully.")
    print("Predictions saved to /app/model/predictions in the volume.")
    
    # Commit the volume to ensure everything is saved
    model_volume.commit()

# @app.function(
#     image=image,
#     volumes={"/app/model": model_volume},
#     timeout=86400, # Keep running
# )
# @modal.web_server(port=6006)
# def tensorboard_app():
#     import subprocess
#     import time
#     import glob
#     import os
    
#     print("Starting TensorBoard...")
    
#     # Dynamically find the log directory
#     # Pattern based on training args: /app/model/TU_DRIVE224/TU_pretrain*/log
#     search_pattern = "/app/model/TU_DRIVE224/TU_pretrain*/log"
#     matches = glob.glob(search_pattern)
    
#     if matches:
#         log_dir = matches[0]
#         print(f"Found specific log directory: {log_dir}")
#     else:
#         log_dir = "/app/model"
#         print(f"Specific log directory not found, defaulting to: {log_dir}")

#     subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006", "--bind_all"])
#     while True:
#         time.sleep(1)

@app.local_entrypoint()
def main():
    print("Starting remote training and testing for DRIVE...")
    print("Results will be saved to the 'transunet-models' volume.")
    # print("To serve TensorBoard, run: modal serve reproduce_drive.py (in a separate terminal)")
    
    train_transunet.remote()
    test_transunet.remote()
