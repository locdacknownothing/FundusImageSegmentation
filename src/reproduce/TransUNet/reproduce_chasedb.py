import modal
import os

app = modal.App("transunet-chasedb")

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
    .pip_install("tiler")
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

    # 2. Check and Prepare CHASEDB Data
    dataset_path = "/app/data/CHASEDB"

    # # DEBUG: Remove existing data
    # if os.path.exists(drive_data_path):
    #     shutil.rmtree(drive_data_path)
    #     data_volume.commit()

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path) or not os.path.exists(os.path.join(dataset_path, "train.csv")):
        print("CHASEDB data not found. Downloading and preparing...")
        os.makedirs(dataset_path, exist_ok=True)
        
        print("Downloading base fundus datasets from deepdyn repo...")
        os.chdir("/app/data")
        cmd_download_base = "curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data"
        os.system(cmd_download_base)
            
        # Generate CSVs
        print("Generating CSV splits...")
        import os.path as osp
        
        path_ims = 'CHASEDB/images'
        # path_masks = 'CHASEDB/masks'
        path_gts = 'CHASEDB/manual'
        
        all_im_names = sorted(os.listdir(path_ims))
        # all_mask_names = sorted(os.listdir(path_masks))
        all_gt_names = sorted([n for n in os.listdir(path_gts) if '1st' in n])
        
        # Use absolute paths
        all_im_names = [osp.join(osp.abspath(path_ims), n) for n in all_im_names]
        # all_mask_names = [osp.join(osp.abspath(path_masks), n) for n in all_mask_names]
        all_gt_names = [osp.join(osp.abspath(path_gts), n) for n in all_gt_names]
        
        # Split: first 8 for train, rest for test
        train_im_names = all_im_names[:8]
        test_im_names = all_im_names[8:]
        
        # train_mask_names = all_mask_names[:8]
        # test_mask_names = all_mask_names[8:]
        
        train_gt_names = all_gt_names[:8]
        test_gt_names = all_gt_names[8:]
        
        df_chasedb_train = pd.DataFrame({'im_paths': train_im_names, 'gt_paths': train_gt_names})
        df_chasedb_test = pd.DataFrame({'im_paths': test_im_names, 'gt_paths': test_gt_names})
        
        # Split train into train/val (80/20)
        num_train = len(df_chasedb_train)
        tr_ims = int(0.8 * num_train)
        df_chasedb_train, df_chasedb_val = df_chasedb_train[:tr_ims], df_chasedb_train[tr_ims:]
        
        df_chasedb_train.to_csv('CHASEDB/train.csv', index=False)
        df_chasedb_val.to_csv('CHASEDB/val.csv', index=False)
        df_chasedb_test.to_csv('CHASEDB/test.csv', index=False)
        
        print("Data preparation complete.")
        data_volume.commit()
    else:
        print("CHASEDB data found in volume.")
        
    os.chdir("/app")

    print("Starting training for CHASEDB...")
    print("Current working directory:", os.getcwd())
    print("Data directory content (/app/data/CHASEDB):", os.listdir("/app/data/CHASEDB"))
    
    cmd = "python transunet/train.py --dataset CHASEDB --vit_name R50-ViT-B_16 --batch_size 16 --base_lr 0.005 --max_epochs 2000 --img_size 224"
    
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        raise Exception("Training failed. Check logs for details.")
    
    print("Training finished successfully.")
    
    # Commit the volume to ensure everything is saved
    model_volume.commit()
    

# @app.function(
#     image=image,
#     gpu="t4",
#     volumes={
#         "/app/model": model_volume,
#         "/app/data": data_volume
#     }, 
#     timeout=86400 # 24 hours
# )
# def test_transunet():
#     import shutil
#     import os

#     # NOTE: delete redundant dir if exists
#     if os.path.exists("/app/model/predictions_drive"):
#         shutil.rmtree("/app/model/predictions_drive")
    
#     os.chdir("/app/transunet")

#     # Run the test command
#     # For test, we need to ensure correct path for test dataset is used (handled by Drive_dataset logic using test.csv)
#     # We add --is_savenii (though we modified utils to save pngs too)
#     test_cmd = "python test.py --dataset CHASEDB --vit_name R50-ViT-B_16 --batch_size 2 --base_lr 0.005 --is_savenii --test_save_dir /app/model/predictions --img_size 224"
    
#     print(f"Executing Testing: {test_cmd}")
#     ret = os.system(test_cmd)
    
#     if ret != 0:
#         raise Exception("Testing failed. Check logs for details.")
        
#     print("Testing finished successfully.")
#     print("Predictions saved to /app/model/predictions in the volume.")
    
#     # Commit the volume to ensure everything is saved
#     model_volume.commit()


@app.local_entrypoint()
def main():
    print("Starting remote training and testing for CHASEDB...")
    print("Results will be saved to the 'transunet-models' volume.")
    # print("To serve TensorBoard, run: modal serve reproduce_drive.py (in a separate terminal)")
    
    train_transunet.remote()
    # test_transunet.remote()
