import modal
import os

app = modal.App("transunet-acdc")

# Volume for checkpoints to persist results
model_volume = modal.Volume.from_name("transunet-models", create_if_missing=True)

# Define the image with dependencies and local files
# We use python 3.9 as recommended by the README
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("wget", "git", "unzip", "libgl1", "libglib2.0-0")
    # Install dependencies. We let pip resolve versions compatible with the system/python
    # unless strict adherence to requirements.txt is needed and possible.
    # torch==1.4.0 is quite old and might have CUDA compatibility issues on modern cloud GPUs.
    # We will try installing modern torch first.
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
        "scikit-image"
    )
    # Download Google pre-trained ViT models to a separate directory in the image
    # We cannot put them directly in /app/model because we want to mount a volume there.
    # .run_commands(
    #     "wget -q https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz",
    #     "mkdir -p /assets/vit_checkpoint/imagenet21k",
    #     "mv R50+ViT-B_16.npz /assets/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    # )
    # Mount the source code
    # Local: src/references/TransUNet
    # Remote: /app/transunet
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
    import sys
    import gdown
    import shutil
    
    # Change working directory to the code
    os.chdir("/app/transunet")
    
    # 1. Prepare Pre-trained Weights
    # The training script expects weights at ../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
    # relative to /app/transunet, which resolves to /app/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
    # Since /app/model is a volume, it might be empty initially.
    
    pretrained_target_dir = "/app/model/vit_checkpoint/imagenet21k"
    pretrained_target_file = os.path.join(pretrained_target_dir, "R50+ViT-B_16.npz")
    
    if not os.path.exists(pretrained_target_file):
        print(f"Pre-trained weights not found at {pretrained_target_file}. Copying from image assets...")
        os.makedirs(pretrained_target_dir, exist_ok=True)
        shutil.copy("/assets/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz", pretrained_target_file)
        print("Pre-trained weights copied.")
        # Commit the volume to save these weights
        model_volume.commit()
    else:
        print("Pre-trained weights found in volume.")

    # 2. Check and prepare data
    # For ACDC, we expect data in /app/data/ACDC
    if not os.path.exists("/app/data/ACDC"):
        print("Data not found in /app/data/ACDC. Downloading...")
        
        # ACDC Dataset URL
        url = 'https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4'
        output_folder = '/app/data/temp_download'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        try:
            # Download the folder
            gdown.download_folder(url=url, output=output_folder, quiet=False, use_cookies=False)
            
            # The download creates a folder (e.g., ACDC) or files inside output_folder.
            # We look for 'transunet_ACDC.zip'
            zip_path = os.path.join(output_folder, 'transunet_ACDC.zip')
            
            if not os.path.exists(zip_path):
                # Try finding it if the folder name is part of the path
                for root, dirs, files in os.walk(output_folder):
                    if 'transunet_ACDC.zip' in files:
                        zip_path = os.path.join(root, 'transunet_ACDC.zip')
                        break
            
            if os.path.exists(zip_path):
                print(f"Unzipping {zip_path}...")
                # Unzip to /app/data/ACDC, default by the zip structure
                os.system(f"unzip -q {zip_path} -d /app/data")
                
            else:
                print(f"transunet_ACDC.zip not found in {output_folder}")
                os.system(f"ls -R {output_folder}")
                raise Exception("Download failed or file missing.")
                
        except Exception as e:
            print(f"Error during data preparation: {e}")
            raise e
        finally:
            # Cleanup
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            if os.path.exists("/app/data/temp_unzip"):
                shutil.rmtree("/app/data/temp_unzip")
                
        # Commit the volume changes
        data_volume.commit()
    else:
        print("Data found at /app/data/ACDC.")

    print("Starting training...")
    print("Current working directory:", os.getcwd())
    if os.path.exists("/app/data/ACDC"):
        print("Data directory content (/app/data/ACDC):", os.listdir("/app/data/ACDC"))
    
    # Run the training command
    # ACDC training
    # We reduce batch size to 12 and learning rate to 0.005 to be safe on memory
    cmd = "python train.py --dataset ACDC --vit_name R50-ViT-B_16 --batch_size 12 --base_lr 0.005"
    
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        raise Exception("Training failed. Check logs for details.")
    
    print("Training finished successfully.")

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
    # Change working directory to the code
    os.chdir("/app/transunet")

    print("Starting testing...")
    print("Current working directory:", os.getcwd())

    # Run the test command
    test_cmd = "python test.py --dataset ACDC --vit_name R50-ViT-B_16 --batch_size 12 --base_lr 0.005 --is_savenii --test_save_dir /app/model/predictions"
    
    print(f"Executing Testing: {test_cmd}")
    ret = os.system(test_cmd)
    
    if ret != 0:
        raise Exception("Testing failed. Check logs for details.")
        
    print("Testing finished successfully.")
    print("Predictions saved to /app/model/predictions in the volume.")
    
    # Commit the volume to ensure everything is saved
    model_volume.commit()

@app.local_entrypoint()
def main():
    print("Starting remote training and testing for ACDC...")
    # print("Please run this command with '--detach' to keep it running in the background:")
    # print("  modal run --detach reproduce_acdc.py")
    print("Results will be saved to the 'transunet-models' volume.")
    
    # train_transunet.remote()
    test_transunet.remote()
