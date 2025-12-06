import modal
import os

app = modal.App("transunet-reproduce-detached")

# Volume for checkpoints to persist results
model_volume = modal.Volume.from_name("transunet-models", create_if_missing=True)

# Define the image with dependencies and local files
# We use python 3.9 as recommended by the README
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("wget", "git", "unzip")
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
        "gdown"
    )
    # Download Google pre-trained ViT models to a separate directory in the image
    # We cannot put them directly in /app/model because we want to mount a volume there.
    .run_commands(
        "wget -q https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz",
        "mkdir -p /assets/vit_checkpoint/imagenet21k",
        "mv R50+ViT-B_16.npz /assets/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    )
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
    # gpu="t4", # T4 is chosen for compatibility with PyTorch 1.4.0 and cost-effectiveness (~$0.60/hr).
              # Newer GPUs like A10G require CUDA 11+, which might conflict with torch==1.4.0.
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
    if not os.path.exists("/app/data/Synapse"):
        print("Data not found in /app/data/Synapse. Downloading...")
        
        # BTCV Dataset URL (from download_data.py)
        url = 'https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd'
        output_folder = '/app/data/temp_download'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        try:
            # Download the folder
            gdown.download_folder(url=url, output=output_folder, quiet=False, use_cookies=False)
            
            # The download creates a folder (e.g., BTCV) or files inside output_folder.
            # Based on user's local experience, it contains 'project_TransUNet.zip'.
            zip_path = os.path.join(output_folder, 'project_TransUNet.zip')
            
            if not os.path.exists(zip_path):
                # Try finding it if the folder name is part of the path
                # Sometimes gdown downloads into a subfolder named after the Drive folder
                for root, dirs, files in os.walk(output_folder):
                    if 'project_TransUNet.zip' in files:
                        zip_path = os.path.join(root, 'project_TransUNet.zip')
                        break
            
            if os.path.exists(zip_path):
                print(f"Unzipping {zip_path}...")
                os.system(f"unzip -q {zip_path} -d /app/data/temp_unzip")
                
                # Move the Synapse folder to /app/data/Synapse
                # Structure in zip: project_TransUNet/data/Synapse
                source_synapse = "/app/data/temp_unzip/project_TransUNet/data/Synapse"
                if os.path.exists(source_synapse):
                    print(f"Moving {source_synapse} to /app/data/Synapse")
                    shutil.move(source_synapse, "/app/data/Synapse")
                else:
                    print(f"Could not find Synapse folder at {source_synapse}. Listing unzip dir:")
                    os.system("ls -R /app/data/temp_unzip")
                    raise Exception("Data structure mismatch after unzip.")
            else:
                print(f"project_TransUNet.zip not found in {output_folder}")
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
        print("Data found at /app/data/Synapse.")

    print("Starting training...")
    print("Current working directory:", os.getcwd())
    print("Data directory content (/app/data):", os.listdir("/app/data"))
    
    # Run the training command
    # We reduce batch size to 12 and learning rate to 0.005 to be safe on memory
    # (Default is bs=24, lr=0.01)
    cmd = "python train.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 12 --base_lr 0.005"
    
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        raise Exception("Training failed. Check logs for details.")
    
    print("Training finished successfully.")
    
@app.function(
    image=image,
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
    # We must use the same batch_size and base_lr so that test.py looks in the correct directory for the model.
    # We add --is_savenii to save the predictions.
    # We save predictions to /app/model/predictions so they persist in the model_volume.
    test_cmd = "python test.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 12 --base_lr 0.005 --is_savenii --test_save_dir /app/model/predictions"
    
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
    print("Starting remote training and testing...")
    # print("Please run this command with '--detach' to keep it running in the background:")
    # print("  modal run --detach reproduce_transunet.py")
    print("Results will be saved to the 'transunet-models' volume.")
    
    # We use .remote() which blocks locally.
    # If run with --detach, the local process will disconnect but the remote task will continue.
    # train_transunet.remote()
    test_transunet.remote()
