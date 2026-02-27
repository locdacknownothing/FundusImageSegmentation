import modal
import os
import os.path as osp
import shutil

from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import resize

app = modal.App("transunet-hrf")

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

    # 2. Check and Prepare HRF Data
    dataset_path = "/app/data/HRF"
    # shutil.rmtree(dataset_path)

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("HRF data not found. Downloading and preparing...")
        os.makedirs(dataset_path, exist_ok=True)
        
        print("Downloading base fundus datasets from deepdyn repo...")
        os.chdir("/app/data")
        cmd_download_base = "curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data"
        os.system(cmd_download_base)

        call = 'wget https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip ' \
            '&& unzip all.zip -d HRF && mv HRF/manual1 HRF/manual' \
            '&& rm all.zip'
        os.system(call)

        # Generate CSVs
        print("Generating CSV splits...")
        os.chdir("/app")
        path_ims = 'data/HRF/images'
        path_masks = 'data/HRF/mask'
        path_gts = 'data/HRF/manual'

        path_ims_resized = 'data/HRF/images_resized'
        os.makedirs(path_ims_resized, exist_ok=True)
        path_masks_resized = 'data/HRF/mask_resized'
        os.makedirs(path_masks_resized, exist_ok=True)
        path_gts_resized = 'data/HRF/manual_resized'
        os.makedirs(path_gts_resized, exist_ok=True)

        all_im_names = sorted(os.listdir(path_ims))
        all_mask_names = sorted(os.listdir(path_masks))
        all_gt_names = sorted(os.listdir(path_gts))

        # append paths
        num_ims = len(all_im_names)
        all_im_names = [osp.join(path_ims, n) for n in all_im_names]
        all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
        all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

        df_hrf_all = pd.DataFrame({'im_paths': all_im_names,
                                    'gt_paths': all_gt_names,
                                    'mask_paths': all_mask_names})

        train_im_names = all_im_names[   :3*5]
        test_im_names =  all_im_names[3*5:   ]

        train_mask_names = all_mask_names[   :3*5]
        test_mask_names =  all_mask_names[3*5:   ]

        train_gt_names = all_gt_names[   :3*5]
        test_gt_names =  all_gt_names[3*5:   ]

        # use smaller images for training **only** on HRF
        train_im_names_resized = [n.replace(path_ims, path_ims_resized) for n in train_im_names]
        train_mask_names_resized = [n.replace(path_masks, path_masks_resized) for n in train_mask_names]
        train_gt_names_resized = [n.replace(path_gts, path_gts_resized) for n in train_gt_names]

        df_hrf_train = pd.DataFrame({'im_paths': train_im_names_resized,
                                    'gt_paths': train_gt_names_resized,
                                    'mask_paths': train_mask_names_resized})
        df_hrf_test = pd.DataFrame({'im_paths': test_im_names,
                                    'gt_paths': test_gt_names,
                                    'mask_paths': test_mask_names})

        num_ims = len(df_hrf_train)
        tr_ims = int(0.8*num_ims)
        df_hrf_train, df_hrf_val = df_hrf_train[:tr_ims], df_hrf_train[tr_ims:]

        df_hrf_train.to_csv('data/HRF/train.csv', index=False)
        df_hrf_val.to_csv('data/HRF/val.csv', index=False)
        df_hrf_test.to_csv('data/HRF/test.csv', index=False)
        df_hrf_all.to_csv('data/HRF/test_all.csv', index=False)

        # need this for AUC analysis on the training set
        df_hrf_train_full_res = pd.DataFrame({'im_paths': train_im_names,
                                    'gt_paths': train_gt_names,
                                    'mask_paths': train_mask_names})
        df_hrf_train_full_res, df_hrf_val_full_res = df_hrf_train_full_res[:tr_ims], df_hrf_train_full_res[tr_ims:]
        df_hrf_train_full_res.to_csv('data/HRF/train_full_res.csv', index=False)
        df_hrf_val_full_res.to_csv('data/HRF/val_full_res.csv', index=False)

        print('Resizing HRF images (**only** for training, but we resize all because A/V training set is test set on Vessels)\n')
        for i in tqdm(range(len(all_im_names))):
            im_name = all_im_names[i]
            im_name_out = im_name.replace('/images/', '/images_resized/')
            im = Image.open(im_name)
            im_res = resize(im, size=(im.size[1] // 2, im.size[0] // 2), interpolation=Image.BICUBIC)
            im_res.save(im_name_out)

            mask_name = im_name.replace('/images/', '/mask/').replace('.JPG', '_mask.tif').replace('.jpg', '_mask.tif')
            mask_name_out = mask_name.replace('/mask/', '/mask_resized/')
            mask = Image.open(mask_name)
            mask_res = resize(mask, size=(mask.size[1] // 2, mask.size[0] // 2), interpolation=Image.NEAREST)
            # get rid of three channels in mask
            mask = Image.fromarray(np.array(mask))
            mask_res = Image.fromarray(np.array(mask_res))
            mask.save(mask_name)
            mask_res.save(mask_name_out)

            gt_name = im_name.replace('/images/', '/manual/').replace('.JPG', '.tif').replace('.jpg', '.tif')
            gt_name_out = gt_name.replace('/manual/', '/manual_resized/')
            gt = Image.open(gt_name)
            gt_res = resize(gt, size=(gt.size[1] // 2, gt.size[0] // 2), interpolation=Image.NEAREST)
            gt_res.save(gt_name_out)
        print('HRF prepared')

        data_volume.commit()
    else:
        print("HRF data found in volume.")
        
    os.chdir("/app")

    print("Starting training for HRF...")
    print("Current working directory:", os.getcwd())
    print("Data directory content (/app/data/HRF):", os.listdir("/app/data/HRF"))
    
    cmd = "python transunet/train.py --dataset HRF --vit_name R50-ViT-B_16 --batch_size 32 --base_lr 0.005 --max_epochs 2000 --img_size 224"
    
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
#     test_cmd = "python test.py --dataset HRF --vit_name R50-ViT-B_16 --batch_size 2 --base_lr 0.005 --is_savenii --test_save_dir /app/model/predictions --img_size 224"
    
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
    print("Starting remote training and testing for HRF...")
    print("Results will be saved to the 'transunet-models' volume.")
    # print("To serve TensorBoard, run: modal serve reproduce_drive.py (in a separate terminal)")
    
    train_transunet.remote()
    # test_transunet.remote()
