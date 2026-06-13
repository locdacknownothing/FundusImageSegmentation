import modal
import shutil
from pathlib import Path

app = modal.App()

volume = modal.Volume.from_name("transunet-models", create_if_missing=False)

@app.function(volumes={"/app/model": volume})
def copy_dir():
    src = Path("/app/model/TU_CHASEDB224/TU_pretrain_R50-ViT-B_16_skip3_epo2000_bs4_lr0.005_224")
    dst = Path("/app/model/TU_CHASEDB224/TU_pretrain_R50-ViT-B_16_skip3_ChaseDBDataset_epo2000_bs4_lr0.005_224")
    shutil.move(src, dst)

# @app.function(volumes={"/app/model": volume})
# def remove_path():
#     parent_dir = Path("/app/model/TU_HRF224")
    
#     for path in parent_dir.glob("TU_pretrain_R50-ViT-B_16_skip3_epo2000_bs64_lr0.005_224"):
#         if path.is_file():
#             path.unlink()
#         elif path.is_dir():
#             shutil.rmtree(path)

#         print(f"Removed: {path}")
