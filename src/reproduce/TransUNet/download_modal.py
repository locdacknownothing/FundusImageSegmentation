import argparse
import os
import subprocess
import sys

def check_modal_installed():
    """Check if modal CLI works or can be invoked via python -m modal"""
    # Try direct command
    try:
        subprocess.run(["modal", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "modal"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try python module
    try:
        subprocess.run([sys.executable, "-m", "modal", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return f"{sys.executable} -m modal"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

def download():
    parser = argparse.ArgumentParser(description="Download file/folder from modal volume")
    parser.add_argument("--volume", default="transunet-models", type=str, 
                        help="Volume name in modal. Default: transunet-models.")
    parser.add_argument("--remote_path", type=str, 
                        help="Remote path in the volume")
    parser.add_argument("--output_root", type=str, default="./result",
                        help="Local root directory to preserve structure. Downloads will be placed in nested directories matching remote structure.")
    
    args = parser.parse_args()
    
    modal_cmd_base = check_modal_installed()
    if not modal_cmd_base:
        print("Error: 'modal' command not found. Please install modal (pip install modal) and ensure it is in your PATH.")
        sys.exit(1)
        
    volume_name = args.volume
    remote_path = args.remote_path
    
    # Directory structure handling
    # We want to preserve the folder structure from the remote path.
    # If remote path is "A/B/log", and output_root is "./result", we want "./result/A/B/log".
    # modal volume get will download "log" into the destination directory if it exists.
    # So we need to ensure "./result/A/B" exists and use that as the destination.
    
    parent_remote = os.path.dirname(remote_path) # e.g., TU_DRIVE224/TU_pretrain...
    local_parent = os.path.join(args.output_root, parent_remote)
    
    print(f"Ensuring local directory exists: {local_parent}")
    os.makedirs(local_parent, exist_ok=True)
    
    # Construct command
    cmd_parts = modal_cmd_base.split()
    cmd = cmd_parts + [
        "volume", "get", 
        volume_name, 
        remote_path, 
        local_parent
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print(f"Downloading {remote_path} from volume {volume_name}...")
    
    try:
        subprocess.check_call(cmd)
        final_path = os.path.join(local_parent, os.path.basename(remote_path))
        print(f"\nSUCCESS: Downloaded to '{final_path}'")
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: Command failed with error: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    download()
