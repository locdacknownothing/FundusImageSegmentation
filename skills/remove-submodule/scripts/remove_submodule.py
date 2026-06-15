#!/usr/bin/env python3
"""
Remove Submodule Helper Script
Completely and cleanly removes a git submodule from the project.
"""

import os
import sys
import shutil
import subprocess
import configparser

def get_active_submodules():
    gitmodules_path = ".gitmodules"
    if not os.path.exists(gitmodules_path):
        return []
    
    config = configparser.ConfigParser()
    try:
        config.read(gitmodules_path)
    except Exception as e:
        print(f"Error reading .gitmodules: {e}", file=sys.stderr)
        return []
        
    submodules = []
    for section in config.sections():
        if section.startswith('submodule '):
            path = config[section].get('path')
            if path:
                submodules.append(path)
    return submodules

def remove_submodule(submodule_path):
    print(f"\nProceeding to completely remove submodule at '{submodule_path}'...")
    
    # 1. deinit submodule
    deinit_cmd = ["git", "submodule", "deinit", "-f", submodule_path]
    print(f"Running: {' '.join(deinit_cmd)}")
    res_deinit = subprocess.run(deinit_cmd, capture_output=True, text=True)
    if res_deinit.returncode != 0:
        print(f"Warning during git submodule deinit: {res_deinit.stderr.strip()}")
    else:
        print("Submodule de-initialized from .git/config.")
        
    # 2. git rm
    rm_cmd = ["git", "rm", "-f", submodule_path]
    print(f"Running: {' '.join(rm_cmd)}")
    res_rm = subprocess.run(rm_cmd, capture_output=True, text=True)
    if res_rm.returncode != 0:
        print(f"Warning during git rm: {res_rm.stderr.strip()}")
    else:
        print(f"Removed submodule path '{submodule_path}' from index and working tree.")
        
    # 3. remove .git/modules directory
    git_modules_dir = os.path.join(".git", "modules", submodule_path)
    if os.path.exists(git_modules_dir):
        print(f"Deleting internal git directory: {git_modules_dir}")
        try:
            shutil.rmtree(git_modules_dir)
            print("Internal git directory deleted successfully.")
        except Exception as e:
            print(f"Error deleting internal git directory {git_modules_dir}: {e}", file=sys.stderr)
    else:
        print(f"No internal git directory found at '{git_modules_dir}' (already removed or never initialized).")
        
    # 4. Clean up any leftover untracked directory if it remains
    if os.path.exists(submodule_path):
        print(f"Note: Submodule folder '{submodule_path}' still exists (likely due to untracked files).")
        confirm = input(f"Do you want to delete the remaining directory '{submodule_path}'? (y/n): ").strip().lower()
        if confirm == 'y':
            try:
                shutil.rmtree(submodule_path)
                print(f"Deleted remaining folder '{submodule_path}'.")
            except Exception as e:
                print(f"Error deleting folder '{submodule_path}': {e}", file=sys.stderr)
                
    print("\n====================================================")
    print("Submodule removed completely and cleanly!")
    print("Remember to commit the removal: ")
    print(f"  git commit -m \"Remove submodule {os.path.basename(submodule_path)}\"")
    print("====================================================")

def main():
    print("====================================================")
    print("Git Submodule Cleaner & Remover")
    print("====================================================")
    
    submodules = get_active_submodules()
    
    # Check if a path was passed as command line argument
    target_path = None
    if len(sys.argv) > 1:
        target_path = sys.argv[1].strip()
        if target_path not in submodules:
            print(f"Warning: '{target_path}' is not listed in .gitmodules.")
            confirm = input("Are you sure you want to attempt removing it? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Aborted.")
                sys.exit(0)
    else:
        if not submodules:
            print("No active submodules detected in .gitmodules.")
            target_path = input("Enter the path of the submodule you want to remove: ").strip()
            if not target_path:
                print("No path provided. Aborting.")
                sys.exit(0)
        else:
            print("Detected submodules:")
            for idx, path in enumerate(submodules, 1):
                print(f"  [{idx}] {path}")
            
            choice = input("\nSelect a submodule number to remove, or enter a custom path: ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(submodules):
                    target_path = submodules[choice_idx]
                else:
                    print("Invalid selection index. Aborting.")
                    sys.exit(1)
            except ValueError:
                # User entered a custom path
                if choice:
                    target_path = choice
                else:
                    print("No path selected. Aborting.")
                    sys.exit(1)
                    
    remove_submodule(target_path)

if __name__ == "__main__":
    main()
