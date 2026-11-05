#!/usr/bin/env python3
"""
Fork Submodule Helper Script
Forks a public GitHub repository to the user's GitHub account and adds it as a git submodule.
Securely handles GitHub Personal Access Tokens (PAT).
"""

import os
import sys
import json
import time
import getpass
import subprocess
import urllib.request
import urllib.error

def parse_repo(repo_input):
    repo_input = repo_input.strip()
    if repo_input.endswith('.git'):
        repo_input = repo_input[:-4]
    
    # Match patterns like:
    # https://github.com/owner/repo
    # git@github.com:owner/repo
    # owner/repo
    if 'github.com' in repo_input:
        parts = repo_input.split('github.com')[-1].strip('/:')
        owner_repo = parts.split('/')
    else:
        owner_repo = repo_input.strip('/').split('/')
        
    if len(owner_repo) == 2:
        return owner_repo[0], owner_repo[1]
    raise ValueError(f"Invalid repository format: '{repo_input}'. Expected 'owner/repo' or GitHub URL.")

def make_request(url, token, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github+json')
    req.add_header('X-GitHub-Api-Version', '2022-11-28')
    req.add_header('User-Agent', 'python-fork-submodule-skill')
    
    if data is not None:
        req.add_header('Content-Type', 'application/json')
        json_data = json.dumps(data).encode('utf-8')
    else:
        json_data = None
        
    try:
        with urllib.request.urlopen(req, data=json_data) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            err_json = json.loads(err_body)
            message = err_json.get('message', err_body)
        except Exception:
            message = e.reason
        raise RuntimeError(f"GitHub API error ({e.code}): {message}")
    except Exception as e:
        raise RuntimeError(f"Connection failed: {str(e)}")

def main():
    print("====================================================")
    print("GitHub Repository Forker & Submodule Adder")
    print("====================================================")
    
    # 1. Retrieve or Prompt for GitHub Token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GitHub Personal Access Token (PAT) not found in GITHUB_TOKEN environment variable.")
        token = getpass.getpass("Enter your GitHub PAT (requires 'repo' scope, input is hidden): ").strip()
    
    if not token:
        print("Error: GitHub Personal Access Token is required.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Get target repository to fork
    upstream_input = ""
    if len(sys.argv) > 1:
        upstream_input = sys.argv[1]
    else:
        upstream_input = input("Enter the public repository to fork (e.g. 'owner/repo' or GitHub URL): ").strip()
        
    try:
        owner, repo = parse_repo(upstream_input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Get submodule destination path
    default_dest = f"src/references/{repo}"
    dest_path = ""
    if len(sys.argv) > 2:
        dest_path = sys.argv[2]
    else:
        dest_path = input(f"Enter local submodule path [default: {default_dest}]: ").strip()
    if not dest_path:
        dest_path = default_dest

    # Ask the user if they want to use HTTPS or SSH for the submodule link
    protocol = "https"
    if len(sys.argv) > 3:
        protocol = sys.argv[3].lower()
    else:
        proto_choice = input("Select git submodule protocol (1: HTTPS [default], 2: SSH): ").strip()
        if proto_choice == "2" or proto_choice.lower() == "ssh":
            protocol = "ssh"
            
    # 4. Authenticate and get current user
    print("\nAuthenticating with GitHub...")
    try:
        status, user_info = make_request("https://api.github.com/user", token)
        username = user_info['login']
        print(f"Successfully authenticated as: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Check if fork already exists, or request fork
    fork_url = f"https://api.github.com/repos/{owner}/{repo}/forks"
    print(f"Requesting fork of '{owner}/{repo}' to '{username}/{repo}'...")
    try:
        status, fork_info = make_request(fork_url, token, method='POST')
        print("Fork request submitted successfully.")
    except Exception as e:
        print(f"Fork request message: {e}")
        print("Checking if repository already exists in your account...")

    # 6. Poll to verify fork exists
    check_url = f"https://api.github.com/repos/{username}/{repo}"
    fork_verified = False
    print("Verifying if forked repository is ready...")
    for i in range(15):
        time.sleep(2)
        try:
            status, check_info = make_request(check_url, token)
            if status == 200:
                print(f"Fork is ready at: {check_info['html_url']}")
                fork_verified = True
                break
        except Exception:
            pass
            
    if not fork_verified:
        print("Warning: Could not verify if fork is fully initialized. Proceeding with submodule creation anyway...")
        
    # 7. Add submodule
    if protocol == "ssh":
        submodule_url = f"git@github.com:{username}/{repo}.git"
    else:
        submodule_url = f"https://github.com/{username}/{repo}.git"
        
    print(f"\nAdding submodule:")
    print(f"  URL:  {submodule_url}")
    print(f"  Path: {dest_path}")
    
    # Check if directory already exists
    if os.path.exists(dest_path):
        print(f"Warning: Destination path '{dest_path}' already exists.")
        confirm = input("Do you want to proceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            sys.exit(0)
            
    cmd = ["git", "submodule", "add", submodule_url, dest_path]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"\nError adding submodule: {res.stderr.strip()}", file=sys.stderr)
        print("Tip: If the submodule is already registered in git index, you may need to deintegrate it first.", file=sys.stderr)
        sys.exit(1)
        
    print("\nSubmodule added successfully. Initializing and updating submodules...")
    sub_init = subprocess.run(["git", "submodule", "update", "--init", "--recursive"], capture_output=True, text=True)
    if sub_init.returncode == 0:
        print("Git submodule update complete.")
    else:
        print(f"Warning: Git submodule update returned an error: {sub_init.stderr.strip()}")
        
    print("\n====================================================")
    print("Task completed successfully!")
    print("Remember to commit the new submodule registration: ")
    print(f"  git commit -m \"Add submodule {repo} at {dest_path}\"")
    print("====================================================")

if __name__ == "__main__":
    main()
