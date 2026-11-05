---
name: fork-submodule
description: Forks a public GitHub repository to the user's GitHub account and adds the fork as a git submodule inside src/references/ securely.
license: MIT
metadata:
  version: "1.0"
  author: Antigravity
---

# Fork and Add GitHub Submodule

This skill automates the process of forking a public GitHub repository to your personal GitHub account and adding that fork as a git submodule inside the [src/references](src/references) folder.

It has been designed with security in mind, ensuring your GitHub credentials (Personal Access Tokens) are handled in-memory only and never written to repository configuration files.

## Prerequisites

1. **GitHub Personal Access Token (PAT)**:
   - Generate a token at [GitHub settings](https://github.com/settings/tokens).
   - The token needs the **`repo`** scope (to fork repositories and access your account details).
2. **Python 3**:
   - The helper script uses Python 3's standard library (`urllib` and `subprocess`), requiring no external dependencies (like `requests` or `jq`).

## Usage

You can run this skill either interactively or non-interactively using the helper script located at [fork_submodule.py](skills/fork-submodule/scripts/fork_submodule.py).

### Option 1: Interactive Execution (Recommended)

Run the script without arguments. It will securely prompt you for your token, the repository, and the submodule path:

```bash
# Optional: Set the token in your environment to skip the token prompt
export GITHUB_TOKEN="your_personal_access_token"

# Run the helper script
python3 skills/fork-submodule/scripts/fork_submodule.py
```

If `GITHUB_TOKEN` is not set, the script will prompt you for it securely using a hidden prompt (characters will not echo to the screen).

### Option 2: CLI Arguments (Non-interactive)

You can pass arguments directly to the script:

```bash
python3 skills/fork-submodule/scripts/fork_submodule.py <upstream_repo> <submodule_path> <protocol>
```

- `<upstream_repo>`: The target repository to fork (e.g. `owner/repo` or a full URL like `https://github.com/owner/repo.git`).
- `<submodule_path>`: The local path to place the submodule (e.g., `src/references/my-forked-repo`).
- `<protocol>`: The Git protocol to use for the submodule (`https` or `ssh`). Defaults to `https`.

Example:
```bash
python3 skills/fork-submodule/scripts/fork_submodule.py agaldran/lwnet src/references/lwnet_forked https
```

---

## Security Guidelines

To keep your credentials secure:
1. **No Hardcoded Tokens**: Never commit your Personal Access Token (PAT) to any file in the repository (including [.gitmodules](.gitmodules) or shell history).
2. **In-Memory Auth**: The helper script retrieves your token via the `GITHUB_TOKEN` environment variable or prompts you using Python's `getpass` module. It is only held in the memory of the running process.
3. **Clean Git URLs**: The git submodule URL registered in [.gitmodules](.gitmodules) will be a clean public URL (e.g., `https://github.com/your-username/repo.git`). It does **not** embed the token inside the URL. Git will authenticate using your system's credential manager or SSH key during subsequent updates.

## Troubleshooting

- **Submodule already exists in index**:
  If you get an error that the path already exists in the index, run:
  ```bash
  git rm -f <submodule_path>
  rm -rf .git/modules/<submodule_path>
  ```
  Then run the script again.
- **Authentication errors (401 / 403 / 404)**:
  Ensure your GitHub Personal Access Token is valid, has not expired, and has the `repo` scope enabled.
