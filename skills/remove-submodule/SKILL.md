---
name: remove-submodule
description: Completely and cleanly removes a git submodule from the project index, config, filesystem, and internal metadata.
license: MIT
metadata:
  version: "1.0"
  author: Antigravity
---

# Complete Submodule Removal

This skill provides the steps and helper scripts to completely and cleanly remove a git submodule from the project repository. 

Simply deleting the folder is not enough, as git stores submodule references in several configuration files and directories. This skill cleans up all of them.

## Prerequisites

- **Python 3**:
  - The helper script uses Python 3's standard library (`shutil`, `subprocess`, `configparser`), requiring no external dependencies.
- **Git**:
  - The repository must be a valid git repository.

---

## Usage

You can remove a submodule using the automated helper script or manually using step-by-step git commands.

### Option 1: Automated Script (Recommended)

Run the helper script [remove_submodule.py](file:///dscilab_hvch_dacloc/workspace/DacLoc/FundusImageSegmentation/skills/remove-submodule/scripts/remove_submodule.py):

#### Interactive Mode:
Run the script without arguments. It will read [.gitmodules](file:///dscilab_hvch_dacloc/workspace/DacLoc/FundusImageSegmentation/.gitmodules), list all active submodules, and let you select one to delete:
```bash
python3 skills/remove-submodule/scripts/remove_submodule.py
```

#### Non-interactive Mode:
Pass the path of the submodule directly as a command-line argument:
```bash
python3 skills/remove-submodule/scripts/remove_submodule.py <submodule_path>
```
Example:
```bash
python3 skills/remove-submodule/scripts/remove_submodule.py src/references/TransUNet
```

---

### Option 2: Manual Removal (Step-by-Step)

If you prefer to run the commands yourself, follow these steps:

1. **De-initialize the submodule** (removes it from `.git/config`):
   ```bash
   git submodule deinit -f <submodule_path>
   ```
   *Example:* `git submodule deinit -f src/references/TransUNet`

2. **Remove the submodule directory from the git index and working tree** (removes the reference from [.gitmodules](file:///dscilab_hvch_dacloc/workspace/DacLoc/FundusImageSegmentation/.gitmodules) and deletes the files):
   ```bash
   git rm -f <submodule_path>
   ```
   *Example:* `git rm -f src/references/TransUNet`

3. **Delete the internal git metadata directory** (prevents stale repository tracking data):
   ```bash
   rm -rf .git/modules/<submodule_path>
   ```
   *Example:* `rm -rf .git/modules/src/references/TransUNet`

4. **Verify and Commit**:
   Check if the submodule entry has been deleted from [.gitmodules](file:///dscilab_hvch_dacloc/workspace/DacLoc/FundusImageSegmentation/.gitmodules) and then commit the changes:
   ```bash
   git commit -m "Remove submodule <submodule_name>"
   ```

---

## Troubleshooting & Verification

To verify that the submodule has been completely removed:
- Check that the submodule path no longer exists in [.gitmodules](file:///dscilab_hvch_dacloc/workspace/DacLoc/FundusImageSegmentation/.gitmodules).
- Run `git config --list` to check that no config lines start with `submodule.<submodule_path>`.
- Verify that the folder `.git/modules/<submodule_path>` has been deleted.
