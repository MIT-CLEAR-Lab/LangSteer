#!/bin/bash
# Install the CALVIN simulator assets into the project environment.
#
# `uv sync` installs calvin_env (pinned in pyproject.toml), but the package does not
# ship its data/ directory (URDFs, meshes, scene configs). This script copies data/
# from the same calvin_env commit into the installed package, and the Franka Panda
# URDFs into pybullet_data. Run it from the repository root after `uv sync`; it uses
# the environment `uv run` uses (UV_PROJECT_ENVIRONMENT if set).

set -euo pipefail

CALVIN_ENV_COMMIT=797142c588c21e76717268b7b430958dbd13bf48   # = [tool.uv.sources] calvin-env

PKG_PATHS=$(uv run --frozen python -c "
import os, calvin_env, pybullet_data
print(os.path.dirname(calvin_env.__file__)); print(pybullet_data.getDataPath())")
CALVIN_PKG=$(echo "$PKG_PATHS" | sed -n 1p)
PYBULLET_DATA=$(echo "$PKG_PATHS" | sed -n 2p)
echo "calvin_env package: $CALVIN_PKG"
echo "pybullet_data:      $PYBULLET_DATA"

if [ -d "$CALVIN_PKG/data/franka_panda" ]; then
    echo "calvin_env data/ already installed"
else
    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT
    git clone --quiet https://github.com/mees/calvin_env.git "$TMP_DIR/calvin_env"
    git -C "$TMP_DIR/calvin_env" checkout --quiet "$CALVIN_ENV_COMMIT"
    mkdir -p "$CALVIN_PKG/data"
    cp -r "$TMP_DIR/calvin_env/data/"* "$CALVIN_PKG/data/"
    echo "installed calvin_env data/ from commit $CALVIN_ENV_COMMIT"
fi

rm -rf "$PYBULLET_DATA/franka_panda"
cp -r "$CALVIN_PKG/data/franka_panda" "$PYBULLET_DATA/"
echo "copied the Franka Panda URDFs to $PYBULLET_DATA/franka_panda"

cat <<'EOF'

Next:
  1. Download the CALVIN task_D_D dataset (its validation/ split is what the
     evaluation samples scenes from):
       wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip && unzip task_D_D.zip
  2. export CALVIN_DATASET_PATH=/path/to/task_D_D
  3. Put the checkpoints under checkpoints/ (see README.md) and run
       uv run python scripts/run_overloaded_evaluation.py matrix --dataset-path $CALVIN_DATASET_PATH --output-dir <out>
EOF
