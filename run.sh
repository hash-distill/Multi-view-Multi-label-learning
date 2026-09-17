#!/usr/bin/env bash
# Run main.py inside the conda environment that holds the dependencies.
#
# Why conda and not a repo-local .venv?
#   This repository lives on /mnt/Data4, which is mounted `noexec`. Shared
#   objects (.so) inside a virtualenv on such a mount cannot be mmap'ed, so
#   `import numpy` fails with:
#       "failed to map segment from shared object"
#   Conda environments live under the conda envs directory (here
#   ~/miniconda3/envs), on the executable home filesystem, so this problem
#   does not arise.
#
# Override the environment with:  CONDA_ENV=myenv bash run.sh …
# Override the base with:         CONDA_BASE=/opt/conda bash run.sh …
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-mvml}"

CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "$CONDA_BASE" ]]; then
  for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$candidate"; break; fi
  done
fi
if [[ -z "$CONDA_BASE" ]]; then
  echo "conda not found; set CONDA_BASE=/path/to/miniconda3" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Resolve the real prefix with conda: environments may live in <base>/envs or
# in ~/.conda/envs depending on how envs_dirs is configured.
PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF; exit}')"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
cd "$HERE"

if [[ -n "$PREFIX" && -x "$PREFIX/bin/python" ]]; then
  # Calling the interpreter directly works even when `conda activate` is
  # unavailable in a non-interactive shell, and keeps stdout unbuffered.
  exec "$PREFIX/bin/python" main.py "$@"
fi

if [[ -n "$PREFIX" ]]; then
  exec conda run --no-capture-output -n "$ENV_NAME" python main.py "$@"
fi

echo "conda environment '$ENV_NAME' not found. Run 'bash setup_env.sh' first" >&2
exit 1
