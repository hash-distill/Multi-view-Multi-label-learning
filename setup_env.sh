#!/usr/bin/env bash
# Create the conda environment this project needs.
#
#   bash setup_env.sh                        # creates conda env "mvml"
#   CONDA_ENV=myenv bash setup_env.sh        # custom env name
#   PIP_INDEX_URL=... bash setup_env.sh
#
# Design notes:
#   * The repository lives on /mnt/Data4, which is mounted `noexec`: shared
#     objects inside a virtualenv placed there cannot be mmap'ed, so
#     `import numpy` fails with "failed to map segment from shared object".
#     Conda environments are created under ~/.conda/envs (home filesystem),
#     which is executable — this is why we use conda rather than a repo-local
#     .venv. For the same reason a script on that mount cannot be executed
#     directly: invoke it as `bash run.sh …`, not `./run.sh …`.
#   * A stale numpy/scipy in ~/.local can shadow the environment. We therefore
#     call the environment's interpreter directly and set PYTHONNOUSERSITE=1,
#     instead of relying on `conda activate` (which is also unavailable in
#     non-interactive shells unless conda.sh has been sourced).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-mvml}"
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# Locate conda
CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "$CONDA_BASE" ]]; then
  for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    [[ -f "$candidate/etc/profile.d/conda.sh" ]] && CONDA_BASE="$candidate" && break
  done
fi
if [[ -z "$CONDA_BASE" ]]; then
  echo "conda not found; set CONDA_BASE=/path/to/miniconda3" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo ">> project    : $HERE"
echo ">> conda base : $CONDA_BASE"
echo ">> env name   : $ENV_NAME"
echo ">> pip index  : $INDEX"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# The environment must exist on an executable filesystem; conda handles that.
conda create -y -n "$ENV_NAME" python=3.10 pip

# Resolve the real prefix: conda may place environments in <base>/envs or in
# ~/.conda/envs, depending on how envs_dirs is configured.
PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF; exit}')"
if [[ -z "$PREFIX" ]]; then
  echo "failed to create conda environment '$ENV_NAME'" >&2
  exit 1
fi
PY="$PREFIX/bin/python"
echo ">> prefix     : $PREFIX"
echo ">> interpreter: $PY"

# Versions pinned to a self-consistent set. scikit-multilearn 0.2.0 (2018) and
# skfeature-chappers predate numpy 2.x and modern scikit-learn, so we pin down.
#
# --ignore-installed matters here. This machine has a numpy/scipy/scikit-learn
# tree in ~/.local. pip inspects *those* when deciding whether a requirement is
# already satisfied and then skips it, leaving only a .dist-info in the env and
# no importable package. Since run.sh sets PYTHONNOUSERSITE=1 (to stop the
# ~/.local tree from shadowing the env), such a package would be missing at
# runtime. Forcing the install puts every file inside the environment.
PIP_COMMON=(--ignore-installed --no-build-isolation -i "$INDEX")

"$PY" -m pip install "${PIP_COMMON[@]}" --upgrade "setuptools<70" wheel Cython
"$PY" -m pip install "${PIP_COMMON[@]}" \
    "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.3.2" \
    "pandas==2.0.3" "openpyxl==3.1.2" \
    "scikit-multilearn==0.2.0" "skfeature-chappers==1.2.1"

# Verify with PYTHONNOUSERSITE=1, exactly as run.sh will execute it: this fails
# if any dependency was skipped because of the ~/.local tree.
PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import sys
import numpy, scipy, sklearn, openpyxl, pandas
from skmultilearn.adapt import MLkNN          # noqa: F401
from skfeature.utility.construct_W import construct_W  # noqa: F401

# Prove the packages really come from the environment, not from ~/.local
env_prefix = sys.prefix
for mod in (numpy, scipy, sklearn, pandas):
    assert mod.__file__.startswith(env_prefix), f"{mod.__name__} loaded from {mod.__file__}"
print("python", sys.version.split()[0], "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)
print("skmultilearn / skfeature / openpyxl / pandas OK — all inside", env_prefix)
PY

# run.sh resolves the environment at run time via `conda env list` and honours
# CONDA_ENV, so there is nothing to rewrite here. Just make it executable.
if [[ -f "$HERE/run.sh" ]]; then
  chmod +x "$HERE/run.sh"
fi

echo
echo ">> done. run experiments with:"
echo "   bash $HERE/run.sh --list"
echo "   bash $HERE/run.sh --check"
echo "   bash $HERE/run.sh --alg DHLI --data emotions"
