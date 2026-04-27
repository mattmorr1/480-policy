#!/bin/bash -l
#$ -P ds549
#$ -N policy_train_shared
#$ -l h_rt=03:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=9.0
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

set -euo pipefail

if [[ -n "${SCRATCH_HOME:-}" ]]; then
  SCRATCH_BASE="$SCRATCH_HOME"
elif [[ -d "/scratch/$USER" ]]; then
  SCRATCH_BASE="/scratch/$USER"
elif [[ -n "${SGE_O_WORKDIR:-}" && "$SGE_O_WORKDIR" == /scratch/* ]]; then
  SCRATCH_BASE="$(dirname "$SGE_O_WORKDIR")"
elif [[ -d "/scratch" ]]; then
  SCRATCH_BASE="/scratch"
else
  echo "Could not determine scratch base. Set SCRATCH_HOME explicitly."
  exit 1
fi

REPO="${REPO:-${SGE_O_WORKDIR:-$PWD}}"
VENV="${VENV:-$SCRATCH_BASE/venvs/policy-310}"
if [[ "$VENV" != /scratch/* ]]; then
  echo "VENV must live on /scratch to avoid home quota limits: $VENV"
  exit 1
fi

if [[ ! -d "$REPO/src" ]]; then
  echo "Missing repo at $REPO"
  echo "Set REPO explicitly or submit from the repo directory."
  exit 1
fi

cd "$REPO"
mkdir -p logs

SCRATCH_PROJECT="${SCRATCH_PROJECT:-$SCRATCH_BASE/policy-480}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRATCH_PROJECT/artifacts}"
mkdir -p "$SCRATCH_PROJECT" "$ARTIFACT_ROOT"

ensure_scratch_link() {
  local repo_rel="$1"
  local scratch_dir="$2"
  local repo_path="$REPO/$repo_rel"
  mkdir -p "$scratch_dir"
  if [[ -L "$repo_path" ]]; then
    ln -sfn "$scratch_dir" "$repo_path"
  elif [[ -e "$repo_path" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --ignore-existing "$repo_path"/ "$scratch_dir"/
    else
      cp -a "$repo_path"/. "$scratch_dir"/ 2>/dev/null || true
    fi
    rm -rf "$repo_path"
    ln -s "$scratch_dir" "$repo_path"
  else
    ln -s "$scratch_dir" "$repo_path"
  fi
}

ensure_scratch_link "shared_adapter" "$ARTIFACT_ROOT/shared_adapter"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
  fi
fi
if ! command -v module >/dev/null 2>&1; then
  echo "module command unavailable in job shell"
  exit 1
fi
module load python3/3.10.12
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  source "$VENV/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export PIP_CACHE_DIR="$SCRATCH_PROJECT/.cache/pip"
export HF_HOME="$SCRATCH_PROJECT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="${TMPDIR:-$SCRATCH_PROJECT/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
FAST_MODE="${FAST_MODE:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-$([[ "$FAST_MODE" == "1" ]] && echo 1 || echo 3)}"

python src/data_prep.py
python src/train_lora.py --mode shared --num_train_epochs "$TRAIN_EPOCHS" --output_dir shared_adapter/
