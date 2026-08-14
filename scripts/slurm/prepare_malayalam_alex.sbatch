#!/bin/bash -l
#SBATCH --job-name=moss-ml-prep
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
unset SLURM_EXPORT_ENV

JOB_ENV=${1:?"usage: sbatch prepare_malayalam_alex.sbatch /absolute/path/alex.env"}
if [[ "${JOB_ENV}" != /* ]]; then
    JOB_ENV=${SLURM_SUBMIT_DIR}/${JOB_ENV}
fi
if [[ ! -f "${JOB_ENV}" ]]; then
    echo "Missing job environment: ${JOB_ENV}" >&2
    exit 2
fi
set -a
source "${JOB_ENV}"
set +a

module load python
conda activate "${MOSS_CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "${MOSS_REPO_DIR}"
echo "job_id=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi

python finetuning/prepare_sharded_dataset.py \
    --codec-path "${MOSS_CODEC_MODEL}" \
    --model-path "${MOSS_EXTENDED_MODEL}" \
    --input-jsonl "${MOSS_TRAIN_JSONL}" \
    --output-dir "${MOSS_PREPARED_DIR}" \
    --prefix train \
    --num-shards "${MOSS_NUM_TRAIN_SHARDS:-32}" \
    --batch-size 8 \
    --max-length "${MOSS_MAX_LENGTH:-1024}" \
    --device cuda

python finetuning/prepare_sharded_dataset.py \
    --codec-path "${MOSS_CODEC_MODEL}" \
    --model-path "${MOSS_EXTENDED_MODEL}" \
    --input-jsonl "${MOSS_VAL_JSONL}" \
    --output-dir "${MOSS_PREPARED_DIR}" \
    --prefix val \
    --num-shards 1 \
    --batch-size 8 \
    --max-length "${MOSS_MAX_LENGTH:-1024}" \
    --device cuda

echo "Prepared manifests: ${MOSS_PREPARED_DIR}"
