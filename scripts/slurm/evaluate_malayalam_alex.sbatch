#!/bin/bash -l
#SBATCH --job-name=moss-ml-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
unset SLURM_EXPORT_ENV

JOB_ENV=${1:?"usage: sbatch evaluate_malayalam_alex.sbatch /absolute/path/alex.env"}
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

cd "${MOSS_REPO_DIR}"
CHECKPOINTS=("${MOSS_EXTENDED_MODEL}" "${MOSS_RUN_DIR}"/checkpoint-epoch-*)
if [[ ${#CHECKPOINTS[@]} -lt 2 || ! -d "${CHECKPOINTS[1]}" ]]; then
    echo "No epoch checkpoints found under ${MOSS_RUN_DIR}" >&2
    exit 2
fi
nvidia-smi

python finetuning/evaluate_checkpoints.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --prompts-jsonl finetuning/configs/malayalam_eval_prompts.jsonl \
    --codec-path "${MOSS_CODEC_MODEL}" \
    --prompt-audio-path "${MOSS_PROMPT_AUDIO}" \
    --output-dir "${MOSS_EVAL_DIR}" \
    --device cuda \
    --dtype bfloat16 \
    --max-new-frames 500 \
    --do-sample 1 \
    --seed 42

echo "Evaluation complete: ${MOSS_EVAL_DIR}/evaluation_summary.json"
