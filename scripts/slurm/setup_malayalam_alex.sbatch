#!/bin/bash -l
#SBATCH --job-name=moss-ml-setup
#SBATCH --partition=a100mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
unset SLURM_EXPORT_ENV

REPO_DIR=${1:?"usage: sbatch setup_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT DATA_DIR"}
RUNTIME_ROOT=${2:?"usage: sbatch setup_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT DATA_DIR"}
DATA_DIR=${3:?"usage: sbatch setup_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT DATA_DIR"}

REPO_DIR=$(realpath "${REPO_DIR}")
DATA_DIR=$(realpath "${DATA_DIR}")
mkdir -p "${RUNTIME_ROOT}"
RUNTIME_ROOT=$(realpath "${RUNTIME_ROOT}")

TRAIN_JSONL=${DATA_DIR}/train.jsonl
VAL_JSONL=${DATA_DIR}/val.jsonl
PROMPT_AUDIO=${DATA_DIR}/eval_prompt.flac
PROMPT_TEXT=${DATA_DIR}/eval_prompt.txt
for required_path in "${TRAIN_JSONL}" "${VAL_JSONL}" "${PROMPT_AUDIO}" "${PROMPT_TEXT}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Missing required dataset artifact: ${required_path}" >&2
        exit 2
    fi
done

CONDA_ENV=${RUNTIME_ROOT}/envs/moss-tts-nano
MODEL_ROOT=${RUNTIME_ROOT}/models
BASE_MODEL=${MODEL_ROOT}/MOSS-TTS-Nano
CODEC_MODEL=${MODEL_ROOT}/MOSS-Audio-Tokenizer-Nano
EXTENDED_MODEL=${MODEL_ROOT}/MOSS-TTS-Nano-Malayalam
TOKENIZER_DIR=${RUNTIME_ROOT}/tokenizer
PREPARED_DIR=${RUNTIME_ROOT}/prepared
RUN_DIR=${RUNTIME_ROOT}/runs/malayalam-sft
EVAL_DIR=${RUN_DIR}/evaluation
JOB_ENV=${RUNTIME_ROOT}/alex.env
GIT_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || printf unknown)

MODEL_REPO=OpenMOSS-Team/MOSS-TTS-Nano
MODEL_REVISION=44502f80dbf9743528fa921cc544d662c685ebec
CODEC_REPO=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
CODEC_REVISION=6aa02b01e445cc585582cf0ba480bc3ea6c8dd68

module load python
ALEX_PROXY_URL=${ALEX_PROXY_URL:-http://proxy.nhr.fau.de:80}
export http_proxy=${http_proxy:-${ALEX_PROXY_URL}}
export https_proxy=${https_proxy:-${ALEX_PROXY_URL}}
export HTTP_PROXY=${HTTP_PROXY:-${ALEX_PROXY_URL}}
export HTTPS_PROXY=${HTTPS_PROXY:-${ALEX_PROXY_URL}}
export CONDA_PKGS_DIRS=${RUNTIME_ROOT}/cache/conda-pkgs
export PIP_CACHE_DIR=${RUNTIME_ROOT}/cache/pip
export HF_HOME=${RUNTIME_ROOT}/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${MODEL_ROOT}" \
    "${TOKENIZER_DIR}" "${PREPARED_DIR}" "${RUN_DIR}" "${EVAL_DIR}"

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    mkdir -p "$(dirname "${CONDA_ENV}")"
    conda create --prefix "${CONDA_ENV}" python=3.11 -y
fi
conda activate "${CONDA_ENV}"

python -m pip install --upgrade pip wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.7.0' 'torchaudio==2.7.0'
python -m pip install -r "${REPO_DIR}/requirements-finetune.txt"

python "${REPO_DIR}/finetuning/download_snapshot.py" \
    --repo-id "${MODEL_REPO}" \
    --revision "${MODEL_REVISION}" \
    --output-dir "${BASE_MODEL}"
python "${REPO_DIR}/finetuning/download_snapshot.py" \
    --repo-id "${CODEC_REPO}" \
    --revision "${CODEC_REVISION}" \
    --output-dir "${CODEC_MODEL}"

python "${REPO_DIR}/finetuning/build_malayalam_tokenizer.py" \
    --base-tokenizer-model "${BASE_MODEL}/tokenizer.model" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-model "${TOKENIZER_DIR}/tokenizer.model" \
    --report-path "${TOKENIZER_DIR}/report.json" \
    --num-new-tokens 2048 \
    --candidate-vocab-size 4096

python "${REPO_DIR}/finetuning/initialize_extended_model.py" \
    --base-model-path "${BASE_MODEL}" \
    --extended-tokenizer-model "${TOKENIZER_DIR}/tokenizer.model" \
    --tokenizer-report "${TOKENIZER_DIR}/report.json" \
    --output-dir "${EXTENDED_MODEL}"

{
    printf 'MOSS_REPO_DIR=%q\n' "${REPO_DIR}"
    printf 'MOSS_GIT_COMMIT=%q\n' "${GIT_COMMIT}"
    printf 'MOSS_CONDA_ENV=%q\n' "${CONDA_ENV}"
    printf 'MOSS_RUNTIME_ROOT=%q\n' "${RUNTIME_ROOT}"
    printf 'MOSS_DATA_DIR=%q\n' "${DATA_DIR}"
    printf 'MOSS_TRAIN_JSONL=%q\n' "${TRAIN_JSONL}"
    printf 'MOSS_VAL_JSONL=%q\n' "${VAL_JSONL}"
    printf 'MOSS_PROMPT_AUDIO=%q\n' "${PROMPT_AUDIO}"
    printf 'MOSS_PROMPT_TEXT=%q\n' "${PROMPT_TEXT}"
    printf 'MOSS_BASE_MODEL=%q\n' "${BASE_MODEL}"
    printf 'MOSS_CODEC_MODEL=%q\n' "${CODEC_MODEL}"
    printf 'MOSS_EXTENDED_MODEL=%q\n' "${EXTENDED_MODEL}"
    printf 'MOSS_TOKENIZER_REPORT=%q\n' "${TOKENIZER_DIR}/report.json"
    printf 'MOSS_PREPARED_DIR=%q\n' "${PREPARED_DIR}"
    printf 'MOSS_RUN_DIR=%q\n' "${RUN_DIR}"
    printf 'MOSS_EVAL_DIR=%q\n' "${EVAL_DIR}"
    printf 'MOSS_MAX_LENGTH=%q\n' 1024
    printf 'MOSS_NUM_TRAIN_SHARDS=%q\n' 32
    printf 'HF_HOME=%q\n' "${HF_HOME}"
} > "${JOB_ENV}"

echo "Setup complete: train_rows=$(wc -l < "${TRAIN_JSONL}") val_rows=$(wc -l < "${VAL_JSONL}")"
echo "Tokenizer report: ${TOKENIZER_DIR}/report.json"
echo "Job environment: ${JOB_ENV}"
