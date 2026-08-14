#!/bin/bash -l
#SBATCH --job-name=moss-ml-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
unset SLURM_EXPORT_ENV

JOB_ENV=${1:?"usage: sbatch train_malayalam_alex.sbatch /absolute/path/alex.env"}
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
mkdir -p "${MOSS_RUN_DIR}"
if [[ ! -f "${MOSS_RUN_DIR}/run_config.env" ]]; then
    {
        printf 'created_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_commit=%q\n' "${MOSS_GIT_COMMIT:-unknown}"
        printf 'model_path=%q\n' "${MOSS_EXTENDED_MODEL}"
        printf 'codec_path=%q\n' "${MOSS_CODEC_MODEL}"
        printf 'train_manifest_pattern=%q\n' "${MOSS_PREPARED_DIR}/train.rank?????-of-*.jsonl"
        printf 'validation_manifest=%q\n' "${MOSS_PREPARED_DIR}/val.rank00000-of-00001.jsonl"
        printf 'max_length=%q\n' "${MOSS_MAX_LENGTH:-1024}"
        printf 'per_device_batch_size=%q\n' 4
        printf 'gradient_accumulation_steps=%q\n' 4
        printf 'learning_rate=%q\n' 1e-5
        printf 'num_epochs=%q\n' 3
    } > "${MOSS_RUN_DIR}/run_config.env"
fi
TRAIN_SPEC="${MOSS_PREPARED_DIR}/train.rank?????-of-$(printf '%05d' "${MOSS_NUM_TRAIN_SHARDS:-32}").jsonl"
VAL_SPEC="${MOSS_PREPARED_DIR}/val.rank00000-of-00001.jsonl"
if ! compgen -G "${TRAIN_SPEC}" >/dev/null; then
    echo "No prepared training shards match: ${TRAIN_SPEC}" >&2
    exit 2
fi
if [[ ! -s "${VAL_SPEC}" ]]; then
    echo "Missing prepared validation manifest: ${VAL_SPEC}" >&2
    exit 2
fi

echo "job_id=${SLURM_JOB_ID} host=$(hostname) train_spec=${TRAIN_SPEC}"
nvidia-smi

ACTIVE_CHILD_PID=""
requeue_before_timeout() {
    echo "Time-limit warning received; stopping the current partial epoch." >&2
    if [[ -n "${ACTIVE_CHILD_PID}" ]]; then
        kill -TERM "${ACTIVE_CHILD_PID}" 2>/dev/null || true
        wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
    fi
    if scontrol requeue "${SLURM_JOB_ID}"; then
        echo "Requeued job ${SLURM_JOB_ID}; it will resume from the newest completed epoch." >&2
        exit 0
    fi
    echo "Automatic requeue failed; resubmit the training job to resume." >&2
    exit 143
}
trap requeue_before_timeout USR1

python -u finetuning/sft.py \
    --model-path "${MOSS_EXTENDED_MODEL}" \
    --codec-path "${MOSS_CODEC_MODEL}" \
    --train-jsonl "${TRAIN_SPEC}" \
    --eval-jsonl "${VAL_SPEC}" \
    --output-dir "${MOSS_RUN_DIR}" \
    --max-length "${MOSS_MAX_LENGTH:-1024}" \
    --per-device-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.03 \
    --num-epochs 3 \
    --max-grad-norm 1.0 \
    --logging-steps 25 \
    --save-every-epochs 1 \
    --num-workers 4 \
    --mixed-precision bf16 \
    --attn-implementation sdpa \
    --channelwise-loss-weight 1,32 \
    --resume-from-checkpoint auto &
ACTIVE_CHILD_PID=$!

set +e
wait "${ACTIVE_CHILD_PID}"
training_status=$?
set -e
ACTIVE_CHILD_PID=""
if [[ ${training_status} -ne 0 ]]; then
    echo "Training process failed with status ${training_status}." >&2
    exit "${training_status}"
fi

echo "Training complete: ${MOSS_RUN_DIR}/checkpoint-last"
