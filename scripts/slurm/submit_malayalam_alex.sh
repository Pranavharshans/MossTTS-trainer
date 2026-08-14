#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=${1:-$(pwd)}
RUNTIME_ROOT=${2:-/anvme/workspace/v123be62-moss-tts}
DATA_DIR=${3:-/anvme/workspace/v123be62-voxcpm-ml/voxcpm-runtime/datasets/rasa-malayalam}

REPO_DIR=$(realpath "${REPO_DIR}")
mkdir -p "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/logs"
RUNTIME_ROOT=$(realpath "${RUNTIME_ROOT}")
DATA_DIR=$(realpath "${DATA_DIR}")
JOB_ENV=${RUNTIME_ROOT}/alex.env
SLURM_DIR=${REPO_DIR}/scripts/slurm
LOG_DIR=${RUNTIME_ROOT}/logs

for command_name in sbatch squeue sacct; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "Required Slurm command is unavailable: ${command_name}" >&2
        exit 2
    }
done

setup_job=$(sbatch --parsable \
    --output="${LOG_DIR}/moss-ml-setup-%j.out" \
    "${SLURM_DIR}/setup_malayalam_alex.sbatch" \
    "${REPO_DIR}" "${RUNTIME_ROOT}" "${DATA_DIR}")
setup_job=${setup_job%%;*}
prepare_job=$(sbatch --parsable \
    --dependency="afterok:${setup_job}" \
    --output="${LOG_DIR}/moss-ml-prep-%j.out" \
    "${SLURM_DIR}/prepare_malayalam_alex.sbatch" "${JOB_ENV}")
prepare_job=${prepare_job%%;*}
train_job=$(sbatch --parsable \
    --dependency="afterok:${prepare_job}" \
    --output="${LOG_DIR}/moss-ml-train-%j.out" \
    "${SLURM_DIR}/train_malayalam_alex.sbatch" "${JOB_ENV}")
train_job=${train_job%%;*}
eval_job=$(sbatch --parsable \
    --dependency="afterok:${train_job}" \
    --output="${LOG_DIR}/moss-ml-eval-%j.out" \
    "${SLURM_DIR}/evaluate_malayalam_alex.sbatch" "${JOB_ENV}")
eval_job=${eval_job%%;*}

jobs_csv=${setup_job},${prepare_job},${train_job},${eval_job}
echo "Submitted setup=${setup_job} prepare=${prepare_job} train=${train_job} eval=${eval_job}"
echo "Queue:   squeue -j ${jobs_csv}"
echo "History: sacct -j ${jobs_csv} --format=JobID,JobName,State,Elapsed,ExitCode"
echo "Setup:   tail -f ${LOG_DIR}/moss-ml-setup-${setup_job}.out"
echo "Prepare: tail -f ${LOG_DIR}/moss-ml-prep-${prepare_job}.out"
echo "Train:   tail -f ${LOG_DIR}/moss-ml-train-${train_job}.out"
echo "Eval:    tail -f ${LOG_DIR}/moss-ml-eval-${eval_job}.out"
echo "Runtime: ${RUNTIME_ROOT}"
