#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p coe-gpu
#SBATCH -t 8:00:00              # 2 GPUs x 16h trips QOSMaxGRESMinutesPerJob; chain with sbatch server.sh <count>
#SBATCH --exclude=atl1-1-03-010-10-0,atl1-1-03-011-13-0,atl1-1-03-011-18-0
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}

hostname

module load cuda
module load uv

# Respect the GPU visibility selected by Slurm. Hard-coding 0,1 can make
# PyTorch probe devices outside the allocation on some cluster GPU nodes.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
export USE_TF=0
export USE_TORCH=1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

# Write hostname to file so tests can find the server
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"

# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER:-<not set>}"

# Default behavior: START island controller unless explicitly disabled
if [ "${SUBMIT_ISLAND_CONTROLLER:-1}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    # refuse to start if the seed would split wrong (a stray "# --OPTION--" made main() evolvable in run 6590841)
    uv run python sota/QuantumVQC/test_seed_blocks.py "${LLMGE_SEED_NETWORK:-sota/QuantumVQC/network.py}" || { scancel "$SLURM_JOB_ID"; exit 1; }
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER})"
fi

uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port 8137 --workers 1
