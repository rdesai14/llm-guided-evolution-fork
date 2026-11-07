#!/bin/bash
#SBATCH --job-name=Island_controller
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p pace-cpu
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=8:30:00   # outlive the 8 h island job so the chain below can run
#SBATCH --output=run_job_outputs/islands/Report_island_controller-%j.out
# Island controller arguments:
# $1 (COUNT): Number of remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): Job ID of the currently-running server (to cancel)
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda

# LLM_Storage
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the repository root (server.sh submits this job from it)
cd "$SLURM_SUBMIT_DIR"

uv run python islands_wrapper.py quantum_run \
    --num_islands 1 \
    --llms llama3 \
    --prompt_groups "FixedPrompts/[cr]o*/*.txt"

# Stop once the last generation is checkpointed: cancel the (now idle) server instead
# of letting it hold 2 GPUs to its time limit, and do not chain another block.
LAST_GEN=$(( $(grep -oP "^num_generations = \K[0-9]+" src/cfg/constants.py) - 1 ))
if [[ -f quantum_run/global_data/global_gen_${LAST_GEN}.pkl ]]; then
    echo "Run finished (generation ${LAST_GEN} checkpointed); stopping server $PREV_SERVER_JOB_ID"
    [[ -n "$PREV_SERVER_JOB_ID" ]] && scancel "$PREV_SERVER_JOB_ID"
    # A run queued behind this one (runs/next_run.sh) starts here, once.
    if [[ -f runs/next_run.sh ]]; then
        mv runs/next_run.sh runs/next_run.started.sh
        echo "Starting the queued run: runs/next_run.started.sh"
        bash runs/next_run.started.sh || echo "Queued run did not start (exit $?)"
    fi
    exit 0
fi

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    # Extra blocks requested mid-run (runs/extra_blocks): each one is spent by keeping the
    # count instead of lowering it, so the chain runs that many more 8 h blocks.
    if [[ -f runs/extra_blocks ]] && (( $(cat runs/extra_blocks) > 0 )); then
        echo $(( $(cat runs/extra_blocks) - 1 )) > runs/extra_blocks
        NEXT_COUNT=$COUNT
        echo "Using one extra block ($(cat runs/extra_blocks) left)"
    fi
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi
