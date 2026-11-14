#!/bin/bash
# Experiment 1 (Nov 2026): baseline vs everything on, 5 trials each, 35 generations, locked seed.
# The arms alternate (queue.txt) so PACE load and any drift hit both equally. Each person lists
# their own trials in runs/exp1/queue.txt, one per line (e.g. baseline_3 then all_3).
# Start the first trial by hand from the repository:  bash runs/exp1/next_trial.sh
# After that the island controller runs a copy (runs/next_run.sh) when a trial finishes: this
# archives that trial to runs/exp1/<trial>/, writes its run_stats JSON, starts the next trial
# and re-arms itself. log.txt records every start and finish.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"   # this clone, whoever runs it
EXP=runs/exp1

# the finished trial's jobs must be gone, apart from the controller running this script
for try in $(seq 20); do
    left=$(squeue -u "$USER" -h -o "%i %j" | grep -E "LLMGE01_Server|Island_controller|Islands_script|llm_oper|evaluateGene" | grep -v "^${SLURM_JOB_ID:-none} " || true)
    [[ -z "$left" ]] && break
    (( try == 20 )) && { echo "jobs still queued:"; echo "$left"; exit 1; }
    sleep 30
done

if [[ -f $EXP/current ]]; then
    T=$(cat $EXP/current)
    mkdir -p $EXP/$T/sota_QuantumVQC
    mv quantum_run run_job_outputs hostname.log $EXP/$T/ 2>/dev/null || true
    mv sota/QuantumVQC/models sota/QuantumVQC/renders sota/QuantumVQC/results $EXP/$T/sota_QuantumVQC/ 2>/dev/null || true
    .venv/bin/python $EXP/run_stats.py $EXP/$T/quantum_run $EXP/$T/sota_QuantumVQC/models \
        $EXP/$T/sota_QuantumVQC/results $EXP/$T/stats.json || echo "run_stats failed for $T"
    echo "$T finished $(date)" >> $EXP/log.txt
    rm $EXP/current
fi
mkdir -p sota/QuantumVQC/models sota/QuantumVQC/results run_job_outputs/{evaluation,evolution,islands,server}

T=$(head -1 $EXP/queue.txt || true)
if [[ -z "$T" ]]; then
    echo "experiment finished $(date)" >> $EXP/log.txt
    exit 0
fi
sed -i 1d $EXP/queue.txt
# every switch is set either way: a trial started by the previous trial's controller inherits its environment.
# The server port is per person, so two people's servers on one GPU node don't collide.
PORT=$((8200 + $(id -u) % 700))
case $T in
    baseline_*) SW="LLMGE_CIRCUIT_VIEW=off,LLMGE_QISKIT_RAG=0,LLMGE_QD_MODE=0,LLMGE_QD_DEDUP=0,LLMGE_PORT=$PORT" ;;
    all_*)      SW="LLMGE_CIRCUIT_VIEW=ascii,LLMGE_QISKIT_RAG=1,LLMGE_QD_MODE=1,LLMGE_QD_DEDUP=1,LLMGE_PORT=$PORT" ;;
    *)          echo "unknown trial $T"; exit 1 ;;
esac
echo "$T" > $EXP/current
# proof that every trial ran the same code and settings
mkdir -p $EXP/$T
git rev-parse HEAD > $EXP/$T/commit.txt
cp src/cfg/constants.py $EXP/$T/constants.py
echo 0 > runs/extra_blocks
cp $EXP/next_trial.sh runs/next_run.sh
echo "$T started $(date) with $SW" >> $EXP/log.txt
sbatch --export=ALL,$SW server.sh 5   # ~26 h needed, so one spare 8 h block
