#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 16:00:00              		# Runtime in D-HH:MM
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p pace-cpu
#SBATCH --mem 16G
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
# Account/QOS/partition are mandatory on ICE; without them sbatch rejects the job
# with "Invalid qos specification". coe-gpu is the GPU partition coe-ice may use.
# The constraint is pinned to H100 because coe-gpu also has H200 nodes on a
# different CPU generation, and that changes floating-point results.

echo "launching LLM Guided Evolution"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda
# CUDA_VISIBLE_DEVICES is deliberately NOT set here. sbatch defaults to
# --export=ALL, so anything exported by this orchestrator leaks into every LLM
# job it submits. Pinning device 0 here made each LLM job ignore the 2 GPUs
# SLURM gave it and fight over physical device 0 instead:
#   RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
# The orchestrator only submits and polls; it runs no CUDA work itself.

export SCRATCH_DIR="/storage/ice1/9/7/rdesai317/scratch.cache"
export UV_CACHE_DIR="$SCRATCH_DIR/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/uv_python"
source /home/hice1/rdesai317/scratch/Quantam/llm-guided-evolution-fork/.venv/bin/activate

# Children are launched with `sbatch`, which defaults to --export=ALL. Make sure
# no stale GPU pinning rides along; SLURM sets this per child job itself.
unset CUDA_VISIBLE_DEVICES

python run_improved.py first_test
