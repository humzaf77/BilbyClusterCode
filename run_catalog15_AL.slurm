#!/bin/bash
#SBATCH --job-name=gwcatalog
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge

# conda + lalsuite activation hooks are not nounset-safe; keep -u off around them
set +u
source /resnick/groups/RelativityTheory/alaeuger/miniconda3/etc/profile.d/conda.sh
conda activate BilbyEnv

# Make sure the env's own libs win over /lib64 fallbacks
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Python: $(which python)"
python --version


EVENT=$1
python run_catalog15.py \
    "$EVENT" \
    --npool "$SLURM_CPUS_PER_TASK"
