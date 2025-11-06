#!/bin/bash
#SBATCH --job-name=neuro_harmonizing
#SBATCH --mail-user=lisa.duttenhoefer@stud.uni-heidelberg.de
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/logs/slurm_test_%j.out
#SBATCH --error=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/logs/slurm_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source /net/data.isilon/ag-cherrmann/lduttenhoefer/project/miniconda3/etc/profile.d/conda.sh
conda activate LISA_ba_env

cd /net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model

echo "=== Testing Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

python -u combat_neuro/neuro_harmonizing.py 
    

echo ""
echo "=== Harmonizing finished at: $(date) ==="