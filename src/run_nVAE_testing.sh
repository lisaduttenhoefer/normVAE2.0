#!/bin/bash
#SBATCH --job-name=vae_v2_testing
#SBATCH --mail-user=lisa.duttenhoefer@stud.uni-heidelberg.de
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/logs/slurm_test_%j.out
#SBATCH --error=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/logs/slurm_test_%j.err
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

# IMPORTANT: Update this path to your actual training results directory
# Example: /net/data.isilon/.../norm_results_HC_Vgm_Vwm_Vcsf_G_T_all_20251022_1625
MODEL_DIR="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/nVAE/TRAINING/norm_results_HC_Vgm_T_G_neuromorpometrics_aparc_dk40_columnwise_20251121_1545"
echo "=== Model Directory ==="
echo "$MODEL_DIR"
echo ""

export PYTHONUNBUFFERED=1

python -u src/RUN_testing_normVAE2.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/nVAE/TESTING" \
    --seed 42 &> logs/testing_output_${SLURM_JOB_ID}.log
     

echo ""
echo "=== Testing finished at: $(date) ==="