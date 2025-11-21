#!/bin/bash
#SBATCH --job-name=conditinoal_vae_training
#SBATCH --mail-user=lisa.duttenhoefer@stud.uni-heidelberg.de
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/logs/slurm_vae_%j.out
#SBATCH --error=/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/logs/slurm_vae_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Aktiviere Conda Environment
source /net/data.isilon/ag-cherrmann/lduttenhoefer/project/miniconda3/etc/profile.d/conda.sh
conda activate LISA_ba_env

cd /net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model

# Kurze Info ins slurm log
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $SLURM_NODELIST"
echo "Full output -> logs/training_output_${SLURM_JOB_ID}.log"

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# Alles nur ins training_output log
python -u src/RUN_training_CondVAE.py \
    --atlas_name aparc_dk40 neuromorphometrics \
    --volume_type G T Vgm \
    --exclude_datasets NU \
    --num_epochs 300 \
    --n_bootstraps 50 \
    --beta 1.0 \
    --latent_dim 40 \
    --contr_loss_weight 0.0 \
    --kl_warmup_epochs 100 \
    --kldiv_weight 1.0 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --normalization_method columnwise  &> logs/training_output_${SLURM_JOB_ID}.log

echo "Job finished at $(date)"

