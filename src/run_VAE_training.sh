#!/bin/bash
#SBATCH --job-name=vae_training
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
python -u src/run_ConVAE_2D_train_adapt.py \
    --atlas_name all \
    --num_epochs 200 \
    --n_bootstraps 100 \
    --volume_type all \
    --batch_size 16 &> logs/training_output_${SLURM_JOB_ID}.log

echo "Job finished at $(date)"
