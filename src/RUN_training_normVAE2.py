import sys
sys.path.append("/home/developer/.local/lib/python3.10/site-packages")
sys.path.append("../src")
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from pathlib import Path
import torch
import torchio as tio
from torch.cuda.amp import GradScaler
import pandas as pd
import scanpy as sc
import seaborn as sns
import torchio as tio
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from diagnose_vae import run_full_diagnostics


matplotlib.use("Agg")


from models.ContrastVAE_2D import (
    NormativeVAE_2D, 
    train_normative_model_plots,
    bootstrap_train_normative_models_plots
)

from utils.support_f import (
    split_df_adapt,
    extract_measurements
)
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    load_checkpoint_model, 
    load_mri_data_2D_prenormalized,
    process_subjects, 
    train_val_split_annotations,
    train_val_split_subjects
)
    
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_model_ready,
    log_model_setup,
    setup_logging,
    log_atlas_mode
)

from utils.plotting_utils import (
    plot_learning_curves,
    plot_bootstrap_metrics,
)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling Training')
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.',  nargs='+', default=["all"])
    parser.add_argument('--volume_type', help='Volume type(s) to use', nargs='*', default=["Vgm", "Vwm", "Vcsf"])
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for', type=int, default=250)
    parser.add_argument('--n_bootstraps', help='Number of bootstrap samples', type=int, default=80)
    parser.add_argument('--kl_warmup_epochs', type=int, default=50, help='Number of epochs for KL warmup')
    parser.add_argument('--norm_diagnosis', help='which diagnosis is considered the "norm"', type=str, default="HC")
    parser.add_argument('--train_ratio', help='Normpslit ratio', type=float, default=0.7)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.000559)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=20) 
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=1.2)
    parser.add_argument('--save_models', help='Save all bootstrap models', action='store_true', default=True)
    parser.add_argument('--no_cuda', help='Disable CUDA (use CPU only)', action='store_true')
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, default=42)
    parser.add_argument('--output_dir', help='Override default output directory', default=None)
    # ========== NEW PARAMETER ==========
    parser.add_argument(
        '--normalization_method',
        type=str,
        default='rowwise',
        choices=['rowwise', 'columnwise'],
        help="Normalization method: 'rowwise' (Pinaya approach) or 'columnwise' (classical neuroimaging)"
    )
    return parser


def main(atlas_name: list, volume_type, num_epochs: int, n_bootstraps: int, norm_diagnosis: str, train_ratio: float, 
         batch_size: int, learning_rate: float, latent_dim: int, kldiv_weight: float, save_models: bool, kl_warmup_epochs: int,
         no_cuda: bool, seed: int, normalization_method: str = 'rowwise', output_dir: str = None):
    
    ## 0. Set Up ----------------------------------------------------------
    path_original = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata_no_bad_scans.csv"
    path_to_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    # ========== FIX: Save splits and get paths ==========
    train_df, test_df = split_df_adapt(
        path_original=path_original,
        path_to_dir=path_to_dir,
        norm_diagnosis=norm_diagnosis,
        train_ratio=train_ratio,
        random_seed=seed,
        save_splits=True  # ← Enable saving
    )
    
    # Get paths to saved CSVs
    TRAIN_CSV = f"{path_to_dir}/train_metadata_{norm_diagnosis}_{train_ratio}_seed{seed}.csv"
    TEST_CSV = f"{path_to_dir}/test_metadata_{norm_diagnosis}_{train_ratio}_seed{seed}.csv"
    
    joined_atlas_name = "_".join(str(a) for a in atlas_name if isinstance(a, str))
    joined_volume_name = "_".join(str(a) for a in volume_type if isinstance(a, str))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if output_dir is None:
        save_dir = f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/nVAE/TRAINING/norm_results_{norm_diagnosis}_{joined_volume_name}_{joined_atlas_name}_{normalization_method}_{timestamp}"
    else:
        save_dir = output_dir
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/latent_space", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/reconstructions", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)    

    config = Config_2D(
        RUN_NAME=f"NormativeVAE20_{joined_atlas_name}_{timestamp}_{norm_diagnosis}_{normalization_method}",
        TRAIN_CSV=[TRAIN_CSV],  # ← Now it's a path string
        TEST_CSV=[TEST_CSV],    # ← Now it's a path string
        MRI_DATA_PATH="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv",
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        VOLUME_TYPE=volume_type,
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "Vcsf", "G", "T"],
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        RECON_LOSS_WEIGHT=16.6449,
        KLDIV_LOSS_WEIGHT=kldiv_weight, 
        CONTR_LOSS_WEIGHT=0.0,
        TOTAL_EPOCHS=num_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=0.00356,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=4e-8,
        SCHEDULE_ON_VALIDATION=True,
        SCHEDULER_PATIENCE=6,
        SCHEDULER_FACTOR=0.5,
        CHECKPOINT_INTERVAL=5,
        DONT_PLOT_N_EPOCHS=0,
        UMAP_NEIGHBORS=30,
        UMAP_DOT_SIZE=20,
        METRICS_ROLLING_WINDOW=10,
        BATCH_SIZE=batch_size,
        DIAGNOSES=norm_diagnosis,  
        LATENT_DIM=latent_dim,
        SHUFFLE_DATA=True,
        SEED=seed
    )

    hidden_dim_1 = 100
    hidden_dim_2 = 100

    # Set up logging
    log_file = f"{save_dir}/logs/{timestamp}_normative_training.log"
    setup_logging(config)
    log_and_print(f"Starting normative modeling with atlas: {joined_atlas_name}, epochs: {num_epochs}, bootstraps: {n_bootstraps}")
    log_and_print(f"Normalization method: {normalization_method}")

    # Save configuration
    config_dict = vars(config)
    config_dict['NORMALIZATION_METHOD'] = normalization_method
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(f"{save_dir}/config.csv", index=False)
    log_and_print(f"Configuration saved to {save_dir}/config.csv")

    # Set seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    # Set device
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print(f"Using device: {device}")
    config.DEVICE = device

    ## 1. Load Data --------------------------------
    log_and_print("Loading NORM control data...")
    
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC_separate_TRAIN.csv"

    subjects_train, train_overview, roi_names_train = load_mri_data_2D_prenormalized(
        normalized_csv_path=NORMALIZED_CSV,
        csv_paths=[TRAIN_CSV],  # ← Now it's a STRING path! ✅
        diagnoses=["HC"],
        atlas_name=config.ATLAS_NAME,      
        volume_type=config.VOLUME_TYPE     
    )
    
    train_data_debug = extract_measurements(subjects_train)
    print(f"[DEBUG] Data shape: {train_data_debug.shape}")
    print(f"[DEBUG] Data min: {train_data_debug.min()}")
    print(f"[DEBUG] Data max: {train_data_debug.max()}")
    print(f"[DEBUG] Data mean: {train_data_debug.mean()}")
    print(f"[DEBUG] Data std: {train_data_debug.std()}")
    print(f"[DEBUG] Has NaN: {torch.isnan(train_data_debug).any()}")
    print(f"[DEBUG] Has Inf: {torch.isinf(train_data_debug).any()}")
    print(f"[DEBUG] Sample values: {train_data_debug[0, :10]}")

    len_atlas = len(roi_names_train)
    log_and_print(f"Number of ROIs in atlas: {len_atlas}")

    # Split norm controls metadata into train and validation
    train_annotations_norm, valid_annotations_norm = train_val_split_annotations(
        annotations=train_overview, 
        diagnoses=norm_diagnosis
    )
    
    # Split the feature maps accordingly
    train_subjects_norm, valid_subjects_norm = train_val_split_subjects(
        subjects=subjects_train, 
        train_ann=train_annotations_norm, 
        val_ann=valid_annotations_norm
    )

    train_annotations_norm.insert(1, "Data_Type", "train")
    valid_annotations_norm.insert(1, "Data_Type", "valid")

    annotations = pd.concat([train_annotations_norm, valid_annotations_norm])
    annotations.sort_values(by=["Data_Type", "Filename"], inplace=True)
    annotations.reset_index(drop=True, inplace=True)

    annotations = annotations.astype(
        {
            "Age": "float",
            "Dataset": "category",
            "Diagnosis": "category",
            "Sex": "category",
            "Data_Type": "category",
            "Filename": "category",
        }
    )

    log_and_print(annotations)

    # Prepare data loaders
    train_loader_norm = process_subjects(
        subjects=train_subjects_norm,
        batch_size=config.BATCH_SIZE,
        shuffle_data=config.SHUFFLE_DATA,
    )
    valid_loader_norm = process_subjects(
        subjects=valid_subjects_norm,
        batch_size=config.BATCH_SIZE,
        shuffle_data=False,
    )

    # Log the used atlas and the number of ROIs
    log_atlas_mode(atlas_name=config.ATLAS_NAME, num_rois=len_atlas)

    # Log data setup
    log_data_loading(
        datasets={
            "Training Data": len(train_subjects_norm),
            "Validation Data": len(valid_subjects_norm),
        }
    )
    
    ## 2. Prepare and Run Normative Modeling Pipeline --------------------------------
    log_model_setup()

    # Extract features as torch tensors
    train_data = extract_measurements(train_subjects_norm)
    valid_data = extract_measurements(valid_subjects_norm)
    log_and_print(f"Training data shape: {train_data.shape}")
    log_and_print(f"Validation data shape: {valid_data.shape}")

    # Save processed data tensors for future use
    torch.save(train_data, f"{save_dir}/data/train_data_tensor.pt")
    torch.save(valid_data, f"{save_dir}/data/valid_data_tensor.pt")
    
    # Initialize the normative VAE model
    normative_model = NormativeVAE_2D(
        input_dim=len_atlas,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=config.LATENT_DIM,
        learning_rate=config.LEARNING_RATE,
        kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
        recon_loss_weight=config.RECON_LOSS_WEIGHT,
        contr_loss_weight=config.CONTR_LOSS_WEIGHT,
        kl_warmup_epochs=kl_warmup_epochs,
        dropout_prob=0.1,
        device=device
    )
    
    log_model_ready(normative_model)
    
    log_and_print("Training baseline model before bootstrap training...")
    baseline_model, baseline_history = train_normative_model_plots(
        train_data=train_data,
        valid_data=valid_data,
        model=normative_model,
        epochs=num_epochs,
        batch_size=batch_size,
        save_best=True,
        return_history=True
    )
    
    # Save baseline model
    torch.save(baseline_model.state_dict(), f"{save_dir}/models/baseline_model.pt")
    
    # Train bootstrap models
    log_and_print("Training bootstrap models...")
    bootstrap_models, bootstrap_metrics = bootstrap_train_normative_models_plots(
        train_data=train_data,
        valid_data=valid_data,
        model=normative_model,
        n_bootstraps=n_bootstraps,
        epochs=num_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        save_models=save_models
    )
    
    ##############

# ========== DIAGNOSTIC ANALYSIS AFTER TRAINING ==========
    log_and_print("\n" + "="*80)
    log_and_print("RUNNING POST-TRAINING DIAGNOSTICS")
    log_and_print("="*80)
    
    try:
        # Run diagnostics comparing HC train vs HC validation
        hc_train_results, hc_valid_results = run_full_diagnostics(
            model=baseline_model,
            hc_loader=train_loader_norm,
            patient_loader=valid_loader_norm,
            hc_name="HC_train",
            patient_name="HC_valid",
            save_dir=f"{save_dir}/diagnostics_post_training"
        )
        
        log_and_print("✓ Post-training diagnostics completed")
        log_and_print(f"  Diagnostic plots saved to: {save_dir}/diagnostics_post_training/")
        
        # Log key findings
        hc_train_kl = hc_train_results['kl_per_sample'].mean()
        hc_valid_kl = hc_valid_results['kl_per_sample'].mean()
        
        log_and_print(f"\nKey Findings:")
        log_and_print(f"  HC Train KL: {hc_train_kl:.4f}")
        log_and_print(f"  HC Valid KL: {hc_valid_kl:.4f}")
        log_and_print(f"  Collapsed dimensions: {hc_train_results['collapse_stats']['collapsed_dims']}/{latent_dim}")
        
        if hc_train_results['collapse_stats']['collapsed_dims'] > 0:
            log_and_print(f"  ⚠️  WARNING: Posterior collapse detected!")
        
    except Exception as e:
        log_and_print(f"⚠️  Warning: Could not complete diagnostics: {e}")
        import traceback
        log_and_print(traceback.format_exc())
    
    # ========== END OF DIAGNOSTIC CODE ==========

    ##############
    log_and_print(f"Successfully trained {len(bootstrap_models)} bootstrap models")

    # Calculate and visualize overall performance
    metrics_df = pd.DataFrame(bootstrap_metrics)
    
    metrics_to_plot = [
        ('final_val_loss', 'Validation Loss'), 
        ('final_train_loss', 'Training Loss'),
        ('final_recon_loss', 'Reconstruction Loss'), 
        ('final_kl_loss', 'KL Divergence Loss')
    ]

    # Save training metadata
    training_metadata = {
        "atlas_name": joined_atlas_name,
        "num_epochs": num_epochs,
        "n_bootstraps": n_bootstraps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_dim": latent_dim,
        "kldiv_weight": kldiv_weight,
        "kl_warmup_epochs": kl_warmup_epochs,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "input_dim": len_atlas,
        "train_samples": len(train_subjects_norm),
        "valid_samples": len(valid_subjects_norm),
        "best_model_val_loss": baseline_history['best_val_loss'],
        "best_model_epoch": baseline_history['best_epoch'],
        "bootstrap_mean_val_loss": metrics_df['final_val_loss'].mean(),
        "bootstrap_std_val_loss": metrics_df['final_val_loss'].std(),
        "device": str(device),
        "normalization_method": normalization_method,  # NEW: Add to metadata
        "timestamp": timestamp
    }
    
    pd.DataFrame([training_metadata]).to_csv(f"{save_dir}/training_metadata.csv", index=False)
    
    log_and_print(f"Normative modeling training completed successfully!\nResults saved to {save_dir}")
    
    return save_dir, bootstrap_models, bootstrap_metrics

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    volume_type_arg = args.volume_type
    
    # ========== FIX: Keep as list! ==========
    if len(volume_type_arg) == 1 and volume_type_arg[0] == "all":
        volume_type_arg = ["Vgm", "Vwm", "Vcsf", "G", "T"]
    
    # Ensure it's always a list
    if isinstance(volume_type_arg, str):
        volume_type_arg = [volume_type_arg]
    
    # Run the main function
    save_dir, bootstrap_models, bootstrap_metrics = main(
        atlas_name=args.atlas_name,
        num_epochs=args.num_epochs,
        norm_diagnosis=args.norm_diagnosis,
        volume_type=volume_type_arg,  # ← Always a list now
        train_ratio=args.train_ratio,
        n_bootstraps=args.n_bootstraps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        kldiv_weight=args.kldiv_weight,
        kl_warmup_epochs=args.kl_warmup_epochs,
        save_models=args.save_models,
        no_cuda=args.no_cuda,
        seed=args.seed,
        normalization_method=args.normalization_method,
        output_dir=args.output_dir
    )
    
    print(f"Normative modeling complete. Results saved to {save_dir}")