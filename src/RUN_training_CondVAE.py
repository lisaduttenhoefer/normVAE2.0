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

from models.ConditionalVAE_2D import (
    ConditionalVAE_2D,
    ConditionalDataset,
    create_conditional_datasets,
    train_conditional_model_plots,
    bootstrap_train_conditional_models_plots,
    extract_latent_space_conditional
)
from utils.support_f import (
    split_df_adapt,
    extract_measurements
)
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    load_checkpoint_model, 
    load_mri_data_2D_conditional,
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
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--norm_diagnosis', help='which diagnosis is considered the "norm"', type=str, default="HC")
    parser.add_argument('--train_ratio', help='Normpslit ratio', type=float, default=0.7)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.000559)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=50) 
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=1.0)
    parser.add_argument('--save_models', help='Save all bootstrap models', action='store_true', default=True)
    parser.add_argument('--no_cuda', help='Disable CUDA (use CPU only)', action='store_true')
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, default=42)
    parser.add_argument('--output_dir', help='Override default output directory', default=None)
    parser.add_argument('--contr_loss_weight', type=float, default=0.1, help='Weight for contrastive loss on age bins')
    parser.add_argument('--n_age_bins', type=int, default=0,help='Number of age bins for contrastive loss')
    parser.add_argument('--normalization_method', type=str, default='rowwise', choices=['rowwise', 'columnwise'], help="Normalization method: 'rowwise' (Pinaya approach) or 'columnwise' (classical neuroimaging)")
    # ========== Dataset exclusion ==========
    parser.add_argument('--exclude_datasets', nargs='*', default=[], help='Datasets to exclude from training (e.g., EPSY NSS). Empty list = include all with IQR matching')
    parser.add_argument('--exclude_mode', type=str, choices=['strict', 'soft'], default='strict', help='strict: completely remove from data. soft: exclude from IQR matching but keep data')
    return parser


def main(atlas_name: list, volume_type, num_epochs: int, n_bootstraps: int, norm_diagnosis: str, train_ratio: float, 
         batch_size: int, learning_rate: float, latent_dim: int, kldiv_weight: float, save_models: bool, 
         kl_warmup_epochs: int, beta: float, dropout: float, no_cuda: bool, seed: int, 
         contr_loss_weight: float, n_age_bins: int, normalization_method: str = 'rowwise', output_dir: str = None, exclude_datasets=None, exclude_mode: str = None):
    
    ## 0. Set Up ----------------------------------------------------------
    path_original = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"
    path_to_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    # RAW MRI DATA (not normalized!)
    RAW_MRI_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    
    joined_atlas_name = "_".join(str(a) for a in atlas_name if isinstance(a, str))
    joined_volume_name = "_".join(str(a) for a in volume_type if isinstance(a, str))
    
    # ========== CREATE OUTPUT DIRECTORY FIRST ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if output_dir is None:
        save_dir = f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/CVAE/TRAINING/CVAE_{norm_diagnosis}_{joined_volume_name}_{joined_atlas_name}_seed{seed}_{timestamp}"
    else:
        save_dir = output_dir
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/latent_space", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/reconstructions", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    
    # ========== CREATE CONFIG ==========
    config = Config_2D(
        RUN_NAME=f"CVAE_{joined_atlas_name}_{timestamp}_seed{seed}",
        TRAIN_CSV=[],  # Not used
        TEST_CSV=[],   # Not used
        MRI_DATA_PATH=RAW_MRI_CSV,
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        VOLUME_TYPE=volume_type,
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "Vcsf", "G", "T"],
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        RECON_LOSS_WEIGHT=1.0,
        KLDIV_LOSS_WEIGHT=kldiv_weight, 
        CONTR_LOSS_WEIGHT=contr_loss_weight,
        TOTAL_EPOCHS=num_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=0.00356,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=1e-3,
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
    
    # ========== SETUP LOGGING ==========
    log_file = f"{save_dir}/logs/{timestamp}_normative_training.log"
    setup_logging(config)
    log_and_print(f"Starting CVAE training (seed={seed})")
    log_and_print(f"Atlas: {joined_atlas_name}, Epochs: {num_epochs}, Bootstraps: {n_bootstraps}")
    log_and_print(f"Normalization method: {normalization_method}")
    
    # Save configuration
    config_dict = vars(config)
    config_dict['NORMALIZATION_METHOD'] = normalization_method
    config_dict['BETA'] = beta
    config_dict['DROPOUT'] = dropout
    config_dict['SEED'] = seed
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(f"{save_dir}/config.csv", index=False)
    log_and_print(f"✓ Configuration saved")
    
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed_all(config.SEED)
    
    # Set device
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print(f"Using device: {device}")
    config.DEVICE = device
    
    ## 1. SPLIT METADATA ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 1: Splitting Metadata")
    log_and_print("="*80)

    # ========== FILTER DATASETS BEFORE SPLIT ==========
    metadata = pd.read_csv(path_original)

    if exclude_datasets:
        log_and_print(f"\n⚠️  EXCLUDING DATASETS: {exclude_datasets}")
        log_and_print(f"   Exclusion mode: {exclude_mode}")
        
        original_count = len(metadata)
        
        if exclude_mode == 'strict':
            # Complete removal from training data
            metadata = metadata[~metadata['Dataset'].isin(exclude_datasets)]
            log_and_print(f"   Strict mode: Removed {original_count - len(metadata)} subjects")
            log_and_print(f"   Remaining: {len(metadata)} subjects")
        
        else:  # soft mode
            # Mark for exclusion from IQR matching but keep in data
            metadata['_exclude_from_iqr'] = metadata['Dataset'].isin(exclude_datasets)
            excluded_count = metadata['_exclude_from_iqr'].sum()
            log_and_print(f"   Soft mode: Marked {excluded_count} subjects for IQR exclusion")
            log_and_print(f"   (Will be matched to nearest training dataset)")
        
        # Show remaining dataset distribution
        log_and_print("\n   Remaining datasets after exclusion:")
        for ds, count in metadata['Dataset'].value_counts().items():
            log_and_print(f"     {ds}: {count}")

    # Save filtered metadata temporarily
    temp_filtered_path = f"{save_dir}/data/temp_filtered_metadata.csv"
    metadata.to_csv(temp_filtered_path, index=False)

    # Now split this filtered data
    train_metadata, test_metadata = split_df_adapt(
        path_original=temp_filtered_path,  # ← Use filtered!
        path_to_dir=path_to_dir,
        norm_diagnosis=norm_diagnosis,
        train_ratio=train_ratio,
        random_seed=seed,
        save_splits=True
    )

    log_and_print(f"✓ Split: {len(train_metadata)} train, {len(test_metadata)} test")
    # Save for documentation
    train_metadata.to_csv(f"{save_dir}/data/train_metadata.csv", index=False)
    test_metadata.to_csv(f"{save_dir}/data/test_metadata.csv", index=False)
    
    ## 2. LOAD AND NORMALIZE MRI DATA ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 2: Loading and Normalizing MRI Data")
    log_and_print("="*80)
    
    # Load raw MRI data
    raw_mri_data = pd.read_csv(RAW_MRI_CSV)
    log_and_print(f"✓ Loaded {len(raw_mri_data)} subjects from raw MRI file")
    
    # Get training HC filenames
    train_hc = train_metadata[train_metadata['Diagnosis'] == norm_diagnosis]
    train_hc_filenames = train_hc['Filename'].tolist()
    
    log_and_print(f"✓ Using {len(train_hc_filenames)} training HC subjects for normalization")
    
    # ========== NORMALIZE ON-THE-FLY (using existing functions!) ==========
    from module.data_processing_hc import normalize_data_iqr, validate_normalization
    
    normalized_mri_data, normalization_stats = normalize_data_iqr(
        mri_data=raw_mri_data,
        train_hc_filenames=train_hc_filenames,
        volume_types=config.VOLUME_TYPE,
        atlas_filter=None
    )
    
    # Validate normalization
    validate_normalization(
        normalized_data=normalized_mri_data,
        hc_filenames=train_hc_filenames,
        normalization_stats=normalization_stats,
        split_name="Training"
    )
    
    # Save normalized data and stats
    normalized_mri_data.to_csv(f"{save_dir}/data/normalized_mri_data.csv", index=False)
    log_and_print(f"✓ Saved normalized MRI data")
    
    import pickle
    with open(f"{save_dir}/data/normalization_stats.pkl", 'wb') as f:
        pickle.dump(normalization_stats, f)
    log_and_print(f"✓ Saved normalization stats")
    
    ## 3. SAVE AS TEMP CSV AND LOAD WITH EXISTING FUNCTION ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 3: Loading Subjects")
    log_and_print("="*80)
    
    # Save to temp CSV
    temp_mri_path = f"{save_dir}/data/temp_normalized_mri.csv"
    temp_meta_path = f"{save_dir}/data/temp_train_metadata.csv"
    
    normalized_mri_data.to_csv(temp_mri_path, index=False)
    train_metadata.to_csv(temp_meta_path, index=False)
    
    subjects_train, train_overview, roi_names_train = load_mri_data_2D_conditional(
        normalized_csv_path=temp_mri_path,
        csv_paths=[temp_meta_path],
        diagnoses=[norm_diagnosis],  # Only HC for training
        atlas_name=config.ATLAS_NAME,
        volume_type=config.VOLUME_TYPE
    )
    
    log_and_print(f"✓ Loaded {len(subjects_train)} training subjects")
    log_and_print(f"✓ Features: {len(roi_names_train)} ROIs")
    
    len_atlas = len(roi_names_train)
    
    ## 4. SPLIT TRAIN/VAL ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 4: Creating Train/Validation Split")
    log_and_print("="*80)
    
    train_annotations_norm, valid_annotations_norm = train_val_split_annotations(
        annotations=train_overview,
        diagnoses=norm_diagnosis
    )
    
    train_subjects_norm, valid_subjects_norm = train_val_split_subjects(
        subjects=subjects_train,
        train_ann=train_annotations_norm,
        val_ann=valid_annotations_norm
    )
    
    log_and_print(f"✓ Train: {len(train_subjects_norm)}, Val: {len(valid_subjects_norm)}")
    
    ## 5. CREATE CONDITIONAL DATASETS ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 5: Creating Conditional Datasets")
    log_and_print("="*80)
    
    train_measurements = extract_measurements(train_subjects_norm).numpy()
    valid_measurements = extract_measurements(valid_subjects_norm).numpy()
    
    train_dataset, valid_dataset = create_conditional_datasets(
        train_measurements=train_measurements,
        train_metadata=train_annotations_norm,
        valid_measurements=valid_measurements,
        valid_metadata=valid_annotations_norm,
        n_age_bins=n_age_bins
    )
    
    log_and_print(f"✓ Datasets: train={len(train_dataset)}, val={len(valid_dataset)}")
    log_and_print(f"✓ Condition dim: {train_dataset.condition_dim}")
    log_and_print(f"✓ Input dim: {len_atlas}")
    
    # Save processed data tensors
    torch.save(train_measurements, f"{save_dir}/data/train_data_tensor.pt")
    torch.save(valid_measurements, f"{save_dir}/data/valid_data_tensor.pt")
    
    ## 6. TRAIN MODEL ----------------------------------------------------------
    log_and_print("\n" + "="*80)
    log_and_print("STEP 6: Training Model")
    log_and_print("="*80)
    
    # Initialize model
    normative_model = ConditionalVAE_2D(
        input_dim=len_atlas,
        condition_dim=train_dataset.condition_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=config.LATENT_DIM,
        learning_rate=learning_rate,
        kldiv_loss_weight=kldiv_weight,
        recon_loss_weight=config.RECON_LOSS_WEIGHT,
        contr_loss_weight=contr_loss_weight,
        kl_warmup_epochs=kl_warmup_epochs,
        dropout_prob=dropout,
        beta=beta,
        device=device,
    )
    
    log_model_ready(normative_model)
    
    # Train baseline model
    log_and_print("Training baseline model...")
    baseline_model, baseline_history = train_conditional_model_plots(
        train_data=train_dataset,
        valid_data=valid_dataset,
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
    bootstrap_models, bootstrap_metrics = bootstrap_train_conditional_models_plots(
        train_data=train_dataset,
        valid_data=valid_dataset,
        model=normative_model,
        n_bootstraps=n_bootstraps,
        epochs=num_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        save_models=save_models
    )
    
    log_and_print(f"✓ Trained {len(bootstrap_models)} bootstrap models")
    
    ## 7. POST-TRAINING ANALYSIS (bleibt wie vorher) ----------------------------------------------------------
    # ... (dein UMAP code etc.) ...
    
    ## 8. SAVE METADATA ----------------------------------------------------------
    metrics_df = pd.DataFrame(bootstrap_metrics)
    
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
        "normalization_method": normalization_method,
        "timestamp": timestamp,
        "beta": beta,
        "dropout": dropout,
        "seed": seed
    }
    
    pd.DataFrame([training_metadata]).to_csv(f"{save_dir}/training_metadata.csv", index=False)
    
    # Save conditioning info
    conditioning_info = {
        'age_scaler_mean': train_dataset.age_scaler.mean_[0],
        'age_scaler_scale': train_dataset.age_scaler.scale_[0],
        'iqr_scaler_mean': train_dataset.iqr_scaler.mean_[0],
        'iqr_scaler_scale': train_dataset.iqr_scaler.scale_[0],
        'dataset_categories': train_dataset.dataset_categories,
        'condition_dim': train_dataset.condition_dim,
        'n_age_bins': n_age_bins if contr_loss_weight > 0 else 0,
        'age_bin_edges': train_dataset.age_bin_edges.tolist() if train_dataset.age_bin_edges is not None else None
    }
    
    import json
    with open(f"{save_dir}/conditioning_info.json", 'w') as f:
        json.dump(conditioning_info, f, indent=2)
    
    # ========== CLEANUP TEMP FILES ==========
    log_and_print("\nCleaning up temporary files...")
    
    temp_files = [
        f"{save_dir}/data/temp_normalized_mri.csv",
        f"{save_dir}/data/temp_train_metadata.csv"
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            log_and_print(f"  Deleted: {temp_file}")
    
    log_and_print("✓ Cleanup complete")
    
    log_and_print(f"\n✅ Training complete! Results saved to {save_dir}")
    
    return save_dir, bootstrap_models, bootstrap_metrics

   

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    volume_type_arg = args.volume_type
    if len(volume_type_arg) == 1 and volume_type_arg[0] == "all":
        volume_type_arg = ["Vgm", "Vwm", "Vcsf", "G", "T"]
    if isinstance(volume_type_arg, str):
        volume_type_arg = [volume_type_arg]
    
    # Run the main function with parsed arguments
    save_dir, bootstrap_models, bootstrap_metrics = main(
        atlas_name=args.atlas_name,
        num_epochs=args.num_epochs,
        norm_diagnosis=args.norm_diagnosis,
        volume_type=volume_type_arg, 
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
        output_dir=args.output_dir,
        beta=args.beta,  
        dropout=args.dropout,
        contr_loss_weight=args.contr_loss_weight,  # ← ADD THIS
        n_age_bins=args.n_age_bins,  # ← ADD THIS
        exclude_datasets=args.exclude_datasets,
        exclude_mode=args.exclude_mode
    )
    
    # Final log message
    print(f"Normative modeling complete. Results saved to {save_dir}")