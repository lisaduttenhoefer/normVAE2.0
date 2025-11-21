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
matplotlib.use("Agg")

from models.ContrastVAE_2D import (
    NormativeVAE_2D, 
    train_normative_model_plots,
    bootstrap_train_normative_models_plots
)
from diagnose_vae import run_full_diagnostics
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
    train_val_split_subjects,
    load_harmonized_data
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

test_split = 0.5 

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling Training')
    # ========== MAIN PARAMETERS ==========
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.',  nargs='+', default=["all"])
    parser.add_argument('--volume_type', help='Volume type(s) to use', nargs='*', default=["Vgm", "Vwm", "Vcsf"])
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for', type=int, default=250)
    parser.add_argument('--n_bootstraps', help='Number of bootstrap samples', type=int, default=80)
    parser.add_argument('--norm_diagnosis', help='which diagnosis is considered the "norm"', type=str, default="HC")
    parser.add_argument('--train_ratio', help='Normpslit ratio', type=float, default=0.7)
    parser.add_argument('--exclude_datasets',nargs='*',default=[],help='Datasets to exclude')
    # ========== MODEL PARAMETERS ==========
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.000559)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=20) 
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=1.2)
    parser.add_argument('--kl_warmup_epochs', type=int, default=50, help='Number of epochs for KL warmup')
    # ==========  BASIC PARAMETERS ==========
    parser.add_argument('--save_models', help='Save all bootstrap models', action='store_true', default=True)
    parser.add_argument('--no_cuda', help='Disable CUDA (use CPU only)', action='store_true')
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, default=42)
    parser.add_argument('--output_dir', help='Override default output directory', default=None)
    parser.add_argument('--normalization_method',type=str, default='rowwise', choices=['rowwise', 'columnwise'], help="Normalization method: 'rowwise' (Pinaya approach) or 'columnwise' (classical neuroimaging)")
    # ========== HARMONIZATION OPTION ==========
    parser.add_argument('--use_harmonized', action='store_true', help='Use pre-harmonized data from neuroHarmonize instead of IQR normalization')
    parser.add_argument('--harmonized_train_path', type=str, default='/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/hc_train_roi_harmonized_50.csv', help='Path to harmonized training data')
    parser.add_argument('--harmonized_app_path', type=str, default='/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/application_roi_harmonized_noNU_50.csv', help='Path to harmonized application data (test HC + patients)')
    parser.add_argument('--harmonized_split_path', type=str, default='/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/app_split_info_noNU_50.csv', help='Path to split info (which samples are test vs patient)')
    return parser


def main(atlas_name: list, volume_type, num_epochs: int, n_bootstraps: int, norm_diagnosis: str, train_ratio: float, 
         batch_size: int, learning_rate: float, latent_dim: int, kldiv_weight: float, save_models: bool, kl_warmup_epochs: int,
         no_cuda: bool, seed: int, normalization_method: str = 'rowwise', output_dir: str = None,
         use_harmonized: bool = False, harmonized_train_path: str = None, harmonized_app_path: str = None, 
         harmonized_split_path: str = None, exclude_datasets=None):
    
    ## 0. Set Up ----------------------------------------------------------
    path_original = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"
    path_to_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    # ==========Filter datasets in case of exlude_dataset parameter is not None ==========
    if exclude_datasets and len(exclude_datasets) > 0:
        log_and_print(f"\n[INFO] Excluding datasets: {exclude_datasets}")
        
        # Load and filter metadata
        full_metadata = pd.read_csv(path_original)
        filtered_metadata = full_metadata[~full_metadata['Dataset'].isin(exclude_datasets)]
        
        log_and_print(f"  Original: {len(full_metadata)} subjects")
        log_and_print(f"  After exclusion: {len(filtered_metadata)} subjects")
        log_and_print(f"  Removed: {len(full_metadata) - len(filtered_metadata)} subjects")
        
        # Save filtered metadata
        filtered_path = f"{path_to_dir}/filtered_metadata_excl_{'_'.join(exclude_datasets)}.csv"
        filtered_metadata.to_csv(filtered_path, index=False)
        
        # Use filtered metadata
        path_original = filtered_path

    #raw mri data 
    RAW_MRI_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    #name atlases /vts
    joined_atlas_name = "_".join(str(a) for a in atlas_name if isinstance(a, str))
    joined_volume_name = "_".join(str(a) for a in volume_type if isinstance(a, str))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Add harmonization info to directory name
    norm_type = "harmonized" if use_harmonized else normalization_method
    
    if output_dir is None:
        save_dir = f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/nVAE/TRAINING/norm_results_{norm_diagnosis}_{joined_volume_name}_{joined_atlas_name}_{norm_type}_{timestamp}"
    else:
        save_dir = output_dir
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/latent_space", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/reconstructions", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)    

    # ========== BRANCH: Harmonized vs Standard Pipeline ==========
    
    if use_harmonized:
        # ========== OPTION A: USE PRE-HARMONIZED DATA ==========
        log_and_print("\n🔬 Using PRE-HARMONIZED data from neuroHarmonize")
        
        normalized_mri_data, train_metadata, test_metadata, normalized_csv_path = load_harmonized_data(
            harmonized_train_path=harmonized_train_path,
            harmonized_app_path=harmonized_app_path,
            harmonized_split_path=harmonized_split_path,
            atlas_name=atlas_name,
            volume_type=volume_type,
            save_dir=save_dir,
            exclude_datasets=exclude_datasets,  
            metadata_path=path_original  
        )
        # Save normalization info
        normalization_info = {
            'method': 'neuroHarmonize',
            'harmonized_train_path': harmonized_train_path,
            'harmonized_app_path': harmonized_app_path,
            'timestamp': timestamp
        }
        
    else:
        # ========== OPTION B: STANDARD IQR NORMALIZATION ==========
        log_and_print("\n📊 Using STANDARD IQR normalization (on-the-fly)")
        
        # Create train/test split
        train_metadata, test_metadata = split_df_adapt(
            path_original=path_original,
            path_to_dir=path_to_dir,
            norm_diagnosis=norm_diagnosis,
            train_ratio=train_ratio,
            random_seed=seed,
            save_splits=True
        )
        
        # Get training HC filenames
        train_hc = train_metadata[train_metadata['Diagnosis'] == norm_diagnosis]
        train_hc_filenames = train_hc['Filename'].tolist()
        
        # Load raw MRI data
        raw_mri_data = pd.read_csv(RAW_MRI_CSV)
        
        # Normalize using training HC stats
        from module.data_processing_hc import normalize_data_iqr
        
        normalized_mri_data, normalization_stats = normalize_data_iqr(
            mri_data=raw_mri_data,
            train_hc_filenames=train_hc_filenames,
            volume_types=volume_type,
            atlas_filter=None
        )
        
        # Save normalized data and stats
        normalized_csv_path = f"{save_dir}/data/temp_normalized_mri.csv"
        normalized_mri_data.to_csv(normalized_csv_path, index=False)
        
        import pickle
        with open(f"{save_dir}/data/normalization_stats.pkl", 'wb') as f:
            pickle.dump(normalization_stats, f)
        
        normalization_info = {
            'method': 'IQR',
            'normalization_stats': normalization_stats,
            'timestamp': timestamp
        }
        
        log_and_print(f"✓ Saved IQR normalization stats")
    
    # Save normalization info
    import json
    with open(f"{save_dir}/data/normalization_info.json", 'w') as f:
        # Convert non-serializable objects to strings
        serializable_info = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                            for k, v in normalization_info.items()}
        json.dump(serializable_info, f, indent=2)
    
    # ========== REST OF PIPELINE IS IDENTICAL ==========
    
    config = Config_2D(
        RUN_NAME=f"NormativeVAE20_{joined_atlas_name}_{timestamp}_{norm_diagnosis}_{norm_type}",
        TRAIN_CSV=[f"{save_dir}/data/train_metadata_harmonized.csv" if use_harmonized else f"{path_to_dir}/train_metadata_{norm_diagnosis}_{train_ratio}_seed{seed}.csv"],
        TEST_CSV=[f"{save_dir}/data/test_metadata_harmonized.csv" if use_harmonized else f"{path_to_dir}/test_metadata_{norm_diagnosis}_{train_ratio}_seed{seed}.csv"],
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
    log_and_print(f"Normalization method: {norm_type}")
    log_and_print(f"Using harmonized data: {use_harmonized}")

    # Save configuration
    config_dict = vars(config)
    config_dict['NORMALIZATION_METHOD'] = normalization_method
    config_dict['USE_HARMONIZED'] = use_harmonized
    config_dict['HARMONIZED_TRAIN_PATH'] = harmonized_train_path if use_harmonized else None
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
    
    # Save temp metadata
    temp_meta_path = f"{save_dir}/data/temp_train_metadata.csv"
    train_metadata.to_csv(temp_meta_path, index=False)
    
    # Load subjects
    from module.data_processing_hc import load_mri_data_2D_conditional
    
    subjects_train, train_overview, roi_names_train = load_mri_data_2D_conditional(
        normalized_csv_path=normalized_csv_path,
        csv_paths=[temp_meta_path],
        diagnoses=[norm_diagnosis],
        atlas_name=config.ATLAS_NAME,
        volume_type=config.VOLUME_TYPE
    )
    
    # Save train metadata for testing later
    train_overview.to_csv(f"{save_dir}/data/train_metadata.csv", index=False)

    train_data_debug = extract_measurements(subjects_train)
    log_and_print(f"[DEBUG] Data shape: {train_data_debug.shape}")
    log_and_print(f"[DEBUG] Data min: {train_data_debug.min()}")
    log_and_print(f"[DEBUG] Data max: {train_data_debug.max()}")
    log_and_print(f"[DEBUG] Data mean: {train_data_debug.mean()}")
    log_and_print(f"[DEBUG] Data std: {train_data_debug.std()}")
    log_and_print(f"[DEBUG] Has NaN: {torch.isnan(train_data_debug).any()}")
    log_and_print(f"[DEBUG] Has Inf: {torch.isinf(train_data_debug).any()}")
    log_and_print(f"[DEBUG] Sample values: {train_data_debug[0, :10]}")

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
        "normalization_method": norm_type,
        "use_harmonized": use_harmonized,
        "timestamp": timestamp
    }
    
    pd.DataFrame([training_metadata]).to_csv(f"{save_dir}/training_metadata.csv", index=False)

    log_and_print("\nCleaning up temporary files...")
    
    temp_files = [
        temp_meta_path
    ]
    
    # Only delete IQR temp file if not using harmonized data
    if not use_harmonized:
        temp_files.append(normalized_csv_path)
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                log_and_print(f"  Deleted: {temp_file}")
            except:
                pass
    
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
        use_harmonized=args.use_harmonized,
        harmonized_train_path=args.harmonized_train_path,
        harmonized_app_path=args.harmonized_app_path,
        harmonized_split_path=args.harmonized_split_path,
        exclude_datasets=args.exclude_datasets
    )
    
    print(f"Normative modeling complete. Results saved to {save_dir}")