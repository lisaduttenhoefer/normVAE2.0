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
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--norm_diagnosis', help='which diagnosis is considered the "norm"', type=str, default="HC")
    parser.add_argument('--train_ratio', help='Normpslit ratio', type=float, default=0.7)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.000559)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=20) 
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=0.15)
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
         batch_size: int, learning_rate: float, latent_dim: int, kldiv_weight: float, save_models: bool, kl_warmup_epochs: int, beta: int, dropout: int,
         no_cuda: bool, seed: int, normalization_method: str = 'rowwise', output_dir: str = None):
    ## 0. Set Up ----------------------------------------------------------
    # Set main paths
    path_original = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"
    path_to_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    # Splits the metadata file with ALL Patients in a training and test set
    TRAIN_CSV, TEST_CSV = split_df_adapt(path_original, path_to_dir, norm_diagnosis, train_ratio, seed)
    
    joined_atlas_name = "_".join(str(a) for a in atlas_name if isinstance(a, str))
    joined_volume_name = "_".join(str(a) for a in volume_type if isinstance(a, str))
    
    # Create output directory with normalization method in name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if output_dir is None:
        save_dir = f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/TRAINING/norm_results_{norm_diagnosis}_{joined_volume_name}_{joined_atlas_name}_{normalization_method}_{timestamp}"
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
        # Input / Output Paths
        TRAIN_CSV=[TRAIN_CSV],
        TEST_CSV=[TEST_CSV],
        MRI_DATA_PATH="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv",
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        VOLUME_TYPE=volume_type,
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "Vcsf", "G", "T"],
        # Loading Model
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        # Loss Parameters
        RECON_LOSS_WEIGHT=16,
        KLDIV_LOSS_WEIGHT=kldiv_weight, 
        CONTR_LOSS_WEIGHT=0.0,
        # Learning and Regularization
        TOTAL_EPOCHS=num_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=0.00356,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=1e-3,
        SCHEDULE_ON_VALIDATION=True,
        SCHEDULER_PATIENCE=6,
        SCHEDULER_FACTOR=0.5,
        # Visualization
        CHECKPOINT_INTERVAL=5,
        DONT_PLOT_N_EPOCHS=0,
        UMAP_NEIGHBORS=30,
        UMAP_DOT_SIZE=20,
        METRICS_ROLLING_WINDOW=10,
        # Data Parameters
        BATCH_SIZE=batch_size,
        DIAGNOSES=norm_diagnosis,  
        # Misc.
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
    log_and_print(f"Normalization method: {normalization_method}")  # NEW: Log normalization method

    # Save configuration
    config_dict = vars(config)
    config_dict['NORMALIZATION_METHOD'] = normalization_method  # NEW: Add to config
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(f"{save_dir}/config.csv", index=False)
    log_and_print(f"Configuration saved to {save_dir}/config.csv")

    # Set seed for reproducibility
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
    
    # ========== MODIFIED: Pass normalization_method parameter ==========
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC_separate_TRAIN.csv"

    subjects_train, train_overview, roi_names_train = load_mri_data_2D_conditional(
        normalized_csv_path=NORMALIZED_CSV,
        csv_paths=[TRAIN_CSV],
        diagnoses=["HC"],
        atlas_name=config.ATLAS_NAME,      
        volume_type=config.VOLUME_TYPE     
    )
    
    # ========== VERIFY ALIGNMENT ==========
    log_and_print("\n=== Verifying subjects and metadata alignment ===")
    log_and_print(f"subjects_train: {len(subjects_train)}")
    log_and_print(f"train_overview: {len(train_overview)}")

    if len(subjects_train) > 0:
        sample_subject = subjects_train[0]
        log_and_print(f"Sample subject keys: {sample_subject.keys()}")
        
        if 'Filename' not in sample_subject:
            raise ValueError("❌ 'Filename' key missing in subjects! Update load_mri_data_2D_prenormalized!")
        
        log_and_print(f"Sample subject Filename: {sample_subject['Filename']}")

    subject_filenames = {s['Filename'] for s in subjects_train}
    overview_filenames = set(train_overview['Filename'].tolist())

    overlap = subject_filenames & overview_filenames
    log_and_print(f"Filename overlap: {len(overlap)}/{len(subject_filenames)}")

    if len(overlap) < len(subject_filenames):
        log_and_print("⚠️  WARNING: Not all subjects have matching metadata!")
    else:
        log_and_print("✓ All subjects have matching metadata")
        
    #schauen das train_overview auch diese Spalten hat
    required_cols = ['Age', 'Sex', 'IQR', 'Dataset']
    missing_cols = [col for col in required_cols if col not in train_overview.columns]
    if missing_cols:
        raise ValueError(f"Metadata fehlt Spalten: {missing_cols}")
    
    # Prüfe ob alle Spalten da sind:
    log_and_print(f"Loaded columns: {train_overview.columns.tolist()}")

    # Wenn Sex One-Hot ist (0/1), ist das perfekt für ConditionalDataset!
    # Nur sicherstellen dass es 'Sex' heißt und nicht 'Sex_Male':
    if 'Sex_Male' in train_overview.columns:
        train_overview = train_overview.rename(columns={'Sex_Male': 'Sex'})

    # Validierung (optional, aber gut für Debugging):
    required_cols = ['Age', 'Sex', 'IQR', 'Dataset']
    missing_cols = [col for col in required_cols if col not in train_overview.columns]
    if missing_cols:
        raise ValueError(f"❌ Fehlende Spalten: {missing_cols}")

    log_and_print("✓ All required columns present!")
    log_and_print(f"  Age range: {train_overview['Age'].min():.1f} - {train_overview['Age'].max():.1f}")
    log_and_print(f"  Sex values: {sorted(train_overview['Sex'].unique())}")
    log_and_print(f"  Datasets: {sorted(train_overview['Dataset'].unique())}")
    log_and_print(f"  IQR range: {train_overview['IQR'].min():.3f} - {train_overview['IQR'].max():.3f}")

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

    # Prepare data loaders -> werden von Conditional Datasets selbst erstellt, keine Funktion mehr nötig

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

    # Extrahiere Measurements
    train_measurements = extract_measurements(train_subjects_norm).numpy()
    valid_measurements = extract_measurements(valid_subjects_norm).numpy()

    # Erstelle Conditional Datasets
    train_dataset, valid_dataset = create_conditional_datasets(
        train_measurements=train_measurements,
        train_metadata=train_annotations_norm,
        valid_measurements=valid_measurements,
        valid_metadata=valid_annotations_norm
    )

    log_and_print(f"Conditional datasets created:")
    log_and_print(f"  Train: {len(train_dataset)} samples")
    log_and_print(f"  Valid: {len(valid_dataset)} samples")
    log_and_print(f"  Condition dim: {train_dataset.condition_dim}")

    # Save processed data tensors for future use
    torch.save(train_measurements, f"{save_dir}/data/train_data_tensor.pt")
    torch.save(valid_measurements, f"{save_dir}/data/valid_data_tensor.pt") 
    
    # Initialize the normative VAE model
    normative_model = ConditionalVAE_2D(
        input_dim=len_atlas,
        condition_dim=train_dataset.condition_dim,  # ← NEU!
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=config.LATENT_DIM,
        learning_rate=learning_rate,
        kldiv_loss_weight=kldiv_weight,
        recon_loss_weight=config.RECON_LOSS_WEIGHT,
        contr_loss_weight=config.CONTR_LOSS_WEIGHT,
        kl_warmup_epochs=kl_warmup_epochs,
        dropout_prob=dropout,
        beta=beta,  # ← NEU für β-VAE
        device=device
    )
    
    log_model_ready(normative_model)
    
    log_and_print("Training baseline model before bootstrap training...")
    baseline_model, baseline_history = train_conditional_model_plots(
        train_data=train_dataset,  # ← ConditionalDataset statt Tensor!
        valid_data=valid_dataset,  # ← ConditionalDataset statt Tensor!
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
    
    log_and_print(f"Successfully trained {len(bootstrap_models)} bootstrap models")

    # Calculate and visualize overall performance
    metrics_df = pd.DataFrame(bootstrap_metrics)
    
    # ========== POST-TRAINING ANALYSIS (HIER EINFÜGEN!) ==========
    log_and_print("\n" + "="*80)
    log_and_print("POST-TRAINING ANALYSIS")
    log_and_print("="*80)

    import umap
    from sklearn.metrics import silhouette_score

    # Extract latents
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    all_latents = []
    all_conditions = []

    baseline_model.eval()
    with torch.no_grad():
        for measurements, conditions, _, _ in train_loader:
            measurements = measurements.to(device)
            conditions = conditions.to(device)
            latent = baseline_model.to_latent(measurements, conditions)
            all_latents.append(latent.cpu().numpy())
            all_conditions.append(conditions.cpu().numpy())

    latents = np.concatenate(all_latents)
    conditions = np.concatenate(all_conditions)

    # Decompose conditions
    ages = conditions[:, 0]
    sexes = conditions[:, 1]
    iqrs = conditions[:, 2]
    dataset_onehot = conditions[:, 3:]
    dataset_idx = np.argmax(dataset_onehot, axis=1)

    # UMAP
    log_and_print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(latents)

    # Silhouette Score (HAUPTMETRIK!)
    silhouette = silhouette_score(latents, dataset_idx)
    log_and_print(f"\n=== DATASET MIXING SCORE ===")
    log_and_print(f"Silhouette Score: {silhouette:.3f}")

    if silhouette < 0.2:
        log_and_print("✓ EXCELLENT! Datasets well mixed (site effects removed)")
    elif silhouette < 0.4:
        log_and_print("✓ GOOD! Datasets moderately mixed")
    else:
        log_and_print("⚠️  WARNING! Datasets still separated (site effects remain)")

    # Correlation with conditions
    log_and_print(f"\n=== CONDITION CORRELATION ===")
    max_age_corr = max([abs(np.corrcoef(latents[:, i], ages)[0, 1]) for i in range(latents.shape[1])])
    max_sex_corr = max([abs(np.corrcoef(latents[:, i], sexes)[0, 1]) for i in range(latents.shape[1])])
    max_iqr_corr = max([abs(np.corrcoef(latents[:, i], iqrs)[0, 1]) for i in range(latents.shape[1])])

    log_and_print(f"Max Age correlation: {max_age_corr:.3f}")
    log_and_print(f"Max Sex correlation: {max_sex_corr:.3f}")
    log_and_print(f"Max IQR correlation: {max_iqr_corr:.3f}")

    if max_age_corr < 0.3 and max_sex_corr < 0.3:
        log_and_print("✓ GOOD! Conditions well disentangled")
    else:
        log_and_print("⚠️  Some conditions still in latent space")

    # Create UMAP plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Age
    scatter = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1],
                                 c=ages, cmap='viridis', s=10, alpha=0.6)
    axes[0, 0].set_title('UMAP by Age (normalized)', fontsize=12)
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[0, 0], label='Age')

    # Sex
    sex_colors = ['#FF6B6B' if s == 0 else '#4ECDC4' for s in sexes]
    axes[0, 1].scatter(embedding[:, 0], embedding[:, 1],
                       c=sex_colors, s=10, alpha=0.6)
    axes[0, 1].set_title('UMAP by Sex', fontsize=12)
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    from matplotlib.patches import Patch
    axes[0, 1].legend(handles=[
        Patch(facecolor='#FF6B6B', label='Female'),
        Patch(facecolor='#4ECDC4', label='Male')
    ])

    # IQR
    scatter = axes[1, 0].scatter(embedding[:, 0], embedding[:, 1],
                                 c=iqrs, cmap='plasma', s=10, alpha=0.6)
    axes[1, 0].set_title('UMAP by IQR (normalized)', fontsize=12)
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[1, 0], label='IQR')

    # Dataset (WICHTIGSTER PLOT!)
    n_datasets = len(train_dataset.dataset_categories)
    colors_map = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    dataset_colors = [colors_map[idx] for idx in dataset_idx]

    axes[1, 1].scatter(embedding[:, 0], embedding[:, 1],
                       c=dataset_colors, s=10, alpha=0.6)
    axes[1, 1].set_title(f'UMAP by Dataset (Silhouette: {silhouette:.2f})', fontsize=12)
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')

    legend_elements = [
        Patch(facecolor=colors_map[i], label=train_dataset.dataset_categories[i])
        for i in range(n_datasets)
    ]
    axes[1, 1].legend(handles=legend_elements, loc='best', fontsize=8, ncol=2)

    plt.suptitle(f'Conditional VAE Latent Space Analysis (β={baseline_model.beta})', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/cvae_latent_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    log_and_print(f"\n✓ Analysis complete! Plots saved to {save_dir}/figures/")
    log_and_print("="*80 + "\n")
    
    # ========== END OF ANALYSIS ==========
    
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
        "normalization_method": normalization_method,
        "silhouette_score": silhouette,  # ← NEU! Save in metadata
        "max_age_corr": max_age_corr,     # ← NEU!
        "max_sex_corr": max_sex_corr,     # ← NEU!
        "max_iqr_corr": max_iqr_corr,     # ← NEU!
        "timestamp": timestamp,
        "beta": beta,                      # ← NEU!
        "dropout": dropout                 # ← NEU!
    }
    
    pd.DataFrame([training_metadata]).to_csv(f"{save_dir}/training_metadata.csv", index=False)
    
    log_and_print(f"Normative modeling training completed successfully!\nResults saved to {save_dir}")
    
    return save_dir, bootstrap_models, bootstrap_metrics  # ← RETURN muss am Ende sein!


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    volume_type_arg = args.volume_type
    if len(volume_type_arg) == 1 and volume_type_arg[0] == "all":
        volume_type_arg = ["Vgm", "Vwm", "Vcsf", "G", "T"]
    elif len(volume_type_arg) == 1:
        volume_type_arg = volume_type_arg[0]
    
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
        dropout=args.dropout
    )
    
    # Final log message
    print(f"Normative modeling complete. Results saved to {save_dir}")