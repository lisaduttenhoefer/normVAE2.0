import argparse
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from models.ConditionalVAE_2D import (
    ConditionalVAE_2D,
    ConditionalDataset,
    create_conditional_datasets

)
from utils.support_f import (
    get_all_data,
    extract_measurements
)
from utils.config_utils_model import Config_2D
from module.data_processing_hc import (
    load_mri_data_2D_prenormalized,
)
from utils.logging_utils import (
    setup_logging_test, 
    log_and_print_test, 
    end_logging
)

# ========== CVAE-SPECIFIC FUNCTIONS ==========
from utils.dev_scores_utils_CVAE import (
    # Core deviation functions (need conditions)
    calculate_deviations_cvae,
    calculate_reconstruction_deviation_cvae,
    calculate_kl_divergence_deviation_cvae,
    calculate_latent_deviation_aguila_cvae,
    compute_hc_latent_stats_cvae,
    visualize_embeddings_multiple_cvae,
    
    # Statistical helpers (no model needed)
    calculate_cliffs_delta,
    bootstrap_cliffs_delta_ci,
    calculate_combined_deviation,
    
    # Formatting helpers
    format_roi_name_for_plotting,
    format_roi_names_list_for_plotting,
    
    # Visualization helpers
    save_latent_visualizations,
    
    # Shared analysis functions (imported from original)
    plot_all_deviation_metrics_errorbar,
    plot_deviation_distributions,
    analyze_regional_deviations,
    create_corrected_correlation_heatmap,
    run_analysis_with_options,
)

# You DON'T need to import from dev_scores_utils.py anymore!

from diagnose_vae import run_full_diagnostics

def main(args):
    # ---------------------- INITIAL SETUP (output dirs, device, seed) --------------------------------------------
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #--------------------------------------- NECESSARY ARGUMENTS -----------------------------------------------------
    # Default model directory - UPDATE THIS to your v2 training results
    default_model_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/catatonia_VAE-main_v2/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_G_T_all_20251022_1625"
    
    # Use command line argument if provided, otherwise use default
    model_dir = args.model_dir if args.model_dir else default_model_dir
    #-----------------------------------------------------------------------------------------------------------------

    # Check if model_dir exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Extract model name from the model directory for consistent naming
    model_name = os.path.basename(model_dir)
    save_dir = f"{args.output_dir}/clinical_deviations_{model_name}_{timestamp}" if args.output_dir else f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/catatonia_VAE-main_v2/analysis/nVAE/TESTING/deviation_results_{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Set up logging
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    # Log start of analysis
    log_and_print_test("Starting COMBINED deviation analysis (all volume types together)")
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # ========== CVAE NEEDS METADATA: Load metadata for conditions ==========
    log_and_print_test("Loading CVAE metadata for test set...")

    # Load the complete metadata CSV
    metadata_cvae = pd.read_csv(
        "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"
    )

    # Rename Sex_Male to Sex if needed
    if 'Sex_Male' in metadata_cvae.columns and 'Sex' not in metadata_cvae.columns:
        metadata_cvae = metadata_cvae.rename(columns={'Sex_Male': 'Sex'})

    log_and_print_test(f"Loaded CVAE metadata: {metadata_cvae.shape}")

    # ---------------------- LOAD MODEL CONFIG FROM TRAINING (consistency)  --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"Loaded model configuration from {config_path}")
        
        # Extract relevant parameters from config.csv
        atlas_name = config_df["ATLAS_NAME"].iloc[0]
        # Handle the case where atlas_name is a list represented as a string
        if atlas_name.startswith('[') and atlas_name.endswith(']'):
            atlas_name = eval(atlas_name)  # Convert string representation to list
        elif atlas_name.startswith('"[') and atlas_name.endswith(']"'):
            atlas_name = eval(atlas_name.strip('"'))
        
        # Extract other parameters
        latent_dim = int(config_df["LATENT_DIM"].iloc[0])
        norm_diagnosis = config_df["DIAGNOSES"].iloc[0] if "DIAGNOSES" in config_df.columns else args.norm_diagnosis
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else ["Vgm", "Vwm", "Vcsf"]
        learning_rate = float(config_df["LEARNING_RATE"].iloc[0])
        kldiv_loss_weight = float(config_df["KLDIV_LOSS_WEIGHT"].iloc[0])
        recon_loss_weight = float(config_df["RECON_LOSS_WEIGHT"].iloc[0])
        contr_loss_weight = float(config_df["CONTR_LOSS_WEIGHT"].iloc[0])

        # Parse volume_type from config
        if volume_type.startswith('[') and volume_type.endswith(']'):
            volume_type = eval(volume_type)  # Convert string representation to list
        elif volume_type.startswith('"[') and volume_type.endswith(']"'):
            volume_type = eval(volume_type.strip('"'))
        if isinstance(volume_type, str):
            volume_type = [volume_type]

        # Get valid volume types
        valid_volume_types = eval(config_df["VALID_VOLUME_TYPES"].iloc[0]) if "VALID_VOLUME_TYPES" in config_df.columns else ["Vgm", "Vwm", "Vcsf", "G", "T"]
        
        # CRITICAL FIX: If volume_type contains "all", replace it with valid_volume_types
        if "all" in volume_type or (len(volume_type) == 1 and volume_type[0] == "all"):
            volume_type = valid_volume_types
            log_and_print_test(f"Resolved 'all' to actual volume types: {volume_type}")
        metadata_test = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv
        
        # NEW: For v2, MRI_DATA_PATH points to the single CSV file
        mri_data_path = config_df["MRI_DATA_PATH"].iloc[0] if "MRI_DATA_PATH" in config_df.columns else "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
        
        # NEW: Metadata path for v2
        metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
        
        atlas_volume_string = f"{'/'.join(atlas_name) if isinstance(atlas_name, list) else atlas_name} - {'/'.join(volume_type)}"
        hidden_dim_1 = 100  # Default
        hidden_dim_2 = 100  # Default
        
        log_and_print_test(f"Using atlas: {atlas_name}")
        log_and_print_test(f"Using latent dimension: {latent_dim}")
        log_and_print_test(f"Using normative diagnosis: {norm_diagnosis}")
        log_and_print_test(f"Using volume type: {volume_type}")
        log_and_print_test(f"Using valid volume types: {valid_volume_types}")
        log_and_print_test(f"Using clinical mri data path: {mri_data_path}")
        log_and_print_test(f"Using metadata CSV: {metadata_test}")
        
    except (FileNotFoundError, KeyError) as e:
        log_and_print_test(f"Warning: Could not load config file properly. Error: {e}")
        log_and_print_test("Using command line arguments as fallback")
        # Set fallback values
        atlas_name = args.atlas_name if args.atlas_name else ["all"]
        latent_dim = args.latent_dim
        norm_diagnosis = args.norm_diagnosis
        volume_type = ["Vgm", "Vwm", "Vcsf", "G", "T"]
        valid_volume_types = ["Vgm", "Vwm", "Vcsf", "G", "T"]
        mri_data_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
        metadata_test = args.clinical_csv
        metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata_no_bad_scans.csv"
        hidden_dim_1 = 100
        hidden_dim_2 = 100
        
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print_test(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # ------------------------------------------ LOADING CLINICAL DATA  --------------------------------------------
    log_and_print_test("Loading clinical data...")
    
    # Set paths for clinical data - now points to the single CSV file
    path_to_clinical_data = mri_data_path
    
    # Clean metadata_test path if needed
    if metadata_test.startswith('[') and metadata_test.endswith(']'):
        metadata_test = eval(metadata_test)[0] if isinstance(eval(metadata_test), list) else eval(metadata_test)
    
    log_and_print_test(f"Loading test data from: {metadata_test}")
    log_and_print_test(f"Using MRI data from: {path_to_clinical_data}")

    # Load clinical data - NEW: using load_mri_data_2D with column-wise normalization
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC_separate_TEST.csv"

    TEST_CSV = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv

    subjects_dev, annotations_dev, roi_names = load_mri_data_2D_prenormalized(
        normalized_csv_path=NORMALIZED_CSV,
        csv_paths=[TEST_CSV],
        diagnoses=None,  # All diagnoses for testing
        covars=[],
        atlas_name=atlas_name,      # ← NEU: Use same atlases as training
        volume_type=volume_type     # ← NEU: Use same volume types as training
    )
    
    # Extract clinical data tensor
    clinical_data = extract_measurements(subjects_dev)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"Clinical data has {clinical_data.shape[1]} features")
    
    # ============================================================================
    # Filter annotations_dev to match clinical_data subjects
    log_and_print_test(f"Annotations shape before filtering: {len(annotations_dev)}")
    log_and_print_test(f"Clinical data subjects: {clinical_data.shape[0]}")
    
    # Get the SubjectID column name (adjust if your column has a different name)
    # Common names: 'SubjectID', 'Subject', 'ID', 'subject_id', etc.
    subject_id_column = None
    for possible_name in ['SubjectID', 'Subject', 'ID', 'subject_id', 'bids_name']:
        if possible_name in annotations_dev.columns:
            subject_id_column = possible_name
            break
    
    if subject_id_column is None:
        log_and_print_test("WARNING: Could not find subject ID column in annotations")
        log_and_print_test(f"Available columns: {annotations_dev.columns.tolist()}")
        # Fallback: assume annotations and clinical_data are already aligned by row order
        # Just truncate annotations to match clinical_data length
        annotations_dev = annotations_dev.iloc[:clinical_data.shape[0]]
        log_and_print_test(f"Fallback: Truncated annotations to first {len(annotations_dev)} rows")
    else:
        # Get subject IDs that actually have MRI data
        # subjects_dev is a list of Subject objects, each has subject_id attribute
        available_subject_ids = [subj.subject_id for subj in subjects_dev]
        
        # Filter annotations to only include subjects with MRI data
        annotations_dev = annotations_dev[annotations_dev[subject_id_column].isin(available_subject_ids)]
        
        # Ensure the order matches subjects_dev
        annotations_dev = annotations_dev.set_index(subject_id_column).loc[available_subject_ids].reset_index()
        
        log_and_print_test(f"Filtered annotations using subject ID column: {subject_id_column}")
    
    log_and_print_test(f"Annotations shape after filtering: {len(annotations_dev)}")

    # Verify alignment
    if len(annotations_dev) != clinical_data.shape[0]:
        raise ValueError(f"Mismatch: {len(annotations_dev)} annotations vs {clinical_data.shape[0]} clinical data")

    log_and_print_test("✓ Annotations and clinical data are now aligned!")

    # ========== SCHRITT 3: MERGE CVAE METADATA (HIER EINFÜGEN!) ==========
    log_and_print_test("\n" + "="*80)
    log_and_print_test("MERGING CVAE METADATA")
    log_and_print_test("="*80)

    # Check which columns are missing
    required_cols = ['Age', 'Sex', 'IQR', 'Dataset']
    missing_cols = [col for col in required_cols if col not in annotations_dev.columns]

    if missing_cols:
        log_and_print_test(f"Missing columns: {missing_cols} - merging from metadata CSV")
        
        # Merge
        merge_cols = ['Filename'] + missing_cols
        annotations_dev = annotations_dev.merge(
            metadata_cvae[merge_cols],  # metadata_cvae wurde in Schritt 2 geladen!
            on='Filename',
            how='left'
        )
        
        log_and_print_test(f"✓ Merged columns: {merge_cols}")
    else:
        log_and_print_test("✓ All required columns already present")

    # Rename Sex_Male if needed
    if 'Sex_Male' in annotations_dev.columns and 'Sex' not in annotations_dev.columns:
        annotations_dev = annotations_dev.rename(columns={'Sex_Male': 'Sex'})

    # Convert Sex to float
    if annotations_dev['Sex'].dtype == 'object':
        log_and_print_test("Converting Sex from string to numeric...")
        annotations_dev['Sex'] = annotations_dev['Sex'].map({
            'F': 0.0, 'Female': 0.0, 'f': 0.0,
            'M': 1.0, 'Male': 1.0, 'm': 1.0,
            0: 0.0, 1: 1.0
        })

    annotations_dev['Sex'] = annotations_dev['Sex'].astype(float)
    annotations_dev['Age'] = annotations_dev['Age'].astype(float)
    annotations_dev['IQR'] = annotations_dev['IQR'].astype(float)

    # Validate - Drop rows with missing metadata
    for col in required_cols:
        n_missing = annotations_dev[col].isna().sum()
        if n_missing > 0:
            log_and_print_test(f"⚠️  {n_missing} samples missing {col} - dropping these rows")
            # Also drop from clinical_data!
            valid_mask = ~annotations_dev[col].isna()
            annotations_dev = annotations_dev[valid_mask].reset_index(drop=True)
            clinical_data = clinical_data[valid_mask.values]
            log_and_print_test(f"   New shape: {len(annotations_dev)} samples")

    log_and_print_test(f"\n✓ Metadata validated")
    log_and_print_test(f"  Age range: {annotations_dev['Age'].min():.1f} - {annotations_dev['Age'].max():.1f}")
    log_and_print_test(f"  Sex values: {sorted(annotations_dev['Sex'].unique())}")
    log_and_print_test(f"  Datasets: {sorted(annotations_dev['Dataset'].unique())}")
    log_and_print_test(f"  IQR range: {annotations_dev['IQR'].min():.3f} - {annotations_dev['IQR'].max():.3f}")
    log_and_print_test(f"  Final sample count: {len(annotations_dev)}")

    # ========== SCHRITT 4: CREATE CONDITIONAL DATASET (DIREKT DANACH) ==========
    log_and_print_test("\n" + "="*80)
    log_and_print_test("CREATING CONDITIONAL DATASET")
    log_and_print_test("="*80)

    # Try to load dataset categories from training
    try:
        train_metadata_path = os.path.join(model_dir, "data", "train_metadata.csv")
        if os.path.exists(train_metadata_path):
            train_metadata = pd.read_csv(train_metadata_path)
            dataset_categories = sorted(train_metadata['Dataset'].unique().tolist())
            log_and_print_test(f"✓ Loaded dataset categories from training: {dataset_categories}")
        else:
            dataset_categories = sorted(annotations_dev['Dataset'].unique().tolist())
            log_and_print_test(f"⚠️  Using test data datasets (training not found): {dataset_categories}")
    except Exception as e:
        dataset_categories = sorted(annotations_dev['Dataset'].unique().tolist())
        log_and_print_test(f"⚠️  Fallback datasets: {dataset_categories}")

    # Load training scalers if available
    try:
        import pickle
        scaler_path = os.path.join(model_dir, "data", "scalers.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            age_scaler = scalers['age_scaler']
            iqr_scaler = scalers['iqr_scaler']
            log_and_print_test("✓ Loaded training scalers")
        else:
            from sklearn.preprocessing import StandardScaler
            age_scaler = StandardScaler()
            iqr_scaler = StandardScaler()
            age_scaler.fit(annotations_dev['Age'].values.reshape(-1, 1))
            iqr_scaler.fit(annotations_dev['IQR'].values.reshape(-1, 1))
            log_and_print_test("⚠️  Created new scalers (training scalers not found)")
    except Exception as e:
        from sklearn.preprocessing import StandardScaler
        age_scaler = StandardScaler()
        iqr_scaler = StandardScaler()
        age_scaler.fit(annotations_dev['Age'].values.reshape(-1, 1))
        iqr_scaler.fit(annotations_dev['IQR'].values.reshape(-1, 1))
        log_and_print_test("⚠️  Created new scalers (fallback)")

    # Create Conditional Dataset
    test_dataset = ConditionalDataset(
        measurements=clinical_data.numpy(),
        metadata=annotations_dev,
        dataset_categories=dataset_categories,
        fit_scalers=False
    )

    # Set scalers
    test_dataset.set_scalers(age_scaler, iqr_scaler)

    log_and_print_test(f"\n✓ Conditional Dataset created:")
    log_and_print_test(f"  Samples: {len(test_dataset)}")
    log_and_print_test(f"  Condition dim: {test_dataset.condition_dim}")
    log_and_print_test(f"  Dataset categories: {test_dataset.dataset_categories}")

    # Extract conditions as tensor
    test_conditions = torch.FloatTensor(test_dataset.conditions)
    log_and_print_test(f"  Conditions shape: {test_conditions.shape}")

    log_and_print_test("="*80 + "\n")


    # ------------------------------------------ LOADING MODELS  --------------------------------------------
    log_and_print_test("Loading bootstrap models...")
    
    # Find all bootstrap models
    models_dir = os.path.join(model_dir, "models")
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])
    
    # Limit number of models if specified
    if args.max_models > 0:
        model_files = model_files[:args.max_models]
        log_and_print_test(f"Using first {args.max_models} models out of {len(model_files)} available")
    
    log_and_print_test(f"Found {len(model_files)} bootstrap models")
    
    # Get input dimension from first model
    first_model_path = os.path.join(models_dir, model_files[0])
    checkpoint = torch.load(first_model_path, map_location=device)
    
    # Get input_dim from first model
    first_model_path = os.path.join(models_dir, model_files[0])
    checkpoint = torch.load(first_model_path, map_location=device)

    # CVAE: encoder.0 hat shape [hidden_dim_1, input_dim + condition_dim]
    encoder_input_size = checkpoint['encoder.0.weight'].shape[1]

    # Subtract condition_dim to get input_dim
    input_dim = encoder_input_size - test_dataset.condition_dim

    log_and_print_test(f"Models were trained with:")
    log_and_print_test(f"  - Total encoder input: {encoder_input_size}")
    log_and_print_test(f"  - Condition dim: {test_dataset.condition_dim}")
    log_and_print_test(f"  - Data input dim: {input_dim}")
    log_and_print_test(f"Models were trained with input_dim: {input_dim}")
    
    # Verify that clinical data has same dimension
    input_dim_from_data = clinical_data.shape[1]
    if input_dim != input_dim_from_data:
        log_and_print_test(f"ERROR: Input dimension mismatch!")
        log_and_print_test(f"  - Models expect: {input_dim} features")
        log_and_print_test(f"  - Test data has: {input_dim_from_data} features")
        log_and_print_test(f"This likely means volume_type settings differ between training and testing.")
        log_and_print_test(f"Please check your volume_type configuration!")
        raise ValueError(f"Input dimension mismatch: models expect {input_dim}, but data has {input_dim_from_data} features")
    
    # Get condition_dim from checkpoint or dataset
    condition_dim = test_dataset.condition_dim
    # Load all bootstrap models
    bootstrap_models = []
    for model_file in model_files:
        model = ConditionalVAE_2D(
            input_dim=input_dim,
            condition_dim=condition_dim,  
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            kldiv_loss_weight=kldiv_loss_weight,
            recon_loss_weight=recon_loss_weight,
            contr_loss_weight=contr_loss_weight,
            dropout_prob=0.1,
            beta=0.5,  # From training
            device=device
        )
        
        # Load weights
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        bootstrap_models.append(model)
        log_and_print_test(f"Loaded model: {model_file}")
    
    log_and_print_test(f"Successfully loaded {len(bootstrap_models)} models")
    

    #########

    # ========== DIAGNOSTIC ANALYSIS BEFORE DEVIATION CALCULATION ==========
    log_and_print_test("\n" + "="*80)
    log_and_print_test("RUNNING PRE-TESTING DIAGNOSTICS")
    log_and_print_test("="*80)
    
    try:
        # Load baseline model FIRST (needed for diagnostics)
        baseline_model_path = os.path.join(models_dir, "baseline_model.pt")
        if os.path.exists(baseline_model_path):
            baseline_model = ConditionalVAE_2D(
                input_dim=input_dim,
                condition_dim=condition_dim,  # ← NEU!
                hidden_dim_1=hidden_dim_1,
                hidden_dim_2=hidden_dim_2,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                kldiv_loss_weight=kldiv_loss_weight,
                recon_loss_weight=recon_loss_weight,
                contr_loss_weight=contr_loss_weight,
                dropout_prob=0.1,
                beta=0.5,
                device=device
            )
            baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
            baseline_model.to(device)
            baseline_model.eval()
            log_and_print_test("✓ Loaded baseline model for diagnostics")
        else:
            log_and_print_test("⚠ Baseline model not found, using first bootstrap model")
            baseline_model = bootstrap_models[0]
        
        # Separate HC and patient groups
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        
        hc_conditions = test_conditions[hc_mask]
        hc_test_dataset = TensorDataset(
            clinical_data[hc_mask],
            hc_conditions,  # ← NEU!
            torch.zeros(hc_mask.sum())
        )
        hc_test_loader = DataLoader(hc_test_dataset, batch_size=32, shuffle=False)

        # Ähnlich für MDD:
        mdd_mask = annotations_dev['Diagnosis'] == 'MDD'
        if mdd_mask.sum() > 0:
            mdd_conditions = test_conditions[mdd_mask]
            mdd_test_dataset = TensorDataset(
                clinical_data[mdd_mask],
                mdd_conditions,  # ← NEU!
                torch.ones(mdd_mask.sum())
            )
            mdd_test_loader = DataLoader(mdd_test_dataset, batch_size=32, shuffle=False)
            
            # Run diagnostics
            hc_results, mdd_results = run_full_diagnostics(
                model=baseline_model,
                hc_loader=hc_test_loader,
                patient_loader=mdd_test_loader,
                hc_name="HC",
                patient_name="MDD",
                save_dir=f"{save_dir}/diagnostics_pre_testing"
            )
        hc_test_loader = DataLoader(hc_test_dataset, batch_size=32, shuffle=False)

        # Create MDD test loader (if available)
        mdd_mask = annotations_dev['Diagnosis'] == 'MDD'
        if mdd_mask.sum() > 0:
            mdd_test_dataset = TensorDataset(
                clinical_data[mdd_mask],
                torch.ones(mdd_mask.sum())
            )
            mdd_test_loader = DataLoader(mdd_test_dataset, batch_size=32, shuffle=False)
            
            # Log key findings
            hc_kl = hc_results['kl_per_sample'].mean()
            mdd_kl = mdd_results['kl_per_sample'].mean()
            
            log_and_print_test(f"\nKey Findings:")
            log_and_print_test(f"  HC KL:  {hc_kl:.4f}")
            log_and_print_test(f"  MDD KL: {mdd_kl:.4f}")
            
            if mdd_kl < hc_kl:
                log_and_print_test(f"  ⚠️  PROBLEM: MDD has LOWER KL than HC!")
                log_and_print_test(f"  → Use Mahalanobis or reconstruction error instead")
            
            # Check Mahalanobis
            if 'mahalanobis' in mdd_results:
                hc_mahal = hc_results['mahalanobis'].mean()
                mdd_mahal = mdd_results['mahalanobis'].mean()
                log_and_print_test(f"\n  Mahalanobis: HC={hc_mahal:.4f}, MDD={mdd_mahal:.4f}")
                if mdd_mahal > hc_mahal:
                    log_and_print_test(f"    ✅ Mahalanobis works (MDD > HC)")
            
            # Check reconstruction
            hc_recon = hc_results['recon_error_per_sample'].mean()
            mdd_recon = mdd_results['recon_error_per_sample'].mean()
            log_and_print_test(f"\n  Reconstruction: HC={hc_recon:.6f}, MDD={mdd_recon:.6f}")
            if mdd_recon > hc_recon:
                log_and_print_test(f"    ✅ Reconstruction works (MDD > HC)")
                log_and_print_test(f"    → RECOMMENDATION: Use reconstruction-based scores!")
            
            log_and_print_test(f"\n  Full report: {save_dir}/diagnostics_pre_testing/diagnostic_report.txt")
        
    except Exception as e:
        log_and_print_test(f"⚠️  Warning: Could not complete diagnostics: {e}")
        import traceback
        log_and_print_test(traceback.format_exc())
    
    # ========== END OF DIAGNOSTIC CODE ==========
    ############

    # ==================================================================================
    # COMBINED ANALYSIS - ALL FEATURES TOGETHER
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("STARTING COMBINED DEVIATION ANALYSIS")
    log_and_print_test(f"Analyzing ALL {input_dim} features together (Vgm + G + T)")
    log_and_print_test("="*80 + "\n")
    
    # Custom colors for consistent visualization
    custom_colors = {
        "HC": "#125E8A",
        "SSD": "#3E885B",
        "MDD": "#BEDCFE",
        "CAT": "#2F4B26",
        "CAT-SSD": "#A67DB8",
        "CAT-MDD": "#160C28"
    }
    
    try:
        # ==================== CALCULATE MULTIPLE DEVIATION SCORES ====================
        log_and_print_test("\n" + "="*80)
        log_and_print_test("Computing MULTIPLE deviation score methods...")
        log_and_print_test("="*80)
        
        # Original bootstrap method (keep this!)
        log_and_print_test("\n[1/5] Calculating BOOTSTRAP deviation scores (original method)...")
        results_df = calculate_deviations_cvae(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            conditions_tensor=test_conditions,
            norm_diagnosis=norm_diagnosis,
            annotations_df=annotations_dev,
            device=device,
            roi_names=roi_names
        )
        
        # Separate HC data for latent stats
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        hc_data = clinical_data[hc_mask]
        log_and_print_test(f"   HC subjects in test set: {len(hc_data)}")
        
        # METHOD 1: Reconstruction Deviation
        log_and_print_test("\n[2/5] Computing D_MSE (Reconstruction-based)...")
        deviation_recon = calculate_reconstruction_deviation_cvae(
            model=baseline_model,
            conditions_tensor=test_conditions,
            data=clinical_data,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_recon.min():.4f}, {deviation_recon.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_recon[hc_mask].mean():.4f} ± {deviation_recon[hc_mask].std():.4f}")
        
        # METHOD 2: KL Divergence
        log_and_print_test("\n[3/5] Computing D_KL (KL Divergence)...")
        deviation_kl = calculate_kl_divergence_deviation_cvae(
            model=baseline_model,
            conditions_tensor=test_conditions,
            data=clinical_data,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_kl.min():.4f}, {deviation_kl.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_kl[hc_mask].mean():.4f} ± {deviation_kl[hc_mask].std():.4f}")
        
        # METHOD 3: Latent Space Deviation (Aguila method)
        log_and_print_test("\n[4/5] Computing D_latent (Aguila et al. method)...")
        hc_latent_stats = compute_hc_latent_stats_cvae(
            model=baseline_model,
            hc_data=hc_data,
            hc_conditions=hc_conditions,
            device=device
        )
        
        deviation_latent, per_dim_deviations = calculate_latent_deviation_aguila_cvae(
            model=baseline_model,
            conditions_tensor=test_conditions,
            data=clinical_data,
            hc_latent_stats=hc_latent_stats,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_latent.min():.4f}, {deviation_latent.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_latent[hc_mask].mean():.4f} ± {deviation_latent[hc_mask].std():.4f}")
        
        # METHOD 4: Combined Deviation Score
        log_and_print_test("\n[5/5] Computing D_combined (weighted combination)...")
        deviation_combined = calculate_combined_deviation(
            recon_dev=deviation_recon,
            kl_dev=deviation_kl,
            alpha=0.7,
            beta=0.3
        )
        log_and_print_test(f"   Range: [{deviation_combined.min():.4f}, {deviation_combined.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_combined[hc_mask].mean():.4f} ± {deviation_combined[hc_mask].std():.4f}")
        
        # ==================== ADD ALL DEVIATION SCORES TO RESULTS_DF ====================
        log_and_print_test("\n✓ Adding all deviation methods to results DataFrame...")
        
        # Convert numpy arrays to pandas Series
        results_df['deviation_score_recon'] = pd.Series(deviation_recon, index=results_df.index)
        results_df['deviation_score_kl'] = pd.Series(deviation_kl, index=results_df.index)
        results_df['deviation_score_latent_aguila'] = pd.Series(deviation_latent, index=results_df.index)
        results_df['deviation_score_combined'] = pd.Series(deviation_combined, index=results_df.index)
        
        log_and_print_test("✓ All 5 deviation methods computed:")
        log_and_print_test("   1. deviation_score (Bootstrap)")
        log_and_print_test("   2. deviation_score_recon (D_MSE)")
        log_and_print_test("   3. deviation_score_kl (D_KL)")
        log_and_print_test("   4. deviation_score_latent_aguila (D_latent)")
        log_and_print_test("   5. deviation_score_combined (Weighted)")

        # ========== CREATE ERRORBAR PLOTS FOR ALL 5 METRICS ==========
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CREATING ERRORBAR PLOTS FOR ALL 5 DEVIATION METRICS")
        log_and_print_test("="*80)
        
        plot_all_deviation_metrics_errorbar(
            results_df=results_df,
            save_dir=save_dir,
            norm_diagnosis=norm_diagnosis,
            custom_colors=custom_colors,
            name="Combined Analysis"
        )
        
        log_and_print_test("✓ Created errorbar plots for all 5 deviation metrics")
        
        ########

        # log_and_print_test("\n" + "="*80)
        # log_and_print_test("ANALYZING EACH VOLUME TYPE SEPARATELY")
        # log_and_print_test("="*80 + "\n")

        # for vtype in volume_type:
        #     log_and_print_test(f"\n{'='*80}")
        #     log_and_print_test(f"PROCESSING VOLUME TYPE: {vtype}")
        #     log_and_print_test(f"{'='*80}\n")
            
        #     try:
        #         analyze_volume_type_separately(
        #             vtype=vtype,
        #             bootstrap_models=bootstrap_models,
        #             clinical_data=clinical_data,
        #             annotations_df=annotations_dev,
        #             roi_names=roi_names,
        #             norm_diagnosis=norm_diagnosis,
        #             device=device,
        #             base_save_dir=save_dir,
        #             mri_data_path=mri_data_path,
        #             atlas_name=atlas_name,
        #             metadata_path=metadata_path,
        #             custom_colors=custom_colors,
        #             split_CAT=False,
        #             add_catatonia_subgroups=False
        #         )
                
        #         log_and_print_test(f"✓ Completed analysis for {vtype}")
                
        #     except Exception as e:
        #         log_and_print_test(f"ERROR analyzing {vtype}: {e}")
        #         import traceback
        #         log_and_print_test(traceback.format_exc())

        # log_and_print_test("\n" + "="*80)
        # log_and_print_test("ALL VOLUME TYPES ANALYZED")
        # log_and_print_test("="*80)

        # #########
        
        # ==================================================================================
        # SAVE RESULTS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("SAVING RESULTS")
        log_and_print_test("="*80 + "\n")
        
        # Save full results with ALL deviation methods
        results_file = os.path.join(save_dir, "deviation_scores_combined.csv")
        results_df.to_csv(results_file, index=False)
        log_and_print_test(f"Saved deviation scores (ALL methods) to: {results_file}")
        
        # Create and save summary statistics
        summary_stats = []
        for diagnosis in results_df['Diagnosis'].unique():
            diag_data = results_df[results_df['Diagnosis'] == diagnosis]
            
            for method in ['deviation_score', 'deviation_score_recon', 'deviation_score_kl', 
                          'deviation_score_latent_aguila', 'deviation_score_combined']:
                summary_stats.append({
                    'Diagnosis': diagnosis,
                    'Method': method,
                    'N': len(diag_data),
                    'Mean': diag_data[method].mean(),
                    'Std': diag_data[method].std(),
                    'Median': diag_data[method].median(),
                    'Q1': diag_data[method].quantile(0.25),
                    'Q3': diag_data[method].quantile(0.75)
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(save_dir, "deviation_score_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        log_and_print_test(f"Saved summary statistics to: {summary_file}")
        
        # ==================================================================================
        # VISUALIZATIONS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CREATING VISUALIZATIONS")
        log_and_print_test("="*80 + "\n")
        
        # # Plot deviation score distributions
        # plot_results = plot_deviation_distributions(
        #     results_df=results_df,
        #     save_dir=save_dir,
        #     norm_diagnosis=norm_diagnosis,
        #     col_jitter=False,
        #     name="combined_analysis"
        # )
        # log_and_print_test("Plotted deviation distributions")
        
        # ==================================================================================
        # REGIONAL ANALYSIS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("REGIONAL DEVIATION ANALYSIS")
        log_and_print_test("="*80 + "\n")
        
        # Create subdirectories for regional analysis
        os.makedirs(f"{save_dir}/figures", exist_ok=True)
        
        regional_results = analyze_regional_deviations(
            results_df=results_df,
            save_dir=save_dir,
            clinical_data_path=mri_data_path,
            volume_type=volume_type,
            atlas_name=atlas_name,
            roi_names=roi_names,
            norm_diagnosis=norm_diagnosis,
            name="combined_analysis",
            add_catatonia_subgroups=False,
            metadata_path=metadata_path,
            subgroup_columns=None,
            high_low_thresholds=None,
            merge_CAT_groups=True
        )
        
        # Save regional effect sizes
        if regional_results is not None and not regional_results.empty:
            regional_file = os.path.join(save_dir, "regional_effect_sizes_combined.csv")
            regional_results.to_csv(regional_file, index=False)
            log_and_print_test(f"Saved regional effect sizes to: {regional_file}")
            
            # Show top 20 ROIs with largest effect sizes (using Cliff's Delta)
            if 'Cliffs_Delta' in regional_results.columns:
                log_and_print_test("\nTop 20 ROIs with largest Cliff's Delta:")
                # Sort by absolute value of Cliff's Delta
                regional_results['Abs_Cliffs_Delta'] = regional_results['Cliffs_Delta'].abs()
                top_rois = regional_results.nlargest(20, 'Abs_Cliffs_Delta')
                log_and_print_test(top_rois[['ROI_Name', 'Diagnosis', 'Cliffs_Delta', 'Significant_Bootstrap_p05_uncorrected']].to_string(index=False))
        else:
            log_and_print_test("No regional results generated")
        
        # ==================================================================================
        # CORRELATION WITH CLINICAL VARIABLES
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CORRELATION WITH CLINICAL VARIABLES")
        log_and_print_test("="*80 + "\n")
        
    #     # Merge results with full metadata if available
    #     if 'Age' in annotations_dev.columns:
    #         # Create correlation heatmap
    #         fig_corr = create_corrected_correlation_heatmap(
    #             results_df=results_df,
    #             clinical_vars=['Age'],  # Add more variables as needed
    #             save_path=os.path.join(save_dir, "figures", "clinical_correlations.png")
    #         )
    #         log_and_print_test("Saved clinical correlation heatmap")
        
    except Exception as e:
        log_and_print_test(f"ERROR in combined analysis: {e}")
        import traceback
        log_and_print_test(traceback.format_exc())
    
    # ==================================================================================
    # LATENT SPACE VISUALIZATION (ALL FEATURES)
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("Visualizing latent space embeddings...")
    log_and_print_test("="*80 + "\n")
    
    try:
        results = visualize_embeddings_multiple_cvae(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            annotations_df=annotations_dev,
            conditions_tensor=test_conditions,
            device=device,
            columns_to_plot=["Diagnosis", "Dataset", "Sex"]  # Adjusted columns
        )
        
        save_latent_visualizations(results, output_dir=f"{save_dir}/figures/latent_embeddings")
        log_and_print_test("Saved latent space visualizations")
        
    except Exception as e:
        log_and_print_test(f"Warning: Could not complete latent space visualization: {e}")
    
    # NOTE: Paper-style plots (regional effect sizes) are created in 
    # analyze_regional_deviations() and saved as:
    #   - figures/paper_style_SSD_vs_HC.png
    #   - figures/paper_style_MDD_vs_HC.png
    #   - figures/paper_style_CAT_vs_HC.png
    
    # ==================================================================================
    # FINAL SUMMARY
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("TESTING COMPLETE - SUMMARY")
    log_and_print_test("="*80)
    log_and_print_test(f"Main output directory: {save_dir}")
    log_and_print_test(f"\nKey outputs:")
    log_and_print_test(f"  - deviation_scores_combined.csv (includes ALL 5 deviation methods)")
    log_and_print_test(f"  - deviation_score_summary.csv")
    log_and_print_test(f"  - regional_effect_sizes_combined.csv")
    log_and_print_test(f"\nPlots:")
    log_and_print_test(f"  Deviation distributions (figures/distributions/):")
    log_and_print_test(f"    - score_errorbar_CAT_combined.png (Bootstrap)")
    log_and_print_test(f"    - recon_errorbar_CAT_combined.png (Reconstruction)")
    log_and_print_test(f"    - kl_errorbar_CAT_combined.png (KL Divergence)")
    log_and_print_test(f"    - latent_aguila_errorbar_CAT_combined.png (Aguila method)")
    log_and_print_test(f"    - combined_errorbar_CAT_combined.png (Combined)")
    log_and_print_test(f"  Regional effect sizes (figures/):")
    log_and_print_test(f"    - paper_style_SSD_vs_HC.png")
    log_and_print_test(f"    - paper_style_MDD_vs_HC.png")
    log_and_print_test(f"    - paper_style_CAT_vs_HC.png")
    log_and_print_test(f"  Latent embeddings (figures/latent_embeddings/)")
    log_and_print_test("="*80 + "\n")
    
    end_logging(Config_2D)

    return save_dir 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate COMBINED deviation scores using all volume types together.")
    parser.add_argument("--model_dir", help="Path to model directory (default: uses predefined path in code)")
    parser.add_argument("--clinical_data_path", help="Path to clinical data CSV (default: uses path from model config)")
    parser.add_argument("--clinical_csv", help="Path to clinical metadata CSV file")
    parser.add_argument("--norm_diagnosis", type=str, default="HC", help="Normative diagnosis (default: HC)")
    parser.add_argument("--atlas_name", nargs='+', help="Atlas name(s) (if not available in config)")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension (if not available in config)")
    parser.add_argument("--max_models", type=int, default=0, help="Maximum number of models to use (0 = all)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)