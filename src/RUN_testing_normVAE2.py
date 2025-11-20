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
import json

from models.ContrastVAE_2D import NormativeVAE_2D  # ← VAE, not CVAE!
from utils.support_f import extract_measurements
from utils.config_utils_model import Config_2D
from utils.cvae_utils import load_and_normalize_test_data
from module.data_processing_hc import load_mri_data_2D_prenormalized

from utils.logging_utils import (
    setup_logging_test, 
    log_and_print_test, 
    end_logging
)

# ========== VAE-SPECIFIC FUNCTIONS (NOT CVAE!) ==========
from utils.dev_scores_utils import (
    # Core deviation functions (NO conditions needed)
    calculate_deviations,  # ← VAE version
    calculate_reconstruction_deviation,
    calculate_kl_divergence_deviation,
    calculate_latent_deviation_aguila,
    compute_hc_latent_stats,
    
    # Visualization
    plot_all_deviation_metrics_errorbar,
    plot_deviation_distributions,
    analyze_regional_deviations,
    visualize_embeddings_multiple,
    save_latent_visualizations,
    
    # Statistical helpers
    calculate_cliffs_delta,
    bootstrap_cliffs_delta_ci,
    calculate_combined_deviation,
)

from diagnose_vae import run_full_diagnostics


def load_harmonized_test_data(model_dir, test_metadata, harmonized_app_path, harmonized_split_path,
                               atlas_name, volume_type):
    """
    Load test data from pre-harmonized files.
    
    Args:
        model_dir: Path to model directory
        test_metadata: Test metadata DataFrame
        harmonized_app_path: Path to harmonized application CSV
        harmonized_split_path: Path to split info CSV
        atlas_name: Atlas name(s) to filter
        volume_type: Volume type(s) to filter
    
    Returns:
        subjects_dev: List of subject dictionaries
        annotations_dev: Test metadata DataFrame
        roi_names: List of ROI names
    """
    log_and_print_test("\n" + "="*70)
    log_and_print_test("LOADING PRE-HARMONIZED TEST DATA")
    log_and_print_test("="*70)
    
    # Load harmonized application data
    app_harm = pd.read_csv(harmonized_app_path, index_col='Filename')
    split_info = pd.read_csv(harmonized_split_path)
    
    log_and_print_test(f"Loaded harmonized application data: {app_harm.shape}")
    log_and_print_test(f"Split info: {split_info.shape}")
    
    # Get test HC filenames
    test_hc_filenames = split_info[split_info['Split'] == 'test']['Filename'].tolist()
    patient_filenames = split_info[split_info['Split'] == 'patient']['Filename'].tolist()
    
    log_and_print_test(f"Test HC samples: {len(test_hc_filenames)}")
    log_and_print_test(f"Patient samples: {len(patient_filenames)}")
    
    # Combine test HC + patients
    all_test_filenames = test_hc_filenames + patient_filenames
    
    # Filter to only filenames in test_metadata
    requested_filenames = set(test_metadata['Filename'].tolist())
    available_filenames = set(app_harm.index)
    
    valid_filenames = list(requested_filenames & available_filenames)
    
    log_and_print_test(f"Requested: {len(requested_filenames)}, Available: {len(available_filenames)}, Valid: {len(valid_filenames)}")
    
    if len(valid_filenames) == 0:
        raise ValueError("No valid filenames found in harmonized data!")
    
    # Filter harmonized data
    test_harm = app_harm.loc[valid_filenames]
    
    # Filter columns by atlas and volume type
    atlas_mapping = {
        'neuromorphometrics': 'Neurom',
        'lpba40': 'lpba40',
        'hammers': 'Hammers',
        'aal3': 'AAL3',
        'cobra': 'cobra',
        'suit': 'SUIT',
        'ibsr': 'IBSR',
        'Schaefer_100': 'Sch100',
        'Schaefer_200': 'Sch200',
        'aparc_DKT40': 'DK40',
        'aparc_dk40': 'DK40',  # ← ADD: lowercase variant
        'aparc_a2009s': 'Destrieux'
    }
    
    filtered_cols = []
    for col in test_harm.columns:
        parts = col.split('_', 2)
        if len(parts) < 2:
            continue
        
        col_volume = parts[0]
        col_atlas = parts[1]
        
        vol_match = col_volume in volume_type
        
        atlas_match = False
        if isinstance(atlas_name, list):
            if "all" in atlas_name:
                atlas_match = True
            else:
                for atlas in atlas_name:
                    atlas_short = atlas_mapping.get(atlas.lower(), atlas)
                    if col_atlas == atlas_short or atlas.lower() in col_atlas.lower():
                        atlas_match = True
                        break
        else:
            atlas_short = atlas_mapping.get(atlas_name.lower(), atlas_name)
            atlas_match = col_atlas == atlas_short or atlas_name.lower() in col_atlas.lower() or atlas_name == "all"
        
        if atlas_match and vol_match:
            filtered_cols.append(col)
    
    log_and_print_test(f"Filtered to {len(filtered_cols)} ROI columns")
    log_and_print_test(f"Example columns: {filtered_cols[:5]}")
    
    if len(filtered_cols) == 0:
        raise ValueError(f"No columns found matching atlas {atlas_name} and volume types {volume_type}!")
    
    test_harm = test_harm[filtered_cols]
    
    # ========== NORMALIZE HARMONIZED DATA WITH IQR ==========
    log_and_print_test("\n[INFO] Applying IQR normalization to harmonized test data...")
    
    # Load normalization stats from training
    norm_stats_path = f"{model_dir}/data/normalization_stats.pkl"
    
    if os.path.exists(norm_stats_path):
        import pickle
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        
        log_and_print_test(f"✓ Loaded normalization stats from training")
        
        # Apply normalization using training stats
        test_harm_df = test_harm.reset_index()  # Filename becomes column
        
        for col in filtered_cols:
            if col in norm_stats:
                median = norm_stats[col]['median']
                iqr = norm_stats[col]['iqr']
                
                if iqr > 0:
                    test_harm_df[col] = (test_harm_df[col] - median) / iqr
                else:
                    test_harm_df[col] = test_harm_df[col] - median
        
        log_and_print_test(f"✓ Normalized {len(filtered_cols)} ROI columns using training IQR statistics")
        
        # Update test_harm with normalized values
        test_harm = test_harm_df.set_index('Filename')[filtered_cols]
        
    else:
        log_and_print_test(f"⚠️ WARNING: No normalization stats found at {norm_stats_path}")
        log_and_print_test(f"  Using harmonized data without additional normalization")
    
    # Create subjects list
    subjects_dev = []
    for filename in test_harm.index:
        measurements = test_harm.loc[filename].values.astype(np.float32).tolist()
        subjects_dev.append({
            'name': filename,
            'Filename': filename,
            'measurements': measurements
        })
    
    # Filter test_metadata to matched subjects
    annotations_dev = test_metadata[test_metadata['Filename'].isin(valid_filenames)].copy()
    annotations_dev = annotations_dev.reset_index(drop=True)
    
    log_and_print_test(f"\n✓ Created {len(subjects_dev)} test subjects")
    log_and_print_test(f"✓ Metadata: {len(annotations_dev)} subjects")
    
    roi_names = filtered_cols
    
    return subjects_dev, annotations_dev, roi_names


def main(args):
    # ---------------------- INITIAL SETUP --------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = args.model_dir
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    model_name = os.path.basename(model_dir)
    save_dir = f"{args.output_dir}/results_{model_name}_{timestamp}" if args.output_dir else f"./deviation_results_{model_name}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    log_and_print_test("="*80)
    log_and_print_test("STARTING VAE DEVIATION ANALYSIS")
    log_and_print_test("="*80)
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # ---------------------- LOAD MODEL CONFIG --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"✓ Loaded model configuration from {config_path}")
        
        # Extract parameters
        atlas_name = config_df["ATLAS_NAME"].iloc[0]
        if isinstance(atlas_name, str) and atlas_name.startswith('['):
            atlas_name = eval(atlas_name)
        
        latent_dim = int(config_df["LATENT_DIM"].iloc[0])
        norm_diagnosis = config_df["DIAGNOSES"].iloc[0] if "DIAGNOSES" in config_df.columns else args.norm_diagnosis
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else ["Vgm", "Vwm", "Vcsf"]
        
        if isinstance(volume_type, str) and volume_type.startswith('['):
            volume_type = eval(volume_type)
        if isinstance(volume_type, str):
            volume_type = [volume_type]
        
        valid_volume_types = eval(config_df["VALID_VOLUME_TYPES"].iloc[0]) if "VALID_VOLUME_TYPES" in config_df.columns else ["Vgm", "Vwm", "Vcsf", "G", "T"]
        
        # Handle "all" volume types
        if "all" in volume_type or (len(volume_type) == 1 and volume_type[0] == "all"):
            volume_type = valid_volume_types
            log_and_print_test(f"Resolved 'all' to: {volume_type}")
        
        learning_rate = float(config_df["LEARNING_RATE"].iloc[0])
        kldiv_loss_weight = float(config_df["KLDIV_LOSS_WEIGHT"].iloc[0])
        recon_loss_weight = float(config_df["RECON_LOSS_WEIGHT"].iloc[0])
        
        # ========== CHECK IF MODEL USED HARMONIZATION ==========
        use_harmonized = config_df["USE_HARMONIZED"].iloc[0] if "USE_HARMONIZED" in config_df.columns else False
        if isinstance(use_harmonized, str):
            use_harmonized = use_harmonized.lower() in ['true', '1', 'yes']
        
        log_and_print_test(f"\nModel was trained with harmonization: {use_harmonized}")
        
        TEST_CSV = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv
        if TEST_CSV.startswith('['):
            TEST_CSV = eval(TEST_CSV)[0]
        
        hidden_dim_1 = 100
        hidden_dim_2 = 100
        
        log_and_print_test(f"\nConfiguration:")
        log_and_print_test(f"  Atlas: {atlas_name}")
        log_and_print_test(f"  Volume types: {volume_type}")
        log_and_print_test(f"  Latent dim: {latent_dim}")
        log_and_print_test(f"  Normative diagnosis: {norm_diagnosis}")
        log_and_print_test(f"  Used harmonization: {use_harmonized}")
        
    except (FileNotFoundError, KeyError) as e:
        log_and_print_test(f"ERROR: Could not load configuration: {e}")
        raise
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print_test(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # ------------------------------------------ LOADING TEST DATA --------------------------------------------
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LOADING TEST DATA")
    log_and_print_test("="*80)

    # Parse TEST_CSV
    TEST_CSV = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv

    if isinstance(TEST_CSV, str):
        if TEST_CSV.startswith('[') and TEST_CSV.endswith(']'):
            TEST_CSV = eval(TEST_CSV)[0]
        TEST_CSV = TEST_CSV.strip("'\"")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    test_metadata = pd.read_csv(TEST_CSV)
    log_and_print_test(f"✓ Test metadata (HC only): {len(test_metadata)} subjects")
    
    # ========== ADD PATIENT DATA ==========
    # The test_metadata from training only contains HC test samples
    # We need to add patients for regional analysis
    
    log_and_print_test("\n[INFO] Loading patient data for regional analysis...")
    
    full_metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"
    full_metadata = pd.read_csv(full_metadata_path)
    
    # Get all patients (not HC)
    patients = full_metadata[full_metadata['Diagnosis'].isin(['MDD', 'SSD', 'CAT', 'CAT-SSD', 'CAT-MDD'])].copy()
    
    log_and_print_test(f"  Found {len(patients)} patients:")
    for diag in ['MDD', 'SSD', 'CAT', 'CAT-SSD', 'CAT-MDD']:
        n = (patients['Diagnosis'] == diag).sum()
        if n > 0:
            log_and_print_test(f"    {diag}: {n}")
    
    # Combine HC test + patients
    test_metadata = pd.concat([test_metadata, patients], ignore_index=True)
    
    log_and_print_test(f"\n✓ Combined test set: {len(test_metadata)} subjects total")
    log_and_print_test(f"  Diagnosis distribution:")
    for diag, count in sorted(test_metadata['Diagnosis'].value_counts().items()):
        log_and_print_test(f"    {diag}: {count}")

    # ========== BRANCH: Load harmonized OR normalized data ==========
    
    if use_harmonized:
        # ========== OPTION A: Use harmonized test data ==========
        log_and_print_test("\n🔬 Loading PRE-HARMONIZED test data")
        
        if args.harmonized_app_path is None or args.harmonized_split_path is None:
            raise ValueError("Model was trained with harmonization, but --harmonized_app_path and --harmonized_split_path not provided!")
        
        subjects_dev, annotations_dev, roi_names = load_harmonized_test_data(
            model_dir=model_dir,
            test_metadata=test_metadata,
            harmonized_app_path=args.harmonized_app_path,
            harmonized_split_path=args.harmonized_split_path,
            atlas_name=atlas_name,
            volume_type=volume_type
        )
    else:
        # ========== OPTION B: Use IQR normalized data ==========
        log_and_print_test("\n📊 Loading IQR-NORMALIZED test data")
        
        RAW_MRI_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
        
        subjects_dev, annotations_dev, roi_names = load_and_normalize_test_data(
            model_dir=model_dir,
            raw_mri_csv=RAW_MRI_CSV,
            test_metadata=test_metadata,
            atlas_name=atlas_name,
            volume_type=volume_type
        )

    clinical_data = extract_measurements(subjects_dev)

    log_and_print_test(f"\n✓ Clinical data: {clinical_data.shape}")
    log_and_print_test(f"✓ Subjects: {len(annotations_dev)}")

    log_and_print_test(f"\nTest set composition:")
    for diag, count in sorted(annotations_dev['Diagnosis'].value_counts().items()):
        log_and_print_test(f"  {diag}: {count}")
    
    # ------------------------------------------ LOADING MODELS --------------------------------------------
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LOADING BOOTSTRAP VAE MODELS")
    log_and_print_test("="*80)
    
    models_dir = os.path.join(model_dir, "models")
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt') and 'bootstrap' in f])
    
    if args.max_models > 0:
        model_files = model_files[:args.max_models]
        log_and_print_test(f"Limited to first {args.max_models} models")
    
    log_and_print_test(f"Found {len(model_files)} bootstrap models")
    
    # Get input dimension
    first_model_path = os.path.join(models_dir, model_files[0])
    checkpoint = torch.load(first_model_path, map_location=device)
    input_dim = checkpoint['encoder.0.weight'].shape[1]  # ← VAE: no condition dim!
    
    log_and_print_test(f"Model expects input_dim: {input_dim}")
    
    if input_dim != clinical_data.shape[1]:
        raise ValueError(f"Dimension mismatch: models={input_dim}, data={clinical_data.shape[1]}")
    
    # Load all models
    bootstrap_models = []
    for model_file in model_files:
        model = NormativeVAE_2D(  # ← VAE, not ConditionalVAE_2D!
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            kldiv_loss_weight=kldiv_loss_weight,
            recon_loss_weight=recon_loss_weight,
            contr_loss_weight=0.0,
            dropout_prob=0.1,
            device=device
        )
        
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        bootstrap_models.append(model)
    
    log_and_print_test(f"✓ Loaded {len(bootstrap_models)} VAE models")
    
    # Load baseline model
    baseline_model_path = os.path.join(models_dir, "baseline_model.pt")
    if os.path.exists(baseline_model_path):
        baseline_model = NormativeVAE_2D(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            kldiv_loss_weight=kldiv_loss_weight,
            recon_loss_weight=recon_loss_weight,
            contr_loss_weight=0.0,
            dropout_prob=0.1,
            device=device
        )
        baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        baseline_model.to(device)
        baseline_model.eval()
        log_and_print_test("✓ Loaded baseline model")
    else:
        baseline_model = bootstrap_models[0]
        log_and_print_test("Using first bootstrap model as baseline")
    
    # ==================================================================================
    # PRE-TESTING DIAGNOSTICS (OPTIONAL)
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("RUNNING PRE-TESTING DIAGNOSTICS")
    log_and_print_test("="*80)
    
    try:
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        
        # Create HC test loader (NO conditions for VAE!)
        hc_test_dataset = TensorDataset(
            clinical_data[hc_mask],
            torch.zeros(hc_mask.sum())  # Dummy labels
        )
        hc_test_loader = DataLoader(hc_test_dataset, batch_size=32, shuffle=False)
        
        # Create patient loader (e.g., MDD)
        mdd_mask = annotations_dev['Diagnosis'] == 'MDD'
        if mdd_mask.sum() > 0:
            mdd_test_dataset = TensorDataset(
                clinical_data[mdd_mask],
                torch.ones(mdd_mask.sum())
            )
            mdd_test_loader = DataLoader(mdd_test_dataset, batch_size=32, shuffle=False)
            
            hc_results, mdd_results = run_full_diagnostics(
                model=baseline_model,
                hc_loader=hc_test_loader,
                patient_loader=mdd_test_loader,
                hc_name="HC",
                patient_name="MDD",
                save_dir=f"{save_dir}/diagnostics_pre_testing"
            )
            
            log_and_print_test(f"✓ Diagnostics complete")
            log_and_print_test(f"  Report: {save_dir}/diagnostics_pre_testing/")
    
    except Exception as e:
        log_and_print_test(f"⚠️  Could not complete diagnostics: {e}")
    
    # ==================================================================================
    # COMBINED ANALYSIS - ALL FEATURES TOGETHER
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("COMBINED DEVIATION ANALYSIS (ALL FEATURES)")
    log_and_print_test("="*80)
    
    custom_colors = {
        "HC": "#125E8A",
        "SSD": "#3E885B",
        "MDD": "#BEDCFE",
        "CAT": "#2F4B26",
        "CAT-SSD": "#A67DB8",
        "CAT-MDD": "#160C28"
    }
    
    metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv"

    # ========== CRITICAL: Use RAW MRI data for direction analysis ==========
    # We need RAW values (not normalized/harmonized) to determine anatomical direction
    # Even if model was trained with harmonization, we compare raw anatomical values
    mri_data_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    
    log_and_print_test("\n[INFO] Regional analysis will use RAW MRI values to determine direction of changes")
    log_and_print_test(f"[INFO] Raw MRI path: {mri_data_path}")
    
    try:
        # [1] Bootstrap method
        log_and_print_test("\n[1/5] BOOTSTRAP deviation scores...")
        results_df = calculate_deviations(  # ← VAE version (no conditions!)
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            norm_diagnosis=norm_diagnosis,
            annotations_df=annotations_dev,
            device=device,
            roi_names=roi_names
        )
        
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        hc_data = clinical_data[hc_mask]
        
        # [2-5] Additional metrics
        log_and_print_test("\n[2/5] D_MSE (Reconstruction)...")
        deviation_recon = calculate_reconstruction_deviation(
            model=baseline_model,
            data=clinical_data.numpy(),
            device=device
        )
        
        log_and_print_test("\n[3/5] D_KL...")
        deviation_kl = calculate_kl_divergence_deviation(
            model=baseline_model,
            data=clinical_data.numpy(),
            device=device
        )
        
        log_and_print_test("\n[4/5] D_latent (Aguila)...")
        hc_latent_stats = compute_hc_latent_stats(
            model=baseline_model,
            hc_data=hc_data.numpy(),
            device=device
        )
        
        deviation_latent, _ = calculate_latent_deviation_aguila(
            model=baseline_model,
            data=clinical_data.numpy(),
            hc_latent_stats=hc_latent_stats,
            device=device
        )
        
        log_and_print_test("\n[5/5] D_combined...")
        deviation_combined = calculate_combined_deviation(
            recon_dev=deviation_recon,
            kl_dev=deviation_kl
        )
        
        # Add all to results
        results_df['deviation_score_recon'] = deviation_recon
        results_df['deviation_score_kl'] = deviation_kl
        results_df['deviation_score_latent_aguila'] = deviation_latent
        results_df['deviation_score_combined'] = deviation_combined
        
        log_and_print_test("✓ All 5 deviation methods computed")
        
        # ========== PLOTS ==========
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CREATING VISUALIZATIONS")
        log_and_print_test("="*80)
        
        plot_all_deviation_metrics_errorbar(
            results_df=results_df,
            save_dir=save_dir,
            norm_diagnosis=norm_diagnosis,
            custom_colors=custom_colors,
            name="Combined_Analysis"
        )
        
        # ========== SAVE RESULTS ==========
        results_file = os.path.join(save_dir, "deviation_scores_combined.csv")
        results_df.to_csv(results_file, index=False)
        log_and_print_test(f"\n✓ Saved: {results_file}")
        
        # ========== SUMMARY STATISTICS ==========
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
        log_and_print_test(f"✓ Saved summary: {summary_file}")
        
        # ========== REGIONAL ANALYSIS ==========
        log_and_print_test("\n" + "="*80)
        log_and_print_test("REGIONAL DEVIATION ANALYSIS")
        log_and_print_test("="*80)
        
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
        
        if regional_results is not None and not regional_results.empty:
            regional_file = os.path.join(save_dir, "regional_effect_sizes_combined.csv")
            regional_results.to_csv(regional_file, index=False)
            log_and_print_test(f"✓ Saved regional results: {regional_file}")
            
            # Top ROIs
            if 'Cliffs_Delta' in regional_results.columns:
                log_and_print_test("\nTop 20 ROIs by effect size:")
                regional_results['Abs_Cliffs_Delta'] = regional_results['Cliffs_Delta'].abs()
                top_rois = regional_results.nlargest(20, 'Abs_Cliffs_Delta')
                log_and_print_test(top_rois[['ROI_Name', 'Diagnosis', 'Cliffs_Delta']].to_string(index=False))
        
    except Exception as e:
        log_and_print_test(f"ERROR in combined analysis: {e}")
        import traceback
        log_and_print_test(traceback.format_exc())
    
    # ==================================================================================
    # LATENT SPACE VISUALIZATION
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LATENT SPACE VISUALIZATION")
    log_and_print_test("="*80)
    
    try:
        results_viz = visualize_embeddings_multiple(  # ← VAE version (no conditions!)
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            annotations_df=annotations_dev,
            device=device,
            columns_to_plot=["Diagnosis", "Dataset", "Sex"]
        )
        
        save_latent_visualizations(
            results_viz,
            output_dir=f"{save_dir}/figures/latent_embeddings"
        )
        log_and_print_test("✓ Saved latent visualizations")
        
    except Exception as e:
        log_and_print_test(f"Warning: Latent visualization failed: {e}")
    
    # ==================================================================================
    # FINAL SUMMARY
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("VAE TESTING COMPLETE")
    log_and_print_test("="*80)
    log_and_print_test(f"\nOutput directory: {save_dir}")
    log_and_print_test(f"\nKey outputs:")
    log_and_print_test(f"  - deviation_scores_combined.csv (all 5 methods)")
    log_and_print_test(f"  - deviation_score_summary.csv")
    log_and_print_test(f"  - regional_effect_sizes_combined.csv")
    log_and_print_test(f"  - figures/distributions/ (errorbar plots)")
    log_and_print_test(f"  - figures/latent_embeddings/ (UMAP)")
    log_and_print_test(f"  - figures/paper_style_*.png (regional effects)")
    log_and_print_test(f"\nNormalization used: {'Harmonized' if use_harmonized else 'IQR'}")
    log_and_print_test("="*80)
    
    end_logging(Config_2D)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Deviation Analysis")
    parser.add_argument("--model_dir", required=True, help="Path to VAE model directory")
    parser.add_argument("--clinical_csv", help="Path to test metadata CSV")
    parser.add_argument("--norm_diagnosis", type=str, default="HC")
    parser.add_argument("--max_models", type=int, default=0, help="Max models (0=all)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # ========== HARMONIZATION OPTIONS ==========
    parser.add_argument(
        '--harmonized_app_path',
        type=str,
        default='/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/application_roi_harmonized_noNU.csv',
        help='Path to harmonized application data (only used if model was trained with harmonization)'
    )
    parser.add_argument(
        '--harmonized_split_path',
        type=str,
        default='/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/app_split_info_noNU.csv',
        help='Path to split info (only used if model was trained with harmonization)'
    )
    
    args = parser.parse_args()
    main(args)