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

# ========== IMPORT CVAE VERSIONS ==========
from models.ConditionalVAE_2D import ConditionalVAE_2D, ConditionalDataset
from utils.support_f import extract_measurements
from utils.config_utils_model import Config_2D

from module.data_processing_hc import load_mri_data_2D_conditional

from utils.logging_utils import (
    setup_logging_test, 
    log_and_print_test, 
    end_logging
)

# ========== IMPORT CVAE DEVIATION UTILS ==========
from utils.dev_scores_utils_CVAE import (
    calculate_deviations_cvae,
    calculate_reconstruction_deviation_cvae,
    calculate_kl_divergence_deviation_cvae,
    calculate_latent_deviation_aguila_cvae,
    calculate_combined_deviation,
    compute_hc_latent_stats_cvae,
    visualize_embeddings_multiple_cvae,
    save_latent_visualizations,
)

# Shared functions (no model interaction)
from utils.dev_scores_utils import (
    plot_all_deviation_metrics_errorbar,
    analyze_regional_deviations,
    create_corrected_correlation_heatmap,
)

from diagnose_vae import run_full_diagnostics


def create_conditions_tensor_from_metadata(metadata_df, conditioning_info):
    """
    Create conditions tensor for CVAE from metadata using training scalers.
    
    Args:
        metadata_df: DataFrame with Age, Sex, IQR, Dataset columns
        conditioning_info: Dict with scalers and categories from training
    
    Returns:
        conditions_tensor: Tensor [n_samples, condition_dim]
    """
    n_samples = len(metadata_df)
    
    # ========== AGE (normalized) ==========
    age_values = metadata_df['Age'].values.reshape(-1, 1)
    age_normalized = (age_values - conditioning_info['age_scaler_mean']) / conditioning_info['age_scaler_scale']
    age_normalized = age_normalized.flatten()
    
    # ========== SEX (binary) ==========
    if metadata_df['Sex'].dtype == 'object':
        sex_values = (metadata_df['Sex'].str.upper() == 'M').astype(float).values
    else:
        sex_values = metadata_df['Sex'].values.astype(float)
    
    # ========== IQR (normalized) ==========
    iqr_values = metadata_df['IQR'].values.reshape(-1, 1)
    iqr_normalized = (iqr_values - conditioning_info['iqr_scaler_mean']) / conditioning_info['iqr_scaler_scale']
    iqr_normalized = iqr_normalized.flatten()
    
    # ========== DATASET (one-hot) ==========
    dataset_categories = conditioning_info['dataset_categories']
    dataset_onehot = np.zeros((n_samples, len(dataset_categories)))
    
    for i, dataset in enumerate(metadata_df['Dataset']):
        if dataset in dataset_categories:
            idx = dataset_categories.index(dataset)
            dataset_onehot[i, idx] = 1.0
        else:
            log_and_print_test(f"Warning: Unknown dataset '{dataset}' - using zeros")
    
    # ========== COMBINE ALL ==========
    conditions = np.column_stack([
        age_normalized,
        sex_values,
        iqr_normalized,
        dataset_onehot
    ])
    
    conditions_tensor = torch.FloatTensor(conditions)
    
    log_and_print_test(f"Created conditions tensor: {conditions_tensor.shape}")
    log_and_print_test(f"  - Age (normalized): 1")
    log_and_print_test(f"  - Sex (binary): 1")
    log_and_print_test(f"  - IQR (normalized): 1")
    log_and_print_test(f"  - Dataset (one-hot): {len(dataset_categories)}")
    
    return conditions_tensor


def analyze_volume_type_separately_cvae(
    vtype, bootstrap_models, clinical_data, conditions_tensor, annotations_df,
    roi_names, norm_diagnosis, device, base_save_dir, conditioning_info,
    mri_data_path, atlas_name, metadata_path, custom_colors
):
    """
    Analyze a specific volume type separately (CVAE version).
    
    This is a CVAE-adapted version of analyze_volume_type_separately.
    """
    
    log_and_print_test(f"\n{'='*80}")
    log_and_print_test(f"ANALYZING VOLUME TYPE: {vtype} (CVAE)")
    log_and_print_test(f"{'='*80}\n")
    
    # Create subdirectory for this volume type
    vtype_save_dir = os.path.join(base_save_dir, f"volume_type_{vtype}")
    os.makedirs(vtype_save_dir, exist_ok=True)
    os.makedirs(f"{vtype_save_dir}/figures", exist_ok=True)
    os.makedirs(f"{vtype_save_dir}/figures/distributions", exist_ok=True)
    
    # Filter ROI names for this volume type
    vtype_roi_indices = [i for i, name in enumerate(roi_names) if name.startswith(f"{vtype}_")]
    vtype_roi_names = [roi_names[i] for i in vtype_roi_indices]
    
    if not vtype_roi_names:
        log_and_print_test(f"No ROIs found for volume type {vtype}, skipping...")
        return
    
    log_and_print_test(f"Found {len(vtype_roi_names)} ROIs for {vtype}")
    
    # Filter clinical data for this volume type
    vtype_data = clinical_data[:, vtype_roi_indices]
    log_and_print_test(f"Filtered data shape: {vtype_data.shape}")
    
    # ========== CALCULATE DEVIATIONS FOR THIS VOLUME TYPE ==========
    log_and_print_test(f"\nCalculating deviation scores for {vtype}...")
    
    # Bootstrap method
    results_df_vtype = calculate_deviations_cvae(
        normative_models=bootstrap_models,
        data_tensor=vtype_data,
        conditions_tensor=conditions_tensor,
        norm_diagnosis=norm_diagnosis,
        annotations_df=annotations_df,
        device=device,
        roi_names=vtype_roi_names
    )
    
    # Additional metrics (using baseline model)
    baseline_model = bootstrap_models[0]
    
    hc_mask = annotations_df['Diagnosis'] == norm_diagnosis
    hc_data_vtype = vtype_data[hc_mask]
    hc_conditions = conditions_tensor[hc_mask]
    
    # D_MSE
    deviation_recon = calculate_reconstruction_deviation_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        device=device
    )
    
    # D_KL
    deviation_kl = calculate_kl_divergence_deviation_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        device=device
    )
    
    # D_latent
    hc_latent_stats = compute_hc_latent_stats_cvae(
        model=baseline_model,
        hc_data=hc_data_vtype.numpy(),
        hc_conditions=hc_conditions.numpy(),
        device=device
    )
    
    deviation_latent, _ = calculate_latent_deviation_aguila_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        hc_latent_stats=hc_latent_stats,
        device=device
    )
    
    # D_combined
    deviation_combined = calculate_combined_deviation(
        recon_dev=deviation_recon,
        kl_dev=deviation_kl
    )
    
    # Add to results
    results_df_vtype['deviation_score_recon'] = deviation_recon
    results_df_vtype['deviation_score_kl'] = deviation_kl
    results_df_vtype['deviation_score_latent_aguila'] = deviation_latent
    results_df_vtype['deviation_score_combined'] = deviation_combined
    
    log_and_print_test(f"✓ Computed all deviation metrics for {vtype}")
    
    # ========== SAVE RESULTS ==========
    results_file = os.path.join(vtype_save_dir, f"deviation_scores_{vtype}.csv")
    results_df_vtype.to_csv(results_file, index=False)
    log_and_print_test(f"Saved: {results_file}")
    
    # ========== PLOTS ==========
    plot_all_deviation_metrics_errorbar(
        results_df=results_df_vtype,
        save_dir=vtype_save_dir,
        norm_diagnosis=norm_diagnosis,
        custom_colors=custom_colors,
        name=f"{vtype}_Analysis"
    )
    
    # ========== REGIONAL ANALYSIS ==========
    log_and_print_test(f"\nPerforming regional analysis for {vtype}...")
    
    regional_results = analyze_regional_deviations(
        results_df=results_df_vtype,
        save_dir=vtype_save_dir,
        clinical_data_path=mri_data_path,
        volume_type=[vtype],
        atlas_name=atlas_name,
        roi_names=vtype_roi_names,
        norm_diagnosis=norm_diagnosis,
        name=f"{vtype}_analysis",
        add_catatonia_subgroups=False,
        metadata_path=metadata_path,
        merge_CAT_groups=True
    )
    
    if regional_results is not None and not regional_results.empty:
        regional_file = os.path.join(vtype_save_dir, f"regional_effect_sizes_{vtype}.csv")
        regional_results.to_csv(regional_file, index=False)
        log_and_print_test(f"Saved regional results: {regional_file}")
    
    log_and_print_test(f"✓ Completed analysis for {vtype}")


def main(args):
    # ---------------------- INITIAL SETUP --------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    default_model_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/analysis/TRAINING/norm_results_HC_..."
    model_dir = args.model_dir if args.model_dir else default_model_dir
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    model_name = os.path.basename(model_dir)
    save_dir = f"{args.output_dir}/clinical_deviations_{model_name}_{timestamp}" if args.output_dir else f"./deviation_results_{model_name}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    log_and_print_test("="*80)
    log_and_print_test("STARTING CVAE DEVIATION ANALYSIS")
    log_and_print_test("="*80)
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # ---------------------- LOAD MODEL CONFIG AND CONDITIONING INFO --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"Loaded model configuration from {config_path}")
        
        # Extract parameters
        atlas_name = config_df["ATLAS_NAME"].iloc[0]
        if atlas_name.startswith('['):
            atlas_name = eval(atlas_name)
        
        latent_dim = int(config_df["LATENT_DIM"].iloc[0])
        norm_diagnosis = config_df["DIAGNOSES"].iloc[0] if "DIAGNOSES" in config_df.columns else args.norm_diagnosis
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else ["Vgm", "T", "G"]
        
        if volume_type.startswith('['):
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
        contr_loss_weight = float(config_df["CONTR_LOSS_WEIGHT"].iloc[0])
        
        TEST_CSV = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv
        mri_data_path = config_df["MRI_DATA_PATH"].iloc[0] if "MRI_DATA_PATH" in config_df.columns else "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
        metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
        
        # ========== LOAD CONDITIONING INFO ==========
        conditioning_info_path = os.path.join(model_dir, "conditioning_info.json")
        
        if os.path.exists(conditioning_info_path):
            with open(conditioning_info_path, 'r') as f:
                conditioning_info = json.load(f)
            log_and_print_test(f"✓ Loaded conditioning info from {conditioning_info_path}")
            log_and_print_test(f"  Condition dim: {conditioning_info['condition_dim']}")
            log_and_print_test(f"  Dataset categories: {conditioning_info['dataset_categories']}")
        else:
            raise FileNotFoundError(
                f"conditioning_info.json not found in {model_dir}!\n"
                "Please re-run training with updated script."
            )
        
        hidden_dim_1 = 100
        hidden_dim_2 = 100
        
        log_and_print_test(f"\nConfiguration:")
        log_and_print_test(f"  Atlas: {atlas_name}")
        log_and_print_test(f"  Volume types: {volume_type}")
        log_and_print_test(f"  Latent dim: {latent_dim}")
        log_and_print_test(f"  Normative diagnosis: {norm_diagnosis}")
        
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
    
    # ------------------------------------------ LOADING CLINICAL DATA --------------------------------------------
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LOADING CLINICAL DATA")
    log_and_print_test("="*80)
    
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_IQR_HC_separate_TEST.csv"
    
    subjects_dev, annotations_dev, roi_names = load_mri_data_2D_conditional(
        normalized_csv_path=NORMALIZED_CSV,
        csv_paths=[TEST_CSV],
        diagnoses=None,  # All diagnoses
        atlas_name=atlas_name,
        volume_type=volume_type
    )
    
    clinical_data = extract_measurements(subjects_dev)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"Subjects: {len(annotations_dev)}")
    
    # ========== CREATE CONDITIONS TENSOR ==========
    log_and_print_test("\n" + "="*80)
    log_and_print_test("CREATING CONDITIONS TENSOR")
    log_and_print_test("="*80)
    
    required_cols = ['Age', 'Sex', 'IQR', 'Dataset']
    missing_cols = [col for col in required_cols if col not in annotations_dev.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    conditions_tensor = create_conditions_tensor_from_metadata(
        metadata_df=annotations_dev,
        conditioning_info=conditioning_info
    )
    
    if conditions_tensor.shape[1] != conditioning_info['condition_dim']:
        raise ValueError(
            f"Condition dimension mismatch: "
            f"expected {conditioning_info['condition_dim']}, "
            f"got {conditions_tensor.shape[1]}"
        )
    
    log_and_print_test("✓ Conditions tensor created successfully")
    
    # ------------------------------------------ LOADING MODELS --------------------------------------------
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LOADING BOOTSTRAP CVAE MODELS")
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
    encoder_input_dim = checkpoint['encoder.0.weight'].shape[1]
    input_dim = encoder_input_dim - conditioning_info['condition_dim']
    
    log_and_print_test(f"Model expects:")
    log_and_print_test(f"  Input dim: {input_dim}")
    log_and_print_test(f"  Condition dim: {conditioning_info['condition_dim']}")
    
    if input_dim != clinical_data.shape[1]:
        raise ValueError(f"Dimension mismatch: models={input_dim}, data={clinical_data.shape[1]}")
    
    # Load all models
    bootstrap_models = []
    for model_file in model_files:
        model = ConditionalVAE_2D(
            input_dim=input_dim,
            condition_dim=conditioning_info['condition_dim'],
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            kldiv_loss_weight=kldiv_loss_weight,
            recon_loss_weight=recon_loss_weight,
            contr_loss_weight=contr_loss_weight,
            dropout_prob=0.1,
            device=device
        )
        
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        bootstrap_models.append(model)
    
    log_and_print_test(f"✓ Loaded {len(bootstrap_models)} CVAE models")
    
    # Load baseline model
    baseline_model_path = os.path.join(models_dir, "baseline_model.pt")
    if os.path.exists(baseline_model_path):
        baseline_model = ConditionalVAE_2D(
            input_dim=input_dim,
            condition_dim=conditioning_info['condition_dim'],
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            kldiv_loss_weight=kldiv_loss_weight,
            recon_loss_weight=recon_loss_weight,
            contr_loss_weight=contr_loss_weight,
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
    # PRE-TESTING DIAGNOSTICS
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("RUNNING PRE-TESTING DIAGNOSTICS")
    log_and_print_test("="*80)
    
    try:
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        
        # Create HC test loader
        hc_test_dataset = TensorDataset(
            clinical_data[hc_mask],
            conditions_tensor[hc_mask],
            torch.zeros(hc_mask.sum())
        )
        hc_test_loader = DataLoader(hc_test_dataset, batch_size=32, shuffle=False)
        
        # Create patient test loader (e.g., MDD)
        mdd_mask = annotations_dev['Diagnosis'] == 'MDD'
        if mdd_mask.sum() > 0:
            mdd_test_dataset = TensorDataset(
                clinical_data[mdd_mask],
                conditions_tensor[mdd_mask],
                torch.ones(mdd_mask.sum())
            )
            mdd_test_loader = DataLoader(mdd_test_dataset, batch_size=32, shuffle=False)
            
            # NOTE: run_full_diagnostics needs to be updated for CVAE!
            # For now, skip or create CVAE version
            log_and_print_test("⚠️  Diagnostic analysis needs CVAE adaptation - skipping for now")
            
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
    
    try:
        # [1] Bootstrap method
        log_and_print_test("\n[1/5] BOOTSTRAP deviation scores...")
        results_df = calculate_deviations_cvae(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            conditions_tensor=conditions_tensor,
            norm_diagnosis=norm_diagnosis,
            annotations_df=annotations_dev,
            device=device,
            roi_names=roi_names
        )
        
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        hc_data = clinical_data[hc_mask]
        hc_conditions = conditions_tensor[hc_mask]
        
        # [2-5] Additional metrics
        log_and_print_test("\n[2/5] D_MSE (Reconstruction)...")
        deviation_recon = calculate_reconstruction_deviation_cvae(
            model=baseline_model,
            data=clinical_data.numpy(),
            conditions=conditions_tensor.numpy(),
            device=device
        )
        
        log_and_print_test("\n[3/5] D_KL...")
        deviation_kl = calculate_kl_divergence_deviation_cvae(
            model=baseline_model,
            data=clinical_data.numpy(),
            conditions=conditions_tensor.numpy(),
            device=device
        )
        
        log_and_print_test("\n[4/5] D_latent (Aguila)...")
        hc_latent_stats = compute_hc_latent_stats_cvae(
            model=baseline_model,
            hc_data=hc_data.numpy(),
            hc_conditions=hc_conditions.numpy(),
            device=device
        )
        
        deviation_latent, _ = calculate_latent_deviation_aguila_cvae(
            model=baseline_model,
            data=clinical_data.numpy(),
            conditions=conditions_tensor.numpy(),
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
    
    # # ==================================================================================
    # # PER-VOLUME-TYPE ANALYSIS
    # # ==================================================================================
    
    # if len(volume_type) > 1:
    #     log_and_print_test("\n" + "="*80)
    #     log_and_print_test("ANALYZING EACH VOLUME TYPE SEPARATELY")
    #     log_and_print_test("="*80)
        
    #     for vtype in volume_type:
    #         try:
    #             analyze_volume_type_separately_cvae(
    #                 vtype=vtype,
    #                 bootstrap_models=bootstrap_models,
    #                 clinical_data=clinical_data,
    #                 conditions_tensor=conditions_tensor,
    #                 annotations_df=annotations_dev,
    #                 roi_names=roi_names,
    #                 norm_diagnosis=norm_diagnosis,
    #                 device=device,
    #                 base_save_dir=save_dir,
    #                 conditioning_info=conditioning_info,
    #                 mri_data_path=mri_data_path,
    #                 atlas_name=atlas_name,
    #                 metadata_path=metadata_path,
    #                 custom_colors=custom_colors
    #             )
    #         except Exception as e:
    #             log_and_print_test(f"ERROR in {vtype} analysis: {e}")
    #             import traceback
    #             log_and_print_test(traceback.format_exc())
    
    # ==================================================================================
    # LATENT SPACE VISUALIZATION
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LATENT SPACE VISUALIZATION")
    log_and_print_test("="*80)
    
    try:
        results_viz = visualize_embeddings_multiple_cvae(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            conditions_tensor=conditions_tensor,
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
    log_and_print_test("CVAE TESTING COMPLETE")
    log_and_print_test("="*80)
    log_and_print_test(f"\nOutput directory: {save_dir}")
    log_and_print_test(f"\nKey outputs:")
    log_and_print_test(f"  COMBINED ANALYSIS:")
    log_and_print_test(f"    - deviation_scores_combined.csv (all 5 methods)")
    log_and_print_test(f"    - deviation_score_summary.csv")
    log_and_print_test(f"    - regional_effect_sizes_combined.csv")
    log_and_print_test(f"  VOLUME-SPECIFIC ANALYSIS:")
    for vtype in volume_type:
        log_and_print_test(f"    - volume_type_{vtype}/")
    log_and_print_test(f"  VISUALIZATIONS:")
    log_and_print_test(f"    - figures/distributions/ (errorbar plots)")
    log_and_print_test(f"    - figures/latent_embeddings/ (UMAP)")
    log_and_print_test(f"    - figures/paper_style_*.png (regional effects)")
    log_and_print_test("="*80)
    
    end_logging(Config_2D)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAE Deviation Analysis - Complete")
    parser.add_argument("--model_dir", help="Path to CVAE model directory")
    parser.add_argument("--clinical_csv", help="Path to clinical metadata CSV")
    parser.add_argument("--norm_diagnosis", type=str, default="HC")
    parser.add_argument("--max_models", type=int, default=0, help="Max models (0=all)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)