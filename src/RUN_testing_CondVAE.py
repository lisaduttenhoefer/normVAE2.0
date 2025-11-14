"""
RUN_testing_CondVAE.py

Complete CVAE testing script - adapted for on-the-fly normalization workflow.
All original functionality preserved, only data loading adapted.
"""

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
import pickle

# ========== IMPORT CVAE VERSIONS ==========
from models.ConditionalVAE_2D import ConditionalVAE_2D, ConditionalDataset
from utils.support_f import extract_measurements
from utils.config_utils_model import Config_2D

from utils.logging_utils import (
    setup_logging_test, 
    log_and_print_test, 
    end_logging
)

# ========== IMPORT CVAE UTILS (NEW!) ==========
from utils.cvae_utils import (
    create_conditions_tensor_from_metadata,
    load_and_normalize_test_data,
    analyze_volume_type_separately_cvae,
    analyze_latent_space_quality_cvae
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
    log_and_print_test("STARTING CVAE DEVIATION ANALYSIS")
    log_and_print_test("="*80)
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # ---------------------- LOAD MODEL CONFIG AND CONDITIONING INFO --------------------------------------------
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
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else ["Vgm", "T", "G"]
        
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
        contr_loss_weight = float(config_df["CONTR_LOSS_WEIGHT"].iloc[0])
        
        # ========== LOAD CONDITIONING INFO ==========
        conditioning_info_path = os.path.join(model_dir, "conditioning_info.json")
        
        if os.path.exists(conditioning_info_path):
            with open(conditioning_info_path, 'r') as f:
                conditioning_info = json.load(f)
            log_and_print_test(f"✓ Loaded conditioning info")
            log_and_print_test(f"  Condition dim: {conditioning_info['condition_dim']}")
            log_and_print_test(f"  Dataset categories: {conditioning_info['dataset_categories']}")
        else:
            raise FileNotFoundError(
                f"conditioning_info.json not found in {model_dir}!\n"
                "Please re-run training with updated script."
            )
        
        # ========== LOAD TEST METADATA ==========
        test_metadata_path = os.path.join(model_dir, "data", "test_metadata.csv")
        
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(f"Test metadata not found: {test_metadata_path}")
        
        test_metadata = pd.read_csv(test_metadata_path)
        log_and_print_test(f"✓ Loaded test metadata: {len(test_metadata)} subjects")
        
        # ========== LOAD TRAINING METADATA (for IQR matching) ==========
        training_metadata_path = os.path.join(model_dir, "data", "train_metadata.csv")
        
        if os.path.exists(training_metadata_path):
            training_metadata = pd.read_csv(training_metadata_path)
            log_and_print_test(f"✓ Loaded training metadata: {len(training_metadata)} subjects")
        else:
            log_and_print_test("⚠️  WARNING: Training metadata not found!")
            log_and_print_test("    IQR-based dataset matching will not be available")
            training_metadata = None
        
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
    
    # ------------------------------------------ LOADING AND NORMALIZING TEST DATA --------------------------------------------
    RAW_MRI_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    
    subjects_dev, annotations_dev, roi_names = load_and_normalize_test_data(
        model_dir=model_dir,
        raw_mri_csv=RAW_MRI_CSV,
        test_metadata=test_metadata,
        atlas_name=atlas_name,
        volume_type=volume_type
    )
    
    clinical_data = extract_measurements(subjects_dev)
    log_and_print_test(f"\n✓ Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"✓ Subjects: {len(annotations_dev)}")
    
    # Check diagnosis distribution
    log_and_print_test(f"\nTest set composition:")
    for diag, count in annotations_dev['Diagnosis'].value_counts().items():
        log_and_print_test(f"  {diag}: {count}")
    
    # ========== CREATE CONDITIONS TENSOR WITH IQR MATCHING ==========
    log_and_print_test("\n" + "="*80)
    log_and_print_test("CREATING CONDITIONS TENSOR WITH IQR-BASED DATASET MATCHING")
    log_and_print_test("="*80)
    
    required_cols = ['Age', 'Sex', 'IQR', 'Dataset']
    missing_cols = [col for col in required_cols if col not in annotations_dev.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    conditions_tensor, dataset_mapping_df = create_conditions_tensor_from_metadata(
        metadata_df=annotations_dev,
        conditioning_info=conditioning_info,
        training_metadata=training_metadata,  # ← NEW!
        match_unknown_datasets=True  # ← NEW!
    )
    
    if conditions_tensor.shape[1] != conditioning_info['condition_dim']:
        raise ValueError(
            f"Condition dimension mismatch: "
            f"expected {conditioning_info['condition_dim']}, "
            f"got {conditions_tensor.shape[1]}"
        )
    
    log_and_print_test("✓ Conditions tensor created successfully")
    
    # ========== SAVE DATASET MAPPING ==========
    mapping_file = os.path.join(save_dir, "dataset_mapping.csv")
    dataset_mapping_df.to_csv(mapping_file, index=False)
    log_and_print_test(f"\n✓ Saved dataset mapping to: {mapping_file}")
    
    # Print summary
    if 'Match_Type' in dataset_mapping_df.columns:
        log_and_print_test("\nDataset mapping summary:")
        for match_type in dataset_mapping_df['Match_Type'].unique():
            count = (dataset_mapping_df['Match_Type'] == match_type).sum()
            log_and_print_test(f"  {match_type}: {count} subjects")
        
        if 'iqr_matched' in dataset_mapping_df['Match_Type'].values:
            log_and_print_test("\nIQR-matched datasets:")
            matched = dataset_mapping_df[dataset_mapping_df['Match_Type'] == 'iqr_matched']
            for orig_ds in matched['Original_Dataset'].unique():
                ds_data = matched[matched['Original_Dataset'] == orig_ds]
                matched_ds = ds_data['Matched_Dataset'].iloc[0]
                mean_dist = ds_data['Match_Distance'].mean()
                log_and_print_test(f"  {orig_ds:15s} → {matched_ds:15s} (dist: {mean_dist:.3f})")
    
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
    # LATENT SPACE QUALITY ANALYSIS
    # ==================================================================================

    try:
        umap_embeddings, latent_quality = analyze_latent_space_quality_cvae(
            model=baseline_model,
            data_tensor=clinical_data,
            conditions_tensor=conditions_tensor,
            annotations_df=annotations_dev,
            save_dir=save_dir,
            device=device
        )
    except Exception as e:
        log_and_print_test(f"⚠️  Warning: Latent space analysis failed: {e}")
        import traceback
        log_and_print_test(traceback.format_exc())
    
    # ==================================================================================
    # PRE-TESTING DIAGNOSTICS (OPTIONAL - Skipped for now)
    # ==================================================================================
    
    # log_and_print_test("\n" + "="*80)
    # log_and_print_test("RUNNING PRE-TESTING DIAGNOSTICS")
    # log_and_print_test("="*80)
    
    # try:
    #     hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
    #     
    #     # Create HC test loader
    #     hc_test_dataset = TensorDataset(
    #         clinical_data[hc_mask],
    #         conditions_tensor[hc_mask],
    #         torch.zeros(hc_mask.sum())
    #     )
    #     hc_test_loader = DataLoader(hc_test_dataset, batch_size=32, shuffle=False)
    #     
    #     # NOTE: run_full_diagnostics needs to be updated for CVAE!
    #     log_and_print_test("⚠️  Diagnostic analysis needs CVAE adaptation - skipping for now")
    #     
    # except Exception as e:
    #     log_and_print_test(f"⚠️  Could not complete diagnostics: {e}")
    
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
    
    metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
    mri_data_path = RAW_MRI_CSV
    
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
    log_and_print_test(f"    - dataset_mapping.csv (IQR-based matching)")
    if len(volume_type) > 1:
        log_and_print_test(f"  VOLUME-SPECIFIC ANALYSIS:")
        for vtype in volume_type:
            log_and_print_test(f"    - volume_type_{vtype}/")
    log_and_print_test(f"  VISUALIZATIONS:")
    log_and_print_test(f"    - figures/distributions/ (errorbar plots)")
    log_and_print_test(f"    - figures/latent_embeddings/ (UMAP)")
    log_and_print_test(f"    - figures/paper_style_*.png (regional effects)")
    log_and_print_test(f"    - latent_analysis/ (quality metrics)")
    log_and_print_test("="*80)
    
    end_logging(Config_2D)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAE Deviation Analysis - Complete")
    parser.add_argument("--model_dir", required=True, help="Path to CVAE model directory")
    parser.add_argument("--norm_diagnosis", type=str, default="HC")
    parser.add_argument("--max_models", type=int, default=0, help="Max models (0=all)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)