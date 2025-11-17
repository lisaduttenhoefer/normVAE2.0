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

from models.ContrastVAE_2D import NormativeVAE_2D  # ← VAE, not CVAE!
from utils.support_f import extract_measurements
from utils.config_utils_model import Config_2D
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
    
    # Use PRE-NORMALIZED data
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC_separate_TEST.csv"
    
    subjects_dev, annotations_dev, roi_names = load_mri_data_2D_prenormalized(
        normalized_csv_path=NORMALIZED_CSV,
        csv_paths=[TEST_CSV],  # ← STRING path, not DataFrame!
        diagnoses=None,  # Load ALL diagnoses for testing
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
    
    metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
    mri_data_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    
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
    
    args = parser.parse_args()
    main(args)