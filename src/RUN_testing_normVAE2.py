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

from models.ContrastVAE_2D import NormativeVAE_2D
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
from utils.dev_scores_utils import (
    calculate_deviations, 
    plot_deviation_distributions, 
    analyze_regional_deviations,
    calculate_reconstruction_deviation,
    calculate_kl_divergence_deviation,
    calculate_latent_deviation_aguila,
    calculate_combined_deviation,
    compute_hc_latent_stats,
    save_latent_visualizations,
    visualize_embeddings_multiple,
    create_corrected_correlation_heatmap,
    run_analysis_with_options
)

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
    save_dir = f"{args.output_dir}/clinical_deviations_{model_name}_{timestamp}" if args.output_dir else f"/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/catatonia_VAE-main_v2/analysis/TESTING/deviation_results_{model_name}_{timestamp}"
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
        metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
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
    NORMALIZED_CSV = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC.csv"

    TEST_CSV = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv

    subjects_dev, annotations_dev, roi_names = load_mri_data_2D_prenormalized(
        normalized_csv_path=NORMALIZED_CSV,  # ← Pre-normalized!
        csv_paths=[TEST_CSV],
        diagnoses=None,  # All diagnoses for testing
        covars=[]
    )
    
    # Extract clinical data tensor
    clinical_data = extract_measurements(subjects)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"Clinical data has {clinical_data.shape[1]} features")
    
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
    
    # Get input_dim from the first layer's weight shape
    input_dim = checkpoint['encoder.0.weight'].shape[1]
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
    
    # Load all bootstrap models
    bootstrap_models = []
    for model_file in model_files:
        # Initialize model with same architecture as training
        model = NormativeVAE_2D(
            input_dim=input_dim,
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
        
        # Load model weights
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device) 
        model.eval()  # Set model to evaluation mode 
        bootstrap_models.append(model)
        log_and_print_test(f"Loaded model: {model_file}")
    
    log_and_print_test(f"Successfully loaded {len(bootstrap_models)} models")
    
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
        "SCHZ": "#3E885B",
        "MDD": "#BEDCFE",
        "CTT": "#2F4B26",
        "CTT-SCHZ": "#A67DB8",
        "CTT-MDD": "#160C28"
    }
    
    try:
        # ==================== CALCULATE MULTIPLE DEVIATION SCORES ====================
        log_and_print_test("\n" + "="*80)
        log_and_print_test("Computing MULTIPLE deviation score methods...")
        log_and_print_test("="*80)
        
        # Original bootstrap method (keep this!)
        log_and_print_test("\n[1/5] Calculating BOOTSTRAP deviation scores (original method)...")
        results_df = calculate_deviations(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            norm_diagnosis=norm_diagnosis,
            annotations_df=annotations_dev,
            device=device,
            roi_names=roi_names
        )
        
        # Use baseline model for additional metrics
        baseline_model_path = os.path.join(models_dir, "../baseline_model.pt")
        if os.path.exists(baseline_model_path):
            baseline_model = NormativeVAE_2D(
                input_dim=input_dim,
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
            log_and_print_test("✓ Loaded baseline model for additional metrics")
        else:
            log_and_print_test("⚠ Baseline model not found, using first bootstrap model")
            baseline_model = bootstrap_models[0]
        
        # Separate HC data for latent stats
        hc_mask = annotations_dev['Diagnosis'] == norm_diagnosis
        hc_data = clinical_data[hc_mask]
        log_and_print_test(f"   HC subjects in test set: {len(hc_data)}")
        
        # METHOD 1: Reconstruction Deviation
        log_and_print_test("\n[2/5] Computing D_MSE (Reconstruction-based)...")
        deviation_recon = calculate_reconstruction_deviation(
            model=baseline_model,
            data=clinical_data,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_recon.min():.4f}, {deviation_recon.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_recon[hc_mask].mean():.4f} ± {deviation_recon[hc_mask].std():.4f}")
        
        # METHOD 2: KL Divergence
        log_and_print_test("\n[3/5] Computing D_KL (KL Divergence)...")
        deviation_kl = calculate_kl_divergence_deviation(
            model=baseline_model,
            data=clinical_data,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_kl.min():.4f}, {deviation_kl.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_kl[hc_mask].mean():.4f} ± {deviation_kl[hc_mask].std():.4f}")
        
        # METHOD 3: Latent Deviation (Aguila)
        log_and_print_test("\n[4/5] Computing D_L (Latent-based, Aguila method)...")
        hc_latent_stats = compute_hc_latent_stats(
            model=baseline_model,
            hc_data=hc_data,
            device=device
        )
        log_and_print_test(f"   HC latent mean: {hc_latent_stats['mean'].mean():.4f}")
        log_and_print_test(f"   HC latent std: {hc_latent_stats['std'].mean():.4f}")
        
        deviation_latent_aguila, per_dim_dev = calculate_latent_deviation_aguila(
            model=baseline_model,
            data=clinical_data,
            hc_latent_stats=hc_latent_stats,
            device=device
        )
        log_and_print_test(f"   Range: [{deviation_latent_aguila.min():.4f}, {deviation_latent_aguila.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_latent_aguila[hc_mask].mean():.4f} ± {deviation_latent_aguila[hc_mask].std():.4f}")
        
        # METHOD 4: Combined
        log_and_print_test("\n[5/5] Computing D_Combined (Weighted combination)...")
        deviation_combined = calculate_combined_deviation(
            recon_dev=deviation_recon,
            kl_dev=deviation_kl,
            alpha=0.7,
            beta=0.3
        )
        log_and_print_test(f"   Range: [{deviation_combined.min():.4f}, {deviation_combined.max():.4f}]")
        log_and_print_test(f"   HC mean: {deviation_combined[hc_mask].mean():.4f} ± {deviation_combined[hc_mask].std():.4f}")
        
        # ADD NEW METRICS TO RESULTS_DF
        log_and_print_test("\n" + "="*80)
        log_and_print_test("Adding new deviation metrics to results...")
        results_df['deviation_recon'] = deviation_recon
        results_df['deviation_kl'] = deviation_kl
        results_df['deviation_latent_aguila'] = deviation_latent_aguila
        results_df['deviation_combined'] = deviation_combined
        log_and_print_test("✓ All deviation metrics computed and added to results!")
        
        log_and_print_test(f"Calculated deviation scores for {len(results_df)} subjects")
        
        # Save deviation scores
        output_file = os.path.join(save_dir, "deviation_scores_combined.csv")
        results_df.to_csv(output_file, index=False)
        log_and_print_test(f"Saved deviation scores to: {output_file}")
        
        # Calculate mean and std per diagnosis for ALL metrics
        log_and_print_test("\n" + "="*80)
        log_and_print_test("SUMMARY: Mean deviation scores by diagnosis")
        log_and_print_test("="*80)
        
        # Original bootstrap deviation
        summary_stats = results_df.groupby('Diagnosis')['deviation_score'].agg(['mean', 'std', 'count'])
        summary_file = os.path.join(save_dir, "deviation_score_summary.csv")
        summary_stats.to_csv(summary_file)
        log_and_print_test(f"\nBootstrap Deviation Score (deviation_score):")
        log_and_print_test(summary_stats.to_string())
        
        # Summary for each new metric
        for metric in ['deviation_recon', 'deviation_kl', 'deviation_latent_aguila', 'deviation_combined']:
            if metric in results_df.columns:
                log_and_print_test(f"\n{metric}:")
                for diag in results_df['Diagnosis'].unique():
                    diag_vals = results_df[results_df['Diagnosis'] == diag][metric]
                    log_and_print_test(f"  {diag:8s}: {diag_vals.mean():8.4f} ± {diag_vals.std():8.4f} (n={len(diag_vals)})")
        
        log_and_print_test("="*80)
        
        # ==================== ERRORBAR PLOTS FOR ALL METRICS ====================
        log_and_print_test("\n" + "="*80)
        log_and_print_test("Creating errorbar plots for ALL deviation metrics...")
        log_and_print_test("="*80)
        
        deviation_columns = [col for col in results_df.columns if col.startswith('deviation_')]
        
        label_map = {
            'deviation_score': 'Bootstrap Deviation',
            'deviation_recon': 'Reconstruction Error (MSE)',
            'deviation_kl': 'KL Divergence',
            'deviation_latent_aguila': 'Latent Deviation (Aguila)',
            'deviation_combined': 'Combined Deviation'
        }
        
        for dev_col in deviation_columns:
            summary = results_df.groupby('Diagnosis')[dev_col].agg(['mean', 'sem', 'count'])
            summary = summary.reset_index()
            
            if norm_diagnosis in summary['Diagnosis'].values:
                hc_row = summary[summary['Diagnosis'] == norm_diagnosis]
                other_rows = summary[summary['Diagnosis'] != norm_diagnosis].sort_values('mean', ascending=False)
                summary = pd.concat([hc_row, other_rows])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(len(summary))
            
            bars = ax.bar(x_pos, summary['mean'], 
                         yerr=summary['sem'],
                         capsize=5,
                         alpha=0.8,
                         color=[custom_colors.get(diag, '#888888') for diag in summary['Diagnosis']],
                         edgecolor='black',
                         linewidth=1.5)
            
            ax.set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
            ylabel = label_map.get(dev_col, dev_col.replace('_', ' ').title())
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
            ax.set_title(f'{ylabel} by Diagnosis', fontsize=16, fontweight='bold', pad=20)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(summary['Diagnosis'], fontsize=12, fontweight='bold')
            
            ax.yaxis.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            for i, (idx, row) in enumerate(summary.iterrows()):
                ax.text(i, row['mean'] + row['sem'], f"n={int(row['count'])}", 
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            metric_name = dev_col.replace('deviation_', '')
            filename = f"{metric_name}_errorbar_ctt_combined.png"
            plt.savefig(f"{save_dir}/figures/distributions/{filename}", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            log_and_print_test(f"✓ Created errorbar plot: {filename}")
        
        log_and_print_test("✓ All errorbar plots created!")
        log_and_print_test("="*80)
        
        # ==================================================================================
        # STATISTICAL TESTS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("STATISTICAL TESTS (Mann-Whitney U)")
        log_and_print_test("="*80 + "\n")
        
        # Get HC scores
        hc_scores = results_df[results_df['Diagnosis'] == norm_diagnosis]['deviation_score'].values
        
        # Test each diagnosis vs HC
        for diagnosis in results_df['Diagnosis'].unique():
            if diagnosis == norm_diagnosis:
                continue
            
            diag_scores = results_df[results_df['Diagnosis'] == diagnosis]['deviation_score'].values
            
            if len(diag_scores) > 0:
                u_stat, p_value = stats.mannwhitneyu(hc_scores, diag_scores, alternative='two-sided')
                log_and_print_test(f"{diagnosis} vs {norm_diagnosis}: U={u_stat:.2f}, p={p_value:.4f}")
        
        # ==================================================================================
        # VISUALIZATIONS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CREATING VISUALIZATIONS")
        log_and_print_test("="*80 + "\n")
        
        # Plot deviation score distributions
        plot_results = plot_deviation_distributions(
            results_df=results_df,
            save_dir=save_dir,
            norm_diagnosis=norm_diagnosis,
            col_jitter=False,
            name="combined_analysis"
        )
        log_and_print_test("Plotted deviation distributions")
        
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
            merge_ctt_groups=True
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
        
    # except Exception as e:
    #     log_and_print_test(f"ERROR in combined analysis: {e}")
    #     import traceback
    #     log_and_print_test(traceback.format_exc())
    
    # ==================================================================================
    # LATENT SPACE VISUALIZATION (ALL FEATURES)
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("Visualizing latent space embeddings...")
    log_and_print_test("="*80 + "\n")
    
    try:
        results = visualize_embeddings_multiple(
            normative_models=bootstrap_models,
            data_tensor=clinical_data,
            annotations_df=annotations_dev,
            device=device,
            columns_to_plot=["Diagnosis", "Dataset", "Sex"]  # Adjusted columns
        )
        
        save_latent_visualizations(results, output_dir=f"{save_dir}/figures/latent_embeddings")
        log_and_print_test("Saved latent space visualizations")
        
    except Exception as e:
        log_and_print_test(f"Warning: Could not complete latent space visualization: {e}")
    
    # NOTE: Paper-style plots (regional effect sizes) are created in 
    # analyze_regional_deviations() and saved as:
    #   - figures/paper_style_SCHZ_vs_HC.png
    #   - figures/paper_style_MDD_vs_HC.png
    #   - figures/paper_style_CTT_vs_HC.png
    
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
    log_and_print_test(f"    - score_errorbar_ctt_combined.png (Bootstrap)")
    log_and_print_test(f"    - recon_errorbar_ctt_combined.png (Reconstruction)")
    log_and_print_test(f"    - kl_errorbar_ctt_combined.png (KL Divergence)")
    log_and_print_test(f"    - latent_aguila_errorbar_ctt_combined.png (Aguila method)")
    log_and_print_test(f"    - combined_errorbar_ctt_combined.png (Combined)")
    log_and_print_test(f"  Regional effect sizes (figures/):")
    log_and_print_test(f"    - paper_style_SCHZ_vs_HC.png")
    log_and_print_test(f"    - paper_style_MDD_vs_HC.png")
    log_and_print_test(f"    - paper_style_CTT_vs_HC.png")
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