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
    load_mri_data_2D,
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
    subjects, annotations_dev, roi_names = load_mri_data_2D(
        data_path=path_to_clinical_data,
        atlas_name=atlas_name,
        csv_paths=[metadata_test],
        annotations=None,
        diagnoses=None,
        train_or_test="test",
        volume_type=volume_type,
        valid_volume_types=valid_volume_types,
        use_tiv_normalization=True,
        normalization_method="columnwise"  # CRITICAL: Use column-wise for combined analysis
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
        # Calculate deviation scores using ALL features
        log_and_print_test("Calculating deviation scores for all subjects...")
        
        results_df = calculate_deviations(
            clinical_data=clinical_data,
            normative_models=bootstrap_models,
            annotations_df=annotations_dev,
            roi_names=roi_names,
            norm_diagnosis=norm_diagnosis,
            device=device
        )
        
        log_and_print_test(f"Calculated deviation scores for {len(results_df)} subjects")
        
        # Save deviation scores
        output_file = os.path.join(save_dir, "deviation_scores_combined.csv")
        results_df.to_csv(output_file, index=False)
        log_and_print_test(f"Saved deviation scores to: {output_file}")
        
        # Calculate mean and std per diagnosis
        summary_stats = results_df.groupby('Diagnosis')['deviation_score'].agg(['mean', 'std', 'count'])
        summary_file = os.path.join(save_dir, "deviation_score_summary.csv")
        summary_stats.to_csv(summary_file)
        log_and_print_test(f"Saved summary statistics to: {summary_file}")
        
        log_and_print_test("\nDeviation Score Summary:")
        log_and_print_test(summary_stats.to_string())
        
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
        fig_dist = plot_deviation_distributions(
            results_df=results_df,
            norm_diagnosis=norm_diagnosis,
            custom_colors=custom_colors
        )
        
        dist_file = os.path.join(save_dir, "figures", "distributions", "deviation_score_distributions.png")
        fig_dist.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close(fig_dist)
        log_and_print_test(f"Saved distribution plot: {dist_file}")
        
        # ==================================================================================
        # REGIONAL ANALYSIS
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("REGIONAL DEVIATION ANALYSIS")
        log_and_print_test("="*80 + "\n")
        
        regional_results = analyze_regional_deviations(
            clinical_data=clinical_data,
            normative_models=bootstrap_models,
            annotations_df=annotations_dev,
            roi_names=roi_names,
            norm_diagnosis=norm_diagnosis,
            device=device
        )
        
        # Save regional effect sizes
        regional_file = os.path.join(save_dir, "regional_effect_sizes_combined.csv")
        regional_results.to_csv(regional_file, index=False)
        log_and_print_test(f"Saved regional effect sizes to: {regional_file}")
        
        # Show top 20 ROIs with largest deviations
        log_and_print_test("\nTop 20 ROIs with largest deviations:")
        top_rois = regional_results.nlargest(20, 'mean_deviation')
        log_and_print_test(top_rois[['roi_name', 'mean_deviation', 'volume_type']].to_string(index=False))
        
        # ==================================================================================
        # CORRELATION WITH CLINICAL VARIABLES
        # ==================================================================================
        
        log_and_print_test("\n" + "="*80)
        log_and_print_test("CORRELATION WITH CLINICAL VARIABLES")
        log_and_print_test("="*80 + "\n")
        
        # Merge results with full metadata if available
        if 'Age' in annotations_dev.columns:
            # Create correlation heatmap
            fig_corr = create_corrected_correlation_heatmap(
                results_df=results_df,
                clinical_vars=['Age'],  # Add more variables as needed
                save_path=os.path.join(save_dir, "figures", "clinical_correlations.png")
            )
            log_and_print_test("Saved clinical correlation heatmap")
        
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
    
    # ==================================================================================
    # FINAL SUMMARY
    # ==================================================================================
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("TESTING COMPLETE - SUMMARY")
    log_and_print_test("="*80)
    log_and_print_test(f"Main output directory: {save_dir}")
    log_and_print_test(f"\nKey outputs:")
    log_and_print_test(f"  - deviation_scores_combined.csv")
    log_and_print_test(f"  - deviation_score_summary.csv")
    log_and_print_test(f"  - regional_effect_sizes_combined.csv")
    log_and_print_test(f"  - figures/distributions/deviation_score_distributions.png")
    log_and_print_test(f"  - figures/latent_embeddings/")
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