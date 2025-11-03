
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
    # Default model directory - will be used if not provided via command line
    default_model_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_0.7_neuromorphometrics_20250521_0641"
    # Use command line argument if provided, otherwise use default
    model_dir = args.model_dir if args.model_dir else default_model_dir
    #-----------------------------------------------------------------------------------------------------------------

    # Check if model_dir exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Extract model name from the model directory for consistent naming
    model_name = os.path.basename(model_dir)
    save_dir = f"{args.output_dir}/clinical_deviations_{model_name}_{timestamp}" if args.output_dir else f"/workspace/project/catatonia_VAE-main_bq/analysis/TESTING/deviation_results_{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    # Set up logging
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    # Log start of analysis
    log_and_print_test("Starting deviation analysis for clinical groups")
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # ---------------------- LOAD MODEL CONFIG FROM TRAINING (consistency)  --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"Loaded model configuration from {config_path}")
        
        # Extract relevant parameters from config.csvmodel
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
        learning_rate = int(config_df["LEARNING_RATE"].iloc[0])
        kldiv_loss_weight = int(config_df["KLDIV_LOSS_WEIGHT"].iloc[0])
        recon_loss_weight = int(config_df["RECON_LOSS_WEIGHT"].iloc[0])
        contr_loss_weight = int(config_df["CONTR_LOSS_WEIGHT"].loc[0])

        if volume_type.startswith('[') and volume_type.endswith(']'):
            volume_type = eval(volume_type)  # Convert string representation to list
        elif volume_type.startswith('"[') and volume_type.endswith(']"'):
            volume_type = eval(volume_type.strip('"'))
        if isinstance(volume_type, str):
            volume_type = [volume_type]

        valid_volume_types = eval(config_df["VALID_VOLUME_TYPES"].iloc[0]) if "VALID_VOLUME_TYPES" in config_df.columns else ["Vgm", "Vwm", "Vcsf"]
        metadata_test = config_df["TEST_CSV"].iloc[0] if "TEST_CSV" in config_df.columns else args.clinical_csv
        mri_data_path = config_df["MRI_DATA_PATH"].iloc[0] if "MRI_DATA_PATH" in config_df.columns else None
        atlas_volume_string = f"{'/'.join(atlas_name)} - {'/'.join(volume_type)}"
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
    
    # Set paths for clinical data
    path_to_clinical_data = mri_data_path
    
    # Determine h5_file_path for ROI name extraction before loading subjects
    if atlas_name != "all" and not isinstance(atlas_name, list):
        h5_file_path = os.path.join(path_to_clinical_data, f"{atlas_name}.h5")
    else:
        all_data_paths = get_all_data(directory=path_to_clinical_data, ext="h5")
        if all_data_paths:
            h5_file_path = all_data_paths[0]
        else:
            h5_file_path = None
            log_and_print_test("Warning: No HDF5 files found for ROI name extraction")
    print(metadata_test)

    # Load clinical data 
    subjects_dev, annotations_dev, roi_names = load_mri_data_2D(
        csv_paths=[metadata_test], #already split and saved csf file from training
        data_path=path_to_clinical_data,
        atlas_name=atlas_name,
        diagnoses=["HC", "SCHZ", "CTT", "MDD", "CTT-MDD", "CTT-SCHZ"], #in testing use all instead of only NORM
        hdf5=True,
        train_or_test="test",
        save=True,
        volume_type=volume_type,
        valid_volume_types=valid_volume_types,
    )
    
    clinical_data = extract_measurements(subjects_dev)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    
    # Get input dimension
    input_dim = clinical_data.shape[1]
    log_and_print_test(f"Input dimension: {input_dim}")
    
    # Verify ROI names match the data dimension
    if roi_names is not None and len(roi_names) != input_dim:
        log_and_print_test(f"Warning: Number of ROI names ({len(roi_names)}) does not match input dimension ({input_dim})")
        log_and_print_test("Generating new ROI names to match input dimension")
        roi_names = [f"Region_{i+1}" for i in range(input_dim)]

    # Count subjects by diagnosis
    diagnosis_counts = annotations_dev["Diagnosis"].value_counts()
    log_and_print_test(f"Subject counts by diagnosis:\n{diagnosis_counts}")
    
    # ---------------------- LOAD BOOTSTRAP MODELS (increases robustness)  --------------------------------------------
    log_and_print_test("Loading normative bootstrap models...")
    bootstrap_models = []
    models_dir = os.path.join(model_dir, "models")
    model_files = [f for f in os.listdir(models_dir) if f.startswith("bootstrap_") and f.endswith(".pt")]
    
    if len(model_files) == 0:
        log_and_print_test("No bootstrap models found. Looking for baseline model...")
        if os.path.exists(os.path.join(models_dir, "baseline_model.pt")):
            model_files = ["baseline_model.pt"]
        else:
            raise FileNotFoundError("No models found in the specified directory.")
    
    # Load up to max_models if specified
    if args.max_models > 0:
        model_files = model_files[:args.max_models]
    
    for model_file in model_files:
        # Initialize model architecture
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
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device) 
        model.eval()  # Set model to evaluation mode 
        bootstrap_models.append(model)
        log_and_print_test(f"Loaded model: {model_file}")
    
    log_and_print_test(f"Successfully loaded {len(bootstrap_models)} models")
    
    # ------------------------------- CALCULATION DEV_SCORES  --------------------------------------------
    log_and_print_test("Calculating deviation scores...")
    results_df = calculate_deviations(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        norm_diagnosis=norm_diagnosis,
        annotations_df=annotations_dev,
        device=device
    )
    custom_colors = {
        "HC": "#125E8A",     # Lapis Lazulli
        "SCHZ": "#3E885B",    # Sea Green  
        "MDD": "#BEDCFE",     # Uranian Blue
        "CTT": "#2F4B26",      # Cal Poly Green
        "CTT-SCHZ": "#A67DB8", # Indian Red
        "CTT-MDD": "#160C28"   # Dark Purple
    }

    run_analysis_with_options(results_df, save_dir, col_jitter=False,norm_diagnosis="HC", split_ctt=True, custom_colors=custom_colors, name = atlas_volume_string)
    
    correlation_matrix, p_matrix, sig_matrix = create_corrected_correlation_heatmap(
        results_df=results_df,
        metadata_df='/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv', 
        save_dir=save_dir,
        correction_method='fdr_bh',
        alpha=0.05,
        merge_ctt_groups=False, 
        name = atlas_volume_string
    )

    # Map ROI names to region columns if we have them
    if roi_names is not None:
        region_cols = [col for col in results_df.columns if col.startswith("region_")]
        if len(region_cols) == len(roi_names):
            # Create a copy of results_df with renamed columns
            named_results_df = results_df.copy()
            roi_mapping = dict(zip(region_cols, roi_names))
            named_results_df.rename(columns=roi_mapping, inplace=True)
            
            # Save version with ROI names
            named_results_df.to_csv(f"{save_dir}/deviation_scores_with_roi_names.csv", index=False)
            log_and_print_test(f"Saved deviation scores with ROI names to {save_dir}/deviation_scores_with_roi_names.csv")
    
    # Save deviation scores
    results_df.to_csv(f"{save_dir}/deviation_scores.csv", index=False)
    log_and_print_test(f"Saved deviation scores to {save_dir}/deviation_scores.csv")
    
    log_and_print_test("Generating visualizations...")
    
    plot_results = plot_deviation_distributions(results_df, save_dir, norm_diagnosis=norm_diagnosis, col_jitter=False, name = atlas_volume_string)
    log_and_print_test("Plotted deviation distributions")
    deviation_score_summary_df = plot_results.get("deviation_score")

    if deviation_score_summary_df is not None:
        # Wählen Sie nur die Spalten 'Diagnosis', 'mean' und 'std' aus
        selected_columns_df = deviation_score_summary_df[['Diagnosis', 'mean', 'std']]

        file_path = os.path.join(save_dir, "deviation_score_mean_std.csv")
        selected_columns_df.to_csv(file_path, index=False)
        print(f"\nMittelwerte und Standardabweichungen für 'deviation_score' gespeichert in: {file_path}")
    else:
        print("\nDie Metrik 'deviation_score' wurde im plot_results-Dictionary nicht gefunden.")

    print("\nSkriptausführung abgeschlossen.")
    # Visualize embeddings in latent space
    log_and_print_test("Visualizing latent space embeddings...")
    
    results = visualize_embeddings_multiple(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        annotations_df=annotations_dev,
        device=device,
        columns_to_plot=["Co_Diagnosis", "Dataset", "Diagnosis", "Sex"]
    )

    save_latent_visualizations(results, output_dir=f"{save_dir}/figures/latent_embeddings")
    
    log_and_print_test("Saved latent space visualizations")
    
    log_and_print_test("Running statistical tests between groups...")
    
    ##------------------------------------------------ROI-WISE DEVIATION SCORES --------------------------------------------------
    log_and_print_test("Analyzing regional deviations...")
    
    if h5_file_path and os.path.exists(h5_file_path):

        regional_results = analyze_regional_deviations(
            results_df=results_df,
            save_dir=save_dir,
            clinical_data_path=h5_file_path,
            volume_type=volume_type,
            atlas_name=atlas_name,
            name= atlas_volume_string,
            roi_names=roi_names,
            norm_diagnosis=norm_diagnosis,
            add_catatonia_subgroups=True,
            merge_ctt_groups=False,
            metadata_path='/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv',
            subgroup_columns=['GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                                    'PANSS_General', 'PANSS_Total', 'BPRS_Total', 'NCRS_Motor', 
                                    'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total', 'NSS_Motor', 'NSS_Total'],
            high_low_thresholds={'GAF_Score': 50, #the higher the better (1-100)
                                'PANSS_Positive': 20, #(minimum score = 7, maximum score = 49)
                                'PANSS_Negative': 20, #(minimum score = 7, maximum score = 49)
                                'PANSS_General': 40, #(minimum score = 16, maximum score = 112)
                                'PANSS_Total':80, #minimum = 30, maximum = 210
                                'BPRS_Total':30, #minimum = 18, maximum = 126
                                'NCRS_Motor':3, #minimum = 0, maximum = 26 ?? geht bei uns nur bis 5
                                'NCRS_Affective':3, #minimum = 0, maximum = 24 ?? geht bei uns nur bis 5
                                'NCRS_Behavioral':3, #minimum = 0, maximum = 30 ?? geht bei uns nur bis 5
                                'NCRS_Total':10, #minimum = 0, maximum = 24
                                'NSS_Motor':10, #minimum = 0, maximum = ?
                                'NSS_Total':25} #minimum = 0, maximum = 19 ??geht bei uns höher?
        )
       
        # Save heatmap data
        regional_results.to_csv(f"{save_dir}/effect_sizes_{norm_diagnosis}.csv")
        
        log_and_print_test(f"Regional deviation analysis complete. Results saved to {save_dir}/regional_effect_sizes.csv")
    else:
        log_and_print_test("Warning: Could not analyze regional deviations due to missing HDF5 file.")
    
        log_and_print_test(f"Regional deviation analysis complete. Results saved to {save_dir}/regional_effect_sizes.csv")
    
    log_and_print_test(f"Deviation analysis complete! Results saved to {save_dir}")

    #-----------------------------------------------------------------------------------------------------------------------------

    log_and_print_test(f"Deviation analysis complete! Results saved to {save_dir}")
    end_logging(Config_2D)

    return save_dir 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate deviation scores for clinical groups.")
    parser.add_argument("--model_dir", help="Path to model directory (default: uses predefined path in code)")
    parser.add_argument("--clinical_data_path", help="Path to clinical data (default: uses path from model config)")
    parser.add_argument("--clinical_csv", help="Path to clinical CSV file")
    parser.add_argument("--norm_diagnosis", type=str, default="HC", help="Normative diagnosis (default: HC)")
    parser.add_argument("--atlas_name", help="Atlas name (if not available in config)")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension (if not available in config)")
    parser.add_argument("--max_models", type=int, default=0, help="Maximum number of models to use (0 = all)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)