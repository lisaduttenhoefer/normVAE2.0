import argparse
import os
import h5py
import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy import stats as scipy_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore')

"""
Add this function to utils/dev_scores_utils.py

This function handles the complete analysis pipeline for a single volume type.
"""

def analyze_volume_type_separately(
    vtype: str,
    bootstrap_models,
    clinical_data,
    annotations_df,
    roi_names: List[str],
    norm_diagnosis: str,
    device: str,
    base_save_dir: str,
    mri_data_path: str,
    atlas_name,
    metadata_path: str = None,
    custom_colors: dict = None,
    split_CAT: bool = True,
    add_catatonia_subgroups: bool = False
):
    """
    Perform complete deviation analysis for a single volume type.
    
    This function:
    1. Filters features for the specific volume type
    2. Calculates deviation scores
    3. Runs statistical analyses
    4. Creates visualizations
    5. Performs regional deviation analysis
    
    Args:
        vtype: Volume type to analyze ('Vgm', 'G', 'T', etc.)
        bootstrap_models: List of trained VAE models
        clinical_data: Full tensor of clinical data (all volume types)
        annotations_df: Metadata DataFrame with diagnosis, age, sex, etc.
        roi_names: List of all ROI feature names
        norm_diagnosis: Normative diagnosis group (e.g., 'HC')
        device: Computing device ('cuda' or 'cpu')
        base_save_dir: Base directory for saving results
        mri_data_path: Path to original MRI data CSV
        atlas_name: Atlas name(s) used
        metadata_path: Path to extended metadata (for correlations)
        custom_colors: Dict with custom colors for diagnoses
        split_CAT: Whether to keep CAT-SSD and CAT-MDD separate
        add_catatonia_subgroups: Whether to create catatonia subgroups
        
    Returns:
        results_vtype: DataFrame with deviation scores for this volume type
    """
    
    print(f"\n{'='*80}")
    print(f"ANALYZING VOLUME TYPE: {vtype}")
    print(f"{'='*80}\n")
    
    # ============================================================
    # 1. FILTER FEATURES FOR THIS VOLUME TYPE
    # ============================================================
    vtype_indices = [i for i, name in enumerate(roi_names) if name.startswith(f"{vtype}_")]
    vtype_roi_names = [roi_names[i] for i in vtype_indices]
    
    if len(vtype_indices) == 0:
        print(f"[WARNING] No features found for {vtype}, skipping")
        return None
    
    print(f"[INFO] Found {len(vtype_indices)} features for {vtype}")
    print(f"[INFO] Example features: {vtype_roi_names[:3]} ... {vtype_roi_names[-3:]}")
    
    # Extract only the relevant columns
    vtype_data = clinical_data[:, vtype_indices]
    print(f"[INFO] Data shape for {vtype}: {vtype_data.shape}")
    
    # ============================================================
    # 2. CALCULATE DEVIATION SCORES
    # ============================================================
    print(f"[INFO] Calculating deviation scores for {vtype}...")
    
    results_vtype = calculate_deviations(
        normative_models=bootstrap_models,
        data_tensor=vtype_data,
        norm_diagnosis=norm_diagnosis,
        annotations_df=annotations_df,
        device=device,
        roi_names=vtype_roi_names
    )
    
    # ============================================================
    # 3. CREATE OUTPUT DIRECTORY
    # ============================================================
    vtype_save_dir = os.path.join(base_save_dir, f"{vtype}_analysis")
    os.makedirs(vtype_save_dir, exist_ok=True)
    os.makedirs(os.path.join(vtype_save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(vtype_save_dir, "figures", "distributions"), exist_ok=True)
    
    print(f"[INFO] Results will be saved to: {vtype_save_dir}")
    
    # ============================================================
    # 4. STATISTICAL ANALYSIS & VISUALIZATION
    # ============================================================
    atlas_volume_string = f"Volume Type: {vtype}"
    
    # Set default colors if not provided
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A",
            "SSD": "#3E885B",
            "MDD": "#BEDCFE",
            "CAT": "#2F4B26",
            "CAT-SSD": "#A67DB8",
            "CAT-MDD": "#160C28"
        }
    
    print(f"[INFO] Running statistical analysis for {vtype}...")
    
    # Main analysis with plots
    run_analysis_with_options(
        results_vtype, 
        vtype_save_dir, 
        col_jitter=False,
        norm_diagnosis=norm_diagnosis, 
        split_CAT=split_CAT, 
        custom_colors=custom_colors, 
        name=atlas_volume_string
    )
    
    # ============================================================
    # 5. CORRELATION ANALYSIS (if metadata available)
    # ============================================================
    if metadata_path and os.path.exists(metadata_path):
        print(f"[INFO] Running correlation analysis for {vtype}...")
        
        try:
            correlation_matrix, p_matrix, sig_matrix = create_corrected_correlation_heatmap(
                results_df=results_vtype,
                metadata_df=metadata_path,
                save_dir=vtype_save_dir,
                correction_method='fdr_bh',
                alpha=0.05,
                merge_CAT_groups=not split_CAT,  # Consistent with split_CAT
                name=atlas_volume_string
            )
            print(f"[INFO] Correlation analysis complete for {vtype}")
        except Exception as e:
            print(f"[WARNING] Could not complete correlation analysis for {vtype}: {e}")
    
    # ============================================================
    # 6. SAVE DEVIATION SCORES
    # ============================================================
    deviation_scores_path = os.path.join(vtype_save_dir, f"deviation_scores_{vtype}.csv")
    results_vtype.to_csv(deviation_scores_path, index=False)
    print(f"[INFO] Saved {vtype} deviation scores to {deviation_scores_path}")
    
    # ============================================================
    # 7. PLOT DISTRIBUTIONS
    # ============================================================
    print(f"[INFO] Generating distribution plots for {vtype}...")
    
    plot_results = plot_deviation_distributions(
        results_vtype, 
        vtype_save_dir, 
        norm_diagnosis=norm_diagnosis, 
        split_CAT=split_CAT,
        custom_colors=custom_colors,
        name=atlas_volume_string
    )
    
    # Save summary statistics
    deviation_score_summary_df = plot_results.get("deviation_score")
    if deviation_score_summary_df is not None:
        selected_columns_df = deviation_score_summary_df[['Diagnosis', 'mean', 'std']]
        summary_path = os.path.join(vtype_save_dir, f"deviation_score_mean_std_{vtype}.csv")
        selected_columns_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved {vtype} summary statistics to: {summary_path}")
    
    # ============================================================
    # 8. REGIONAL DEVIATION ANALYSIS
    # ============================================================
    print(f"[INFO] Analyzing regional deviations for {vtype}...")
    
    try:
        regional_results = analyze_regional_deviations(
            results_df=results_vtype,
            save_dir=vtype_save_dir,
            clinical_data_path=mri_data_path,
            volume_type=[vtype],  # Only this volume type
            atlas_name=atlas_name,
            roi_names=vtype_roi_names,
            norm_diagnosis=norm_diagnosis,
            name=atlas_volume_string,
            add_catatonia_subgroups=add_catatonia_subgroups,
            metadata_path=metadata_path,
            merge_CAT_groups=not split_CAT
        )
        
        if regional_results is not None and not regional_results.empty:
            regional_path = os.path.join(vtype_save_dir, f"regional_effect_sizes_{vtype}.csv")
            regional_results.to_csv(regional_path, index=False)
            print(f"[INFO] Saved {vtype} regional analysis to: {regional_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not complete regional analysis for {vtype}: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # 9. SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print(f"{vtype} ANALYSIS COMPLETE")
    print(f"Results saved to: {vtype_save_dir}")
    print(f"Key files:")
    print(f"  - Deviation scores: deviation_scores_{vtype}.csv")
    print(f"  - Summary stats: deviation_score_mean_std_{vtype}.csv")
    print(f"  - Figures: figures/distributions/")
    print(f"{'='*80}\n")
    
    return results_vtype


#helper function 1 for dev_score plotting
# Creates a summary table showing statistics for each diagnosis group colored by the color column
#for colored jitter plots 
def create_color_summary_table(data, metric, color_col, diagnoses, save_dir):
    
    summary_stats = []
    for diagnosis in diagnoses:
        diag_data = data[data['Diagnosis_x'] == diagnosis]
        
        # Basic stats for the metric
        metric_stats = {
            'Diagnosis': diagnosis,
            'N': len(diag_data),
            f'{metric}_mean': diag_data[metric].mean(),
            f'{metric}_std': diag_data[metric].std(),
        }
        
        # Handle categorical vs continuous variables for color column
        if diag_data[color_col].dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis', 'Dataset']:
            # For categorical variables, show counts and percentages
            value_counts = diag_data[color_col].value_counts()
            for val, count in value_counts.items():
                metric_stats[f'{color_col}_{val}_count'] = count
                metric_stats[f'{color_col}_{val}_percent'] = (count / len(diag_data)) * 100
        else:
            # For continuous variables, show mean, std, min, max
            metric_stats.update({
                f'{color_col}_mean': diag_data[color_col].mean(),
                f'{color_col}_std': diag_data[color_col].std(),
                f'{color_col}_min': diag_data[color_col].min(),
                f'{color_col}_max': diag_data[color_col].max()
            })
        
        summary_stats.append(metric_stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    return summary_df

def create_colored_jitter_plots(data, metadata_df, metric, summary_df, plot_order, norm_diagnosis,
                               save_dir, color_columns, diagnosis_palette, split_CAT=False, custom_colors=None):
    """Create jitter plots colored by numerical values from specified columns
    
    Args:
        data: results dataset containing the metric and diagnosis information
        metadata_df: Additional dataframe containing metadata columns for coloring (scores etc)
        split_CAT: If True, keep CAT-SSD and CAT-MDD separate. If False, combine as CAT
        custom_colors: Optional dict with custom color mapping for diagnoses
    """
    
    os.makedirs(f"{save_dir}/figures/distributions/colored_by_columns", exist_ok=True)
    
    # Handle CAT splitting option
    data_processed = data.copy()
    if not split_CAT:
        # Combine CAT-SSD and CAT-MDD into CAT
        data_processed.loc[data_processed['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
    
    # Check if we can merge on filename or need to use index
    merged_data = pd.merge(data_processed, metadata_df, on='Filename', how='inner')
    print(f"Merged data on 'Filename' column. Merged data shape: {merged_data.shape}")
    
    if merged_data.empty:
        print("Error: Could not merge data and metadata. Check if they have common identifiers.")
        return
    #changed column names after merging
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex', 'Dataset_x': 'Dataset'})
    
    # Filter color_columns to only include ones that exist in merged_data
    available_color_columns = [col for col in color_columns if col in merged_data.columns]
    
    # Define columns that have complete data (all patients) for all diagnoses vs. limited diagnoses (WhiteCAT & NSS metadata)
    complete_data_columns = ['Age', 'Sex', 'Dataset']  # Assuming these have data for all diagnoses
    limited_data_columns = [col for col in available_color_columns if col not in complete_data_columns]
    
    for color_col in available_color_columns:
        print(f"Creating plot for column: {color_col}")
        
        
        if color_col in complete_data_columns:
            # Use all diagnoses for Age and Sex -> got metadata for all
            current_plot_order = plot_order.copy()
            # Adjust plot order based on CAT splitting
            if not split_CAT and 'CAT-SSD' in current_plot_order and 'CAT-MDD' in current_plot_order:
                current_plot_order = [d for d in current_plot_order if d not in ['CAT-SSD', 'CAT-MDD']]
                if 'CAT' not in current_plot_order:
                    current_plot_order.append('CAT')
            filtered_data = merged_data.copy()
            plot_title_suffix = "All Diagnoses"
        else:
            # Use only CAT-SSD and CAT-MDD for other columns -> got metadata only for WhiteCAT and NSS patients
            if split_CAT:
                current_plot_order = ['CAT-SSD', 'CAT-MDD']
                filtered_data = merged_data[merged_data['Diagnosis_x'].isin(current_plot_order)].copy()
                plot_title_suffix = "CAT-SSD vs CAT-MDD"
            else:
                current_plot_order = ['CAT']
                filtered_data = merged_data[merged_data['Diagnosis_x'] == 'CAT'].copy()
                plot_title_suffix = "CAT Combined"
        
        filtered_data = filtered_data.dropna(subset=[color_col, metric])
        
        if len(filtered_data) == 0:
            print(f"Warning: No data available for {color_col} after removing missing values. Skipping this column.")
            continue
        
    
        plt.figure(figsize=(14, 6))
        color_values = filtered_data[color_col].copy()
        # Handle categorical variables by converting to numeric
        if color_values.dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis', 'Dataset']:
            unique_values = color_values.unique()
            value_to_code = {val: i for i, val in enumerate(unique_values)}
            color_values_numeric = color_values.map(value_to_code)
            if color_col == 'Sex':
                colors = custom_colors.get('Sex', ['#ff69b4', '#4169e1']) if custom_colors else ['#ff69b4', '#4169e1']
                if len(unique_values) == 2:
                    cmap = LinearSegmentedColormap.from_list('sex_colors', colors, N=2)
                else:
                    cmap = plt.cm.Set1
            else:
                cmap = plt.cm.Set1
                
            color_values = color_values_numeric
            categorical_labels = unique_values
            is_categorical = True
        else:
            cmap = plt.cm.viridis
            categorical_labels = None
            is_categorical = False
        
        scatter = plt.scatter(filtered_data[metric],
                            [current_plot_order.index(diag) for diag in filtered_data['Diagnosis_x']],
                            c=color_values,
                            cmap=cmap,
                            s=30,
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5)
       
        y_positions = [current_plot_order.index(diag) for diag in filtered_data['Diagnosis_x']]
        jitter_strength = 0.3
        y_jittered = [y + np.random.uniform(-jitter_strength, jitter_strength) for y in y_positions]
        
        # Clear the previous scatter and create new one with jittered positions
        plt.clf()
        plt.figure(figsize=(14, 6))
        
        scatter = plt.scatter(filtered_data[metric],
                            y_jittered,
                            c=color_values,
                            cmap=cmap,
                            s=30,
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5)
        
        # Add colorbar with appropriate labels
        cbar = plt.colorbar(scatter)
        if is_categorical and categorical_labels is not None:
            cbar.set_ticks(range(len(categorical_labels)))
            cbar.set_ticklabels(categorical_labels)
            cbar.set_label(f'{color_col.replace("_", " ").title()}', rotation=270, labelpad=20)
        else:
            cbar.set_label(f'{color_col.replace("_", " ").title()}', rotation=270, labelpad=20)
        
        plt.yticks(range(len(current_plot_order)), current_plot_order)
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis\nColored by {color_col.replace('_', ' ').title()} ({plot_title_suffix})",
                 fontsize=14, pad=20)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        CAT_suffix = "split" if split_CAT else "combined"
        filename = f"{metric}_jitterplot_colored_by_{color_col}_CAT_{CAT_suffix}.png"
        plt.savefig(f"{save_dir}/figures/distributions/colored_by_columns/{filename}",
                   dpi=300, bbox_inches='tight')
        plt.close()
        create_color_summary_table(filtered_data, metric, color_col, current_plot_order, save_dir)
 
def calculate_deviations(normative_models, data_tensor, norm_diagnosis, annotations_df, device="cuda", roi_names=None):
    """
    Calculate deviation scores using bootstrap models.
    
    CORRECTED VERSION: All normalization is now done RELATIVE TO HC (normative group)
    
    Args:
        normative_models: List of trained VAE models
        data_tensor: Tensor of clinical data (all subjects)
        norm_diagnosis: Normative diagnosis group (e.g., 'HC')
        annotations_df: DataFrame with metadata (Diagnosis, Age, Sex, etc.)
        device: Computing device ('cuda' or 'cpu')
        roi_names: Optional list of ROI names for column naming
    
    Returns:
        results_df: DataFrame with deviation scores normalized relative to HC
    """
    
    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
    # ========== ALIGNMENT CHECK ==========
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # ========== INITIALIZE ARRAYS ==========
    all_recon_errors = np.zeros((total_subjects, total_models))
    all_kl_divs = np.zeros((total_subjects, total_models))
    all_z_scores = np.zeros((total_subjects, data_tensor.shape[1], total_models))
    
    # ========== PROCESS EACH BOOTSTRAP MODEL ==========
    print(f"[INFO] Processing {total_models} bootstrap models...")
    
    for i, model in enumerate(normative_models):
        model.eval()
        model.to(device)
        with torch.no_grad():
            batch_data = data_tensor.to(device)
            recon, mu, log_var = model(batch_data)
            
            # Reconstruction error (MSE per subject)
            recon_error = torch.mean((batch_data - recon) ** 2, dim=1).cpu().numpy()
            all_recon_errors[:, i] = recon_error
            
            # KL divergence (per subject)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()
            all_kl_divs[:, i] = kl_div
            
            # Region-wise squared errors (for later analysis)
            z_scores = ((batch_data - recon) ** 2).cpu().numpy()
            all_z_scores[:, :, i] = z_scores
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    print(f"[INFO] Finished processing models")
    
    # ========== AVERAGE ACROSS BOOTSTRAP MODELS ==========
    mean_recon_error = np.mean(all_recon_errors, axis=1)
    std_recon_error = np.std(all_recon_errors, axis=1)
    mean_kl_div = np.mean(all_kl_divs, axis=1)
    std_kl_div = np.std(all_kl_divs, axis=1)
    
    # Region-wise mean z-scores
    mean_region_z_scores = np.mean(all_z_scores, axis=2)
    
    # ========== CREATE BASE DATAFRAME ==========
    results_df = annotations_df[["Filename", "Diagnosis", "Age", "Sex", "Dataset"]].copy()
    results_df["reconstruction_error"] = mean_recon_error
    results_df["reconstruction_error_std"] = std_recon_error
    results_df["kl_divergence"] = mean_kl_div
    results_df["kl_divergence_std"] = std_kl_div
    
    # ========== ADD REGION-WISE Z-SCORES ==========
    if roi_names is not None and len(roi_names) == mean_region_z_scores.shape[1]:
        print(f"[INFO] Using {len(roi_names)} ROI names for region columns")
        column_names = [f"{name}_z_score" for name in roi_names]
    else:
        if roi_names is not None:
            print(f"[WARNING] ROI names length ({len(roi_names)}) doesn't match features ({mean_region_z_scores.shape[1]})")
            print("[WARNING] Using generic region_X names instead")
        column_names = [f"region_{i}_z_score" for i in range(mean_region_z_scores.shape[1])]
    
    new_columns = pd.DataFrame(mean_region_z_scores, columns=column_names)
    results_df = pd.concat([results_df, new_columns], axis=1)
    
    # ========================================================================
    # CORRECTED NORMALIZATION: RELATIVE TO HC (NORMATIVE GROUP)
    # ========================================================================
    
    print(f"\n[INFO] Normalizing deviation scores relative to {norm_diagnosis}...")
    
    # Identify HC subjects
    hc_mask = annotations_df["Diagnosis"] == norm_diagnosis
    n_hc = hc_mask.sum()
    
    if n_hc == 0:
        print(f"[ERROR] No subjects found with diagnosis '{norm_diagnosis}'!")
        print(f"[ERROR] Available diagnoses: {annotations_df['Diagnosis'].unique()}")
        raise ValueError(f"Normative diagnosis '{norm_diagnosis}' not found in data")
    
    print(f"[INFO] Found {n_hc} {norm_diagnosis} subjects for normalization reference")
    
    # Extract HC statistics
    hc_recon = mean_recon_error[hc_mask]
    hc_kl = mean_kl_div[hc_mask]
    
    recon_mean_hc = np.mean(hc_recon)
    recon_std_hc = np.std(hc_recon)
    kl_mean_hc = np.mean(hc_kl)
    kl_std_hc = np.std(hc_kl)
    
    print(f"[INFO] HC Reconstruction Error: mean={recon_mean_hc:.6f}, std={recon_std_hc:.6f}")
    print(f"[INFO] HC KL Divergence: mean={kl_mean_hc:.6f}, std={kl_std_hc:.6f}")
    
    # ========== METHOD 1: Z-SCORE NORMALIZATION (RELATIVE TO HC) ==========
    # Formula: (x - mean_HC) / std_HC
    # HC subjects will have mean ≈ 0, std ≈ 1
    # Patient groups show their true deviation from HC in standard deviations
    
    z_norm_recon = (mean_recon_error - recon_mean_hc) / (recon_std_hc + 1e-8)
    z_norm_kl = (mean_kl_div - kl_mean_hc) / (kl_std_hc + 1e-8)
    
    # Combined deviation score (Z-score based)
    results_df["deviation_score_zscore"] = (z_norm_recon + z_norm_kl) / 2
    
    print(f"[INFO] Z-Score normalization complete")
    print(f"       HC mean deviation_score_zscore: {results_df[hc_mask]['deviation_score_zscore'].mean():.3f}")
    print(f"       HC std deviation_score_zscore: {results_df[hc_mask]['deviation_score_zscore'].std():.3f}")
    
    # ========== METHOD 2: PERCENTILE-BASED SCORING (RELATIVE TO HC) ==========
    # Each patient's score is their percentile rank within the HC distribution
    # HC subjects will be uniformly distributed between 0-1
    # Patients exceeding HC range will be >1.0
    
    from scipy import stats as scipy_stats
    
    recon_percentiles = np.array([
        scipy_stats.percentileofscore(hc_recon, x, kind='rank') / 100 
        for x in mean_recon_error
    ])
    kl_percentiles = np.array([
        scipy_stats.percentileofscore(hc_kl, x, kind='rank') / 100 
        for x in mean_kl_div
    ])
    
    results_df["deviation_score_percentile"] = (recon_percentiles + kl_percentiles) / 2
    
    print(f"[INFO] Percentile normalization complete")
    print(f"       HC median percentile: {results_df[hc_mask]['deviation_score_percentile'].median():.3f}")
    
    # ========== METHOD 3: ROBUST MIN-MAX (RELATIVE TO HC RANGE) ==========
    # Normalize to [0, 1] based on HC range
    # Values below HC min → 0
    # Values above HC max → 1
    # HC subjects → spread between 0 and 1
    
    # Use percentiles for robustness (5th and 95th)
    min_recon_hc = np.percentile(hc_recon, 5)
    max_recon_hc = np.percentile(hc_recon, 95)
    min_kl_hc = np.percentile(hc_kl, 5)
    max_kl_hc = np.percentile(hc_kl, 95)
    
    print(f"[INFO] HC Recon range (5th-95th percentile): [{min_recon_hc:.6f}, {max_recon_hc:.6f}]")
    print(f"[INFO] HC KL range (5th-95th percentile): [{min_kl_hc:.6f}, {max_kl_hc:.6f}]")
    
    # Clip and normalize
    norm_recon = np.clip(mean_recon_error, min_recon_hc, max_recon_hc)
    norm_recon = (norm_recon - min_recon_hc) / (max_recon_hc - min_recon_hc + 1e-8)
    
    norm_kl = np.clip(mean_kl_div, min_kl_hc, max_kl_hc)
    norm_kl = (norm_kl - min_kl_hc) / (max_kl_hc - min_kl_hc + 1e-8)
    
    # Combined deviation score (Min-Max based) - THIS IS THE MAIN SCORE
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2
    
    print(f"[INFO] Min-Max normalization complete")
    print(f"       HC mean deviation_score: {results_df[hc_mask]['deviation_score'].mean():.3f}")
    print(f"       HC std deviation_score: {results_df[hc_mask]['deviation_score'].std():.3f}")
    
    # ========== SUMMARY STATISTICS ==========
    print(f"\n[INFO] Deviation Score Summary by Diagnosis:")
    print("="*60)
    
    for diagnosis in sorted(results_df["Diagnosis"].unique()):
        diag_mask = results_df["Diagnosis"] == diagnosis
        n = diag_mask.sum()
        
        mean_score = results_df[diag_mask]["deviation_score"].mean()
        std_score = results_df[diag_mask]["deviation_score"].std()
        
        mean_zscore = results_df[diag_mask]["deviation_score_zscore"].mean()
        std_zscore = results_df[diag_mask]["deviation_score_zscore"].std()
        
        print(f"{diagnosis:10s} (n={n:3d}): "
              f"score={mean_score:.3f}±{std_score:.3f}, "
              f"zscore={mean_zscore:.3f}±{std_zscore:.3f}")
    
    print("="*60)
    print(f"[INFO] Deviation calculation complete!\n")
    
    return results_df


"""
PATCH 7: Add to dev_scores_utils.py

Function to create errorbar plots for ALL deviation metrics
"""

def plot_deviation_distributions_all_metrics(results_df, save_dir, norm_diagnosis='HC', custom_colors=None):
    """
    Create errorbar plots for ALL deviation metrics in results_df.
    
    Creates plots for:
    - deviation_score (bootstrap)
    - deviation_recon
    - deviation_kl
    - deviation_latent_aguila
    - deviation_combined
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Default colors
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A",
            "SSD": "#3E885B",
            "MDD": "#BEDCFE",
            "CAT": "#2F4B26",
            "CAT-SSD": "#A67DB8",
            "CAT-MDD": "#160C28"
        }
    
    # Find all deviation columns
    deviation_columns = [col for col in results_df.columns if col.startswith('deviation_')]
    
    # Nice labels
    label_map = {
        'deviation_score': 'Bootstrap Deviation',
        'deviation_recon': 'Reconstruction Error (MSE)',
        'deviation_kl': 'KL Divergence',
        'deviation_latent_aguila': 'Latent Deviation (Aguila)',
        'deviation_combined': 'Combined Deviation'
    }
    
    for dev_col in deviation_columns:
        # Calculate means and SEMs per diagnosis
        summary = results_df.groupby('Diagnosis')[dev_col].agg(['mean', 'sem', 'count'])
        summary = summary.reset_index()
        
        # Sort: HC first, then others
        if norm_diagnosis in summary['Diagnosis'].values:
            hc_row = summary[summary['Diagnosis'] == norm_diagnosis]
            other_rows = summary[summary['Diagnosis'] != norm_diagnosis].sort_values('mean', ascending=False)
            summary = pd.concat([hc_row, other_rows])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(summary))
        
        # Create bars with error bars
        bars = ax.bar(x_pos, summary['mean'], 
                     yerr=summary['sem'],
                     capsize=5,
                     alpha=0.8,
                     color=[custom_colors.get(diag, '#888888') for diag in summary['Diagnosis']],
                     edgecolor='black',
                     linewidth=1.5)
        
        # Labels
        ax.set_xlabel('Diagnosis', fontsize=14, fontweight='bold')
        ylabel = label_map.get(dev_col, dev_col.replace('_', ' ').title())
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{ylabel} by Diagnosis', fontsize=16, fontweight='bold', pad=20)
        
        # X-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['Diagnosis'], fontsize=12, fontweight='bold')
        
        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add sample sizes
        for i, (idx, row) in enumerate(summary.iterrows()):
            ax.text(i, row['mean'] + row['sem'], f"n={int(row['count'])}", 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        metric_name = dev_col.replace('deviation_', '')
        filename = f"{metric_name}_errorbar_CAT_combined.png"
        plt.savefig(f"{save_dir}/figures/distributions/{filename}", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created errorbar plot: {filename}")
# ==================== NEW DEVIATION SCORE FUNCTIONS ====================

def calculate_reconstruction_deviation(model, data, device='cuda'):
    """
    D_MSE - Reconstruction-based deviation (Pinaya method)
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        reconstructed, _, _ = model(data_tensor)
        mse = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
    return mse.cpu().numpy()


def calculate_kl_divergence_deviation(model, data, device='cuda'):
    """
    D_KL - KL Divergence as deviation metric
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        _, mu, logvar = model(data_tensor)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_div.cpu().numpy()


def calculate_latent_deviation_aguila(model, data, hc_latent_stats, device='cuda'):
    """
    D_L - Latent-based deviation (Aguila et al. 2022)
    
    D_L = (1/K) * Σ |μ_kj - μ̄_k| / √(σ²_k + σ²_kj)
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        _, mu, logvar = model(data_tensor)
        
        sigma_kj = torch.exp(0.5 * logvar)
        hc_mean = torch.FloatTensor(hc_latent_stats['mean']).to(device)
        hc_std = torch.FloatTensor(hc_latent_stats['std']).to(device)
        
        numerator = torch.abs(mu - hc_mean)
        denominator = torch.sqrt(hc_std**2 + sigma_kj**2)
        per_dim_deviations = numerator / denominator
        deviation_scores = torch.mean(per_dim_deviations, dim=1)
        
    return deviation_scores.cpu().numpy(), per_dim_deviations.cpu().numpy()


def calculate_combined_deviation(recon_dev, kl_dev, alpha=0.7, beta=0.3):
    """
    D_combined - Weighted combination of reconstruction and KL
    """
    recon_norm = (recon_dev - recon_dev.min()) / (recon_dev.max() - recon_dev.min() + 1e-8)
    kl_norm = (kl_dev - kl_dev.min()) / (kl_dev.max() - kl_dev.min() + 1e-8)
    return alpha * recon_norm + beta * kl_norm


def compute_hc_latent_stats(model, hc_data, device='cuda'):
    """
    Compute HC population statistics in latent space for Aguila method
    """
    model.eval()
    with torch.no_grad():
        hc_tensor = torch.FloatTensor(hc_data).to(device)
        _, mu, _ = model(hc_tensor)
        hc_mean = mu.mean(dim=0)
        hc_std = mu.std(dim=0)
    return {
        'mean': hc_mean.cpu().numpy(),
        'std': hc_std.cpu().numpy()
    }

def plot_all_deviation_metrics_errorbar(results_df, save_dir, norm_diagnosis='HC', 
                                        custom_colors=None, name="Analysis"):
    """
    Create errorbar plots for ALL deviation metrics - BOTH mean AND median.
    Creates 10 plots total (5 metrics × 2 statistics).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    
    # Default colors
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A",
            "SSD": "#3E885B",
            "MDD": "#BEDCFE",
            "CAT": "#2F4B26",
            "CAT-SSD": "#A67DB8",
            "CAT-MDD": "#160C28"
        }
    
    # Find all deviation columns
    deviation_columns = [col for col in results_df.columns if col.startswith('deviation_')]
    
    # Nice labels for plots
    label_map = {
        'deviation_score': 'Bootstrap Deviation Score',
        'deviation_score_recon': 'Reconstruction Error (D_MSE)',
        'deviation_score_kl': 'KL Divergence (D_KL)',
        'deviation_score_latent_aguila': 'Latent Deviation (D_L - Aguila)',
        'deviation_score_combined': 'Combined Deviation Score'
    }
    
    
    # ========== FILTER TO 4 MAIN DIAGNOSES ==========
    # Keep only HC, MDD, SSD, CAT
    keep_diagnoses = ['HC', 'MDD', 'SSD', 'CAT']

    # First merge CAT-SSD and CAT-MDD into CAT
    results_df_filtered = results_df.copy()
    results_df_filtered.loc[results_df_filtered['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'

    # Then filter to only keep the 4 main diagnoses
    results_df_filtered = results_df_filtered[results_df_filtered['Diagnosis'].isin(keep_diagnoses)]

    # Determine diagnosis order from filtered data
    available_diagnoses = results_df_filtered['Diagnosis'].unique()

    
    if norm_diagnosis in available_diagnoses:
        diagnosis_order = [norm_diagnosis] + [d for d in available_diagnoses if d != norm_diagnosis]
    else:
        diagnosis_order = list(available_diagnoses)
    
    # Reverse for bottom-to-top plotting
    diagnosis_order_plot = diagnosis_order[::-1]
    
    print(f"\n[INFO] Creating errorbar plots (mean + median) for {len(deviation_columns)} deviation metrics...")
    print(f"       Total plots to create: {len(deviation_columns) * 2}")
    
    # ========== LOOP OVER BOTH STATISTICS ==========
    for statistic in ['mean', 'median']:
        
        print(f"\n[INFO] Creating {statistic.upper()} plots...")
        
        for dev_col in deviation_columns:
            if dev_col not in results_df_filtered.columns:
                print(f"[WARNING] Column {dev_col} not found, skipping")
                continue
            
            # Calculate summary statistics
            if statistic == 'mean':
                summary_df = (
                    results_df_filtered
                    .groupby("Diagnosis")[dev_col]
                    .agg(['mean', 'std', 'count'])
                    .reset_index()
                )
                summary_df.rename(columns={'mean': 'center'}, inplace=True)
                # Calculate SEM for error bars
                summary_df["error"] = summary_df["std"] / np.sqrt(summary_df["count"])
            else:  # median
                summary_df = (
                    results_df_filtered
                    .groupby("Diagnosis")[dev_col]
                    .agg(['median', 'count'])
                    .reset_index()
                )
                summary_df.rename(columns={'median': 'center'}, inplace=True)
                # Calculate IQR for error bars (Q1 to Q3)
                q1 = results_df_filtered.groupby("Diagnosis")[dev_col].quantile(0.25)
                q3 = results_df_filtered.groupby("Diagnosis")[dev_col].quantile(0.75)
                summary_df["error_low"] = summary_df["center"] - q1.values
                summary_df["error_high"] = q3.values - summary_df["center"]
            
            # Calculate p-values vs norm diagnosis
            if norm_diagnosis in available_diagnoses:
                norm_data = results_df_filtered[results_df_filtered["Diagnosis"] == norm_diagnosis][dev_col].values
                
                p_values = []
                for diagnosis in summary_df["Diagnosis"]:
                    if diagnosis == norm_diagnosis:
                        p_values.append(np.nan)
                    else:
                        diag_data = results_df_filtered[results_df_filtered["Diagnosis"] == diagnosis][dev_col].values
                        if len(diag_data) > 0:
                            _, p_val = scipy_stats.mannwhitneyu(
                                diag_data, norm_data, alternative='two-sided'
                            )
                            p_values.append(p_val)
                        else:
                            p_values.append(np.nan)
                
                summary_df["p_value"] = p_values
            else:
                summary_df["p_value"] = np.nan
            
            # Sort in plot order
            summary_df["Diagnosis"] = pd.Categorical(
                summary_df["Diagnosis"], 
                categories=diagnosis_order_plot, 
                ordered=True
            )
            summary_df = summary_df.sort_values("Diagnosis")
            
            # Create plot
            plt.figure(figsize=(8, 6))
            
            # Errorbar plot
            if statistic == 'mean':
                plt.errorbar(
                    summary_df["center"], 
                    summary_df["Diagnosis"],
                    xerr=summary_df["error"],
                    fmt='s', 
                    color='black', 
                    capsize=5, 
                    markersize=8
                )
            else:  # median with asymmetric error bars
                plt.errorbar(
                    summary_df["center"], 
                    summary_df["Diagnosis"],
                    xerr=[summary_df["error_low"], summary_df["error_high"]],
                    fmt='D',  # Diamond for median
                    color='black', 
                    capsize=5, 
                    markersize=8
                )
            
            # Add colored scatter with p-value coloring
            p_values_for_color = summary_df["p_value"].fillna(0.5)
            scatter = plt.scatter(
                summary_df["center"], 
                summary_df["Diagnosis"],
                c=p_values_for_color, 
                cmap='RdYlBu_r',
                s=100, 
                alpha=0.7, 
                edgecolors='black',
                vmin=0,
                vmax=0.1
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('p-value', rotation=270, labelpad=15)
            
            # Labels and title
            nice_label = label_map.get(dev_col, dev_col.replace('_', ' ').title())
            stat_label = "Mean ± SEM" if statistic == 'mean' else "Median (IQR)"
            plt.title(f"{stat_label} | Norm: {norm_diagnosis}\n{name}", fontsize=14)
            plt.xlabel(f"{nice_label}", fontsize=12)
            plt.ylabel("Diagnosis", fontsize=12)
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save
            metric_name = dev_col.replace('deviation_score_', '').replace('deviation_score', 'score')
            filename = f"{metric_name}_errorbar_{statistic}_CAT_combined.png"
            save_path = f"{save_dir}/figures/distributions/{filename}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Created: {filename}")
    
    print(f"\n[INFO] All {len(deviation_columns) * 2} errorbar plots created!")
    print(f"       - {len(deviation_columns)} MEAN plots (error bars = SEM)")
    print(f"       - {len(deviation_columns)} MEDIAN plots (error bars = IQR)")



def create_paper_style_boxplots(deviation_df, save_dir, norm_diagnosis='HC'):
    """
    Create Figure 3 style boxplots from Aguila et al. paper
    One plot per deviation metric with p-values
    """
    from scipy.stats import mannwhitneyu
    
    dev_columns = [col for col in deviation_df.columns if col.startswith('deviation_')]
    
    label_map = {
        'deviation_recon': '$D_{MSE}$ (Reconstruction)',
        'deviation_kl': '$D_{KL}$ (KL Divergence)',
        'deviation_latent_aguila': '$D_L$ (Latent - Aguila)',
        'deviation_combined': '$D_{Combined}$'
    }
    
    diagnoses = deviation_df['Diagnosis'].unique()
    patient_diagnoses = [d for d in diagnoses if d != norm_diagnosis]
    
    for dev_col in dev_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = []
        plot_labels = []
        
        # HC first
        if norm_diagnosis in diagnoses:
            hc_data = deviation_df[deviation_df['Diagnosis'] == norm_diagnosis][dev_col].dropna()
            plot_data.append(hc_data)
            plot_labels.append(norm_diagnosis)
        
        # Other diagnoses
        for diag in patient_diagnoses:
            diag_data = deviation_df[deviation_df['Diagnosis'] == diag][dev_col].dropna()
            if len(diag_data) > 0:
                plot_data.append(diag_data)
                plot_labels.append(diag)
        
        # Boxplot
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                        showfliers=True, widths=0.6)
        
        # Color HC differently
        colors = ['#3498db'] + ['#e74c3c'] * (len(plot_labels) - 1)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # P-values
        if norm_diagnosis in diagnoses:
            hc_vals = deviation_df[deviation_df['Diagnosis'] == norm_diagnosis][dev_col].dropna()
            y_max = max([d.max() for d in plot_data])
            y_step = (y_max - min([d.min() for d in plot_data])) * 0.1
            
            for i, diag in enumerate(patient_diagnoses, start=2):
                diag_vals = deviation_df[deviation_df['Diagnosis'] == diag][dev_col].dropna()
                if len(diag_vals) > 0:
                    _, p_value = mannwhitneyu(hc_vals, diag_vals, alternative='two-sided')
                    
                    y_pos = y_max + (i-1) * y_step * 0.3
                    ax.plot([1, i], [y_pos, y_pos], 'k-', linewidth=0.8)
                    
                    if p_value < 0.001:
                        p_text = 'p<0.001'
                    elif p_value < 0.01:
                        p_text = 'p<0.01'
                    else:
                        p_text = f'p={p_value:.3f}'
                    
                    ax.text((1 + i) / 2, y_pos * 1.02, p_text, 
                           ha='center', va='bottom', fontsize=9)
        
        # Labels
        nice_label = label_map.get(dev_col, dev_col)
        ax.set_ylabel(nice_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
        ax.set_title(f'{nice_label}: {norm_diagnosis} vs Disease Cohorts', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        metric_name = dev_col.replace('deviation_', '')
        plt.savefig(f"{save_dir}/figures/paper_style_{metric_name}_boxplot.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        log_and_print_test(f"✓ Created paper-style boxplot for {nice_label}")
        
def calculate_group_pvalues(results_df, norm_diagnosis, split_CAT=False):
    #Calculate p-values for each diagnosis group compared to the control group
    

    # Handle CAT splitting
    results_processed = results_df.copy()
    if not split_CAT:
        # Combine CAT-SSD and CAT-MDD into CAT
        results_processed.loc[results_processed['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'

    # Get control group data
    control_mask = results_processed["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"WARNING: No control group '{norm_diagnosis}' found in data. Available diagnoses: {results_processed['Diagnosis'].unique()}")
        # Use bottom 25% as reference if no explicit control group
        control_indices = np.argsort(results_processed["deviation_score_zscore"])[:len(results_processed)//4]
        control_mask = np.zeros(len(results_processed), dtype=bool)
        control_mask[control_indices] = True
        print(f"Using bottom 25% ({control_mask.sum()} subjects) as reference group")
    
    control_data = results_processed[control_mask]
    print(f"Control group ({norm_diagnosis}) size: {len(control_data)}")
    
    # Metrics to test
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    # Calculate p-values for each diagnosis group vs control
    group_pvalues = {}
    
    diagnoses = results_processed["Diagnosis"].unique()
    diagnoses = [d for d in diagnoses if d != norm_diagnosis]  # Exclude control group
    
    for metric in metrics:
        group_pvalues[metric] = {}
        control_values = control_data[metric].values
        
        for diagnosis in diagnoses:
            group_data = results_processed[results_processed["Diagnosis"] == diagnosis]
            if len(group_data) > 0:
                group_values = group_data[metric].values
               
                # Use Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = scipy_stats.mannwhitneyu(
                        group_values, control_values,
                        alternative='two-sided'
                    )
                    print(f"    Mann-Whitney U: statistic={statistic:.2f}, p={p_value:.6f}")
                    
                    # Double-check with t-test for comparison
                    t_stat, t_pval = scipy_stats.ttest_ind(
                        group_values, control_values,
                        equal_var=False
                    )
                    print(f"    T-test (comparison): t={t_stat:.2f}, p={t_pval:.6f}")
                    
                    group_pvalues[metric][diagnosis] = p_value
                except Exception as e:
                    print(f"Error with statistical tests")
    
    return group_pvalues

def create_diagnosis_palette(split_CAT=False, custom_colors=None):
    #Create consistent diagnosis color palette
    
    if custom_colors:
        return custom_colors
    
    # Default color palette
    base_palette = sns.light_palette("blue", n_colors=6, reverse=True)
    
    if split_CAT:
        diagnosis_order = ["HC", "SSD", "MDD", "CAT", "CAT-MDD", "CAT-SSD"]
    else:
        diagnosis_order = ["HC", "SSD", "MDD", "CAT"]
        base_palette = base_palette[:4]  # Use fewer colors when not splitting CAT
    
    diagnosis_palette = dict(zip(diagnosis_order, base_palette))
    
    return diagnosis_palette

def plot_deviation_distributions(results_df, save_dir, col_jitter, norm_diagnosis, name,
                                split_CAT=False, custom_colors=None):
    #Plot distributions of deviation metrics by diagnosis group with group p-values
    
    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Handle CAT splitting
    results_processed = results_df.copy()
    if not split_CAT:
        # Combine CAT-SSD and CAT-MDD into CAT
        results_processed.loc[results_processed['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
    
    # Create color palette
    diagnosis_palette = create_diagnosis_palette(split_CAT, custom_colors)

    # Calculate group p-values
    group_pvalues = calculate_group_pvalues(results_processed, norm_diagnosis, split_CAT)

    # Determine selected diagnoses based on CAT splitting
    if split_CAT:
        selected_diagnoses = ["HC", "SSD", "MDD", "CAT", "CAT-MDD", "CAT-SSD"]
    else:
        selected_diagnoses = ["HC", "SSD", "MDD", "CAT"]

    # Filter to only include diagnoses that exist in the data
    available_diagnoses = [d for d in selected_diagnoses if d in results_processed["Diagnosis"].unique()]

    # Plot reconstruction error distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="reconstruction_error", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Mean Reconstruction Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    CAT_suffix = "split" if split_CAT else "combined"
    plt.savefig(f"{save_dir}/figures/distributions/recon_error_dist_CAT_{CAT_suffix}.png", dpi=300)
    plt.close()
    
    # Plot KL divergence distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Mean KL Divergence", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/kl_div_dist_CAT_{CAT_suffix}.png", dpi=300)
    plt.close()
    
    # Plot combined deviation score distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="deviation_score", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Deviation Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/deviation_score_dist_CAT_{CAT_suffix}.png", dpi=300)
    plt.close()
    
    # Plot violin plots for all metrics
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="reconstruction_error", palette=diagnosis_palette, order=available_diagnoses)
    plt.title("Reconstruction Error by Diagnosis", fontsize=14)
    plt.xlabel("")
    plt.subplot(3, 1, 2)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, 
                   legend=False, order=available_diagnoses)
    plt.title("KL Divergence by Diagnosis", fontsize=14)
    plt.xlabel("")
    plt.subplot(3, 1, 3)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="deviation_score", palette=diagnosis_palette, order=available_diagnoses)
    plt.title("Combined Deviation Score by Diagnosis", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/metrics_violin_plots_CAT_{CAT_suffix}.png", dpi=300)
    plt.close()

    # Calculate summary statistics for errorbar plots
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    summary_dict = {}

    for metric in metrics:
        # Filter data for selected diagnoses
        filtered_data = results_processed[results_processed["Diagnosis"].isin(available_diagnoses)]
        
        summary_df = (
            filtered_data
            .groupby("Diagnosis")[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        
        # Calculate 95 confidence interval
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"])
        
        # Add group p-values
        summary_df["p_value"] = summary_df["Diagnosis"].map(
            lambda d: group_pvalues[metric].get(d, np.nan) if d != norm_diagnosis else np.nan
        )
       
        # Sort in desired order (bottom to top)
        diagnosis_order_plot = available_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], categories=diagnosis_order_plot, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")
        
        summary_dict[metric] = summary_df
        
        # Simple errorbar plot -> Pinaya paper
        plt.figure(figsize=(8, 6))
        
        # Filter only diagnoses that actually have data
        plot_order = [d for d in diagnosis_order_plot if d in filtered_data["Diagnosis"].unique()]
        
        plt.errorbar(summary_df["mean"], summary_df["Diagnosis"],
                    xerr=summary_df["ci95"],
                    fmt='s', color='black', capsize=5, markersize=8)
        
        # Add mean p-value as color coding (like in original)
        summary_df_plot = summary_df[summary_df["Diagnosis"].isin(plot_order)]
        # Use group p-values for coloring
        p_values_for_color = summary_df_plot["p_value"].fillna(0.5)  # Fill NaN with neutral value
        scatter = plt.scatter(summary_df_plot["mean"], summary_df_plot["Diagnosis"],
                            c=p_values_for_color, cmap='RdYlBu_r',
                            s=100, alpha=0.7, edgecolors='black')
        
        plt.title(f"Norm Diagnosis: {norm_diagnosis} \n {name}", fontsize=14)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)    
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_errorbar_CAT_{CAT_suffix}.png", dpi=300)
        plt.close()
        
        # Create jitterplot with p-values and mean values
        plt.figure(figsize=(12, 6))  # Made wider to accommodate value labels
        
        # Use consistent color from palette
        if 'MDD' in diagnosis_palette:
            plot_color = diagnosis_palette['MDD']
        else:
            plot_color = '#4c72b0'  # fallback color
        
        sns.stripplot(data=filtered_data, y="Diagnosis", x=metric,
                    order=plot_order, color=plot_color,
                    size=3, alpha=0.6, jitter=0.3)
        
        # Add errorbars, p-values, and mean values
        for i, diagnosis in enumerate(plot_order):
            diagnosis_data = summary_df[summary_df["Diagnosis"] == diagnosis]
            if len(diagnosis_data) > 0:
                mean_val = diagnosis_data["mean"].iloc[0]
                ci_val = diagnosis_data["ci95"].iloc[0]
                p_val = diagnosis_data["p_value"].iloc[0]
                n_val = diagnosis_data["count"].iloc[0]

                plt.errorbar(mean_val, i, xerr=ci_val, fmt='none',
                            color='black', capsize=4, capthick=1.5,
                            elinewidth=1.5, alpha=0.8)
        
        plt.title(f"{name} (vs {norm_diagnosis})", fontsize=14)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.subplots_adjust(left=0.25)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_jitterplot_with_values_CAT_{CAT_suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

        if col_jitter:
            metadata_df = pd.read_csv('/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv')
            #column names in the metadata df that should be used for coloring
            potential_color_columns = ['Age', 'Sex', 'Dataset',
                                       'GAF_Score', 'PANSS_Positive', 'PANSS_Negative',
                                       'PANSS_General', 'PANSS_Total', 'BPRS_Total', 'NCRS_Motor',
                                       'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total', 'NSS_Motor', 'NSS_Total']

            color_columns = [col for col in potential_color_columns if col in metadata_df.columns]
            print(f"Found {len(color_columns)} columns for coloring: {color_columns}")

            if len(color_columns) == 0:
                print("No color columns found! Please check your column names in the metadata dataframe.")
            else:
                
                create_colored_jitter_plots(
                    data=filtered_data,
                    metadata_df=metadata_df,
                    metric=metric,    
                    summary_df=summary_df,
                    plot_order=plot_order,
                    norm_diagnosis=norm_diagnosis,
                    save_dir=save_dir,
                    color_columns=color_columns,
                    diagnosis_palette=diagnosis_palette,
                    split_CAT=split_CAT,
                    custom_colors=custom_colors
                )

    return summary_dict

def setup_plotting_parameters(split_CAT=False, custom_colors=None):
    #Setup consistent plotting parameters for all functions
   
    
    return {
        'split_CAT': split_CAT,
        'custom_colors': custom_colors,
        'diagnosis_palette': create_diagnosis_palette(split_CAT, custom_colors)
    }

def run_analysis_with_options(results_df, save_dir, col_jitter, norm_diagnosis, name,
                             split_CAT=False, custom_colors=None):
    #Run complete analysis with CAT splitting and color options
    
    print(f"Running analysis with CAT {'split' if split_CAT else 'combined'}")
    if custom_colors:
        print(f"Using custom colors: {custom_colors}")
    
    # Run the main plotting function with new parameters
    summary_dict = plot_deviation_distributions(
        results_df=results_df,
        save_dir=save_dir,
        col_jitter=col_jitter,
        norm_diagnosis=norm_diagnosis,
        split_CAT=split_CAT,
        custom_colors=custom_colors,
        name=name
    )
    
    return summary_dict

def extract_roi_names(h5_file_path, volume_type):
   
    #Extract ROI names from HDF5 file
    roi_names = []
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # different options depending on if the files store ROI names as attributes or as dataset
            if volume_type in f:
                # Get ROI names from dataset attributes if they exist
                if 'roi_names' in f[volume_type].attrs:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f[volume_type].attrs['roi_names']]
                # Get ROI names from specific dataset if it exists
                elif 'roi_names' in f[volume_type]:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f[volume_type]['roi_names'][:]]
                # Try to get indices/keys that correspond to measurements
                elif 'measurements' in f[volume_type]:
                    # Some HDF5 files have indices stored separately
                    if 'indices' in f[volume_type]:
                        roi_names = [str(idx) for idx in f[volume_type]['indices'][:]]
                    else:
                        num_rois = f[volume_type]['measurements'].shape[1]
                        roi_names = [f"ROI_{i+1}" for i in range(num_rois)]
            else:
                # Try to look for ROI names at the root level
                if 'roi_names' in f.attrs:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f.attrs['roi_names']]
                elif 'roi_names' in f:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f['roi_names'][:]]
                # Try to infer from top-level structure
                else:
                    # Sometimes ROIs are stored as separate datasets
                    roi_candidates = [key for key in f.keys() if key != 'metadata']
                    if roi_candidates:
                        roi_names = roi_candidates
    except Exception as e:
        print(f"Error extracting ROI names from {h5_file_path}: {e}")
    
    # If still no ROI names, create generic ones based on atlas name
    if not roi_names:
        from pathlib import Path
        atlas_name = Path(h5_file_path).stem
        # Try to get the number of measurements from the file
        try:
            with h5py.File(h5_file_path, 'r') as f:
                if volume_type in f and 'measurements' in f[volume_type]:
                    num_rois = f[volume_type]['measurements'].shape[1]
                else:
                    num_rois = 100  # Default assumption
                roi_names = [f"{atlas_name}_ROI_{i+1}" for i in range(num_rois)]
        except:
            roi_names = [f"{atlas_name}_ROI_{i+1}" for i in range(100)]  
    return roi_names

def visualize_embeddings_multiple(normative_models, data_tensor, annotations_df, 
                                 columns_to_plot=None, device="cuda", figsize=(12, 10)):
    
    #visualizes the latent space and colores the data depending on given metadata -> X-Cov control 
    #returns dictionary with column names as keys and (figure, plot_df) tuples as values
    
    total_subjects = data_tensor.shape[0]
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        print("Creating properly aligned dataset by extracting common subjects...")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Use first model for visualization
    model = normative_models[0]
    model.eval()
    model.to(device)
    
    all_embeddings = []
    batch_size = 16
    
    data_loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("Generating embeddings...")
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            _, mu, _ = model(batch_data)
            all_embeddings.append(mu.cpu().numpy())
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # UMAP for visualization (only need to do this once)
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Determine which columns in metadata df to plot
    if columns_to_plot is None:
        # Automatically detect categorical columns (excluding purely numerical ones)
        columns_to_plot = []
        for col in annotations_df.columns:
            if annotations_df[col].dtype == 'object' or annotations_df[col].nunique() <= 20:
                columns_to_plot.append(col)
        print(f"Auto-detected columns for visualization: {columns_to_plot}")
    
    # Create visualizations for each column
    results = {}
    
    for col in columns_to_plot:
        if col not in annotations_df.columns:
            print(f"Warning: Column '{col}' not found in annotations_df. Skipping.")
            continue
            
        print(f"Creating visualization for column: {col}")
        
        plot_df = annotations_df[[col]].copy()
        plot_df["umap_1"] = umap_embeddings[:, 0]
        plot_df["umap_2"] = umap_embeddings[:, 1]
        
        plot_df = plot_df.dropna(subset=[col])
        
        plt.figure(figsize=figsize)
        unique_values = plot_df[col].nunique()

        custom_palette = [
            "#125E8A",  # Lapis Lazuli
            "#3E885B",  # Sea Green  
            "#BEDCFE",  # Uranian Blue
            "#2F4B26",  # Cal Poly Green
            "#A67DB8",  # Indian Red 
            "#160C28"   # Dark Purple
        ]
        #continous vs binary color palettes depending on data
        if unique_values <= len(custom_palette):
            palette = custom_palette[:unique_values]  # schneidet auf die Anzahl an Klassen zu
        else:
            palette = sns.color_palette("viridis", n_colors=unique_values)
        
        if plot_df[col].dtype in ['object', 'category'] or unique_values <= 20:
            sns.scatterplot(
                data=plot_df,
                x="umap_1",
                y="umap_2",
                hue=col,
                palette=palette,
                s=40,
                alpha=0.7
            )
        else:
            scatter = plt.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=plot_df[col],
                cmap=palette,
                s=40,
                alpha=0.7
            )
            plt.colorbar(scatter, label=col)
        
        plt.title(f"UMAP Visualization - Colored by {col}", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=13)
        plt.ylabel("UMAP 2", fontsize=13)
        
        if plt.gca().get_legend() is not None:
            plt.legend(title=col, fontsize=10, title_fontsize=11, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        results[col] = (plt.gcf(), plot_df.copy())
        plt.show()
    
    return results


def save_latent_visualizations(results, output_dir, dpi=300):
   
    os.makedirs(output_dir, exist_ok=True)
    
    for col_name, (fig, plot_df) in results.items():
        clean_name = col_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        fig.savefig(
            os.path.join(output_dir, f"umap_{clean_name}.png"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        print(f"Saved visualization for '{col_name}'")


def calculate_cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # For each pair (x_i, y_j):
    #  +1 if x_i > y_j
    #  -1 if x_i < y_j
    #   0 if x_i == y_j
    dominance = np.zeros((len(x), len(y)))
    for i, x_i in enumerate(x):
        dominance[i] = np.sign(x_i - y)
    
    # Calculate Cliff's Delta as the mean of the dominance matrix
    delta = np.mean(dominance)
    
    return delta
def create_catatonia_subgroups(results_df, metadata_df, subgroup_columns, high_low_thresholds):
    #Create subgroups of Catatonia patients based on extended WHiteCAT & NSS metadata
    subgroups = {}
    
    # Get Catatonia patients
    CAT_patients = results_df[results_df["Diagnosis"].str.startswith("CAT")].copy()
    print(f"Found Catatonia diagnoses: {CAT_patients['Diagnosis'].unique()}")
        
    if len(CAT_patients) == 0:
        print("No CAT patients found for subgroup analysis")
        return subgroups
    
    # Merge with metadata
    if 'Filename' in CAT_patients.columns and 'Filename' in metadata_df.columns:
        CAT_with_metadata = CAT_patients.merge(metadata_df, on='Filename', how='left')
    else:
        print("Warning: Could not merge metadata. Check ID column names.")
        return subgroups
    
    # Create subgroups for each specified column
    for col in subgroup_columns:
        if col not in CAT_with_metadata.columns:
            print(f"Warning: Column '{col}' not found in metadata")
            continue
        
        # Remove rows with missing values for this column
        valid_data = CAT_with_metadata.dropna(subset=[col])
        
        if len(valid_data) == 0:
            print(f"Warning: No valid data for column '{col}'")
            continue
        
        # Determine threshold
        if col in high_low_thresholds:
            threshold = high_low_thresholds[col]
        else:
            # Use median as default threshold
            threshold = valid_data[col].median()
            print(f"Using median threshold for {col}: {threshold}")
        
        # Create high and low subgroups
        high_group = valid_data[valid_data[col] >= threshold]
        low_group = valid_data[valid_data[col] < threshold]
        
        if len(high_group) > 0:
            subgroups[f"CAT-high_{col}"] = high_group
            print(f"Created CAT-high_{col} subgroup: n={len(high_group)}")
        
        if len(low_group) > 0:
            subgroups[f"CAT-low_{col}"] = low_group
            print(f"Created CAT-low_{col} subgroup: n={len(low_group)}")
    
    return subgroups

def get_atlas_abbreviations():
    return {
        "cobra": "[C]",
        "lpba40": "[L]",
        "neuromorphometrics": "[N]",
        "Neurom": "[N]",
        "suit": "[S]",
        "SUIT": "[S]",
        "thalamic_nuclei": "[TN]",
        "thalamus": "[T]",
        "aal3": "[A]",
        "AAL3": "[AAL3]",
        "ibsr": "[I]",
        "IBSR": "[I]",
        "schaefer100": "[S100]",
        "Sch100": "[S100]",
        "schaefer200": "[S200]",
        "Sch200": "[S200]",
        "aparc_dk40": "[DK]",
        "DK40": "[DK]",
        "aparc_destrieux": "[DES]",
        "Destrieux": "[DES]",      
    }

def format_roi_name_for_plotting(original_roi_name: str, atlas_name_from_config: str | List[str] = None) -> str:
    """
    Format ROI name for plotting.
    
    NEW FORMAT: [V] RightHippocampus (Neurom)
    
    Args:
        original_roi_name: e.g., "Vgm_Neurom_RightHippocampus"
        atlas_name_from_config: Atlas name(s) from config
    
    Returns:
        Formatted string: "[V] RightHippocampus (Neurom)"
    """
    
    atlas_abbreviations = {
        "cobra": "[C]",
        "lpba40": "[L]",
        "neuromorphometrics": "[N]",
        "Neurom": "[N]",
        "suit": "[S]",
        "SUIT": "[S]",
        "thalamic_nuclei": "[TN]",
        "thalamus": "[T]",
        "aal3": "[A]",
        "AAL3": "[AAL3]",
        "ibsr": "[I]",
        "IBSR": "[I]",
        "schaefer100": "[S100]",
        "Sch100": "[S100]",
        "schaefer200": "[S200]",
        "Sch200": "[S200]",
        "aparc_dk40": "[DK]",
        "DK40": "[DK]",
        "aparc_destrieux": "[DES]",
        "Destrieux": "[DES]",      
    }
    
    # Split the original name
    parts = original_roi_name.split('_')
    
    if len(parts) < 3:
        return original_roi_name
    
    # Extract components
    volume_type = parts[0]           # e.g., "Vgm", "G", "T"
    atlas_prefix = parts[1]          # e.g., "Neurom", "DK40", "lpba40"
    roi_name = "_".join(parts[2:])   # e.g., "RightHippocampus" or "Left_Amygdala"
    
    # Get volume type abbreviation
    if volume_type.startswith('V'):
        vtype_abbr = f"[{volume_type[1:].upper()}]"  # Vgm → [GM]
    else:
        vtype_abbr = f"[{volume_type}]"              # G → [G], T → [T]
    
    # ========== NEW FORMAT ==========
    # [V] RightHippocampus (Neurom)
    return f"{vtype_abbr} {roi_name} ({atlas_prefix})"

def format_roi_names_list_for_plotting(roi_names_list: List[str], atlas_name_from_config: str | List[str] = None) -> List[str]:
    return [format_roi_name_for_plotting(name, atlas_name_from_config) for name in roi_names_list]

def bootstrap_cliffs_delta_ci(data1: np.ndarray, data2: np.ndarray, num_bootstraps: int = 100, ci_level: float = 0.95):
    bootstrapped_deltas = []
    n1 = len(data1)
    n2 = len(data2)

    if n1 < 2 or n2 < 2:
        # Rückgabe von NaN für CI-Grenzen UND p-Wert
        return np.nan, np.nan, np.nan

    # Berechne das originale Cliff's Delta, das wir testen wollen
    original_delta = calculate_cliffs_delta(data1, data2)
    if np.isnan(original_delta):
        return np.nan, np.nan, np.nan

    for _ in range(num_bootstraps):
        sample1 = np.random.choice(data1, n1, replace=True)
        sample2 = np.random.choice(data2, n2, replace=True)
        
        delta = calculate_cliffs_delta(sample1, sample2)
        if not np.isnan(delta): # NaN-Werte aus Bootstraps ignorieren
            bootstrapped_deltas.append(delta)

    if not bootstrapped_deltas: # Falls alle Bootstrap-Deltas NaN waren
        return np.nan, np.nan, np.nan

    sorted_deltas = np.sort(bootstrapped_deltas)

    # Konfidenzintervall Berechnung (wie in Ihrer Originalfunktion)
    lower_bound_idx = int(num_bootstraps * (1 - ci_level) / 2)
    upper_bound_idx = int(num_bootstraps * (1 - (1 - ci_level) / 2))

    # Sicherstellen, dass Indizes nicht außerhalb der Array-Grenzen liegen
    lower_bound = sorted_deltas[lower_bound_idx] if lower_bound_idx < len(sorted_deltas) else np.nan
    upper_bound = sorted_deltas[upper_bound_idx] if upper_bound_idx < len(sorted_deltas) else np.nan
    # Count of bootstrapped deltas that are on the "other side" of 0
    # compared to the original delta, or are exactly 0.
    if original_delta >= 0:
       
        count_extreme = np.sum(np.abs(sorted_deltas) >= np.abs(original_delta))
        p_value = count_extreme / num_bootstraps
        
    else: 
        count_extreme = np.sum(np.abs(sorted_deltas) >= np.abs(original_delta))
        p_value = count_extreme / num_bootstraps
    
    p_value = max(p_value, 1.0 / num_bootstraps) # Minimaler p-Wert ist 1/N_bootstraps
    p_value = min(p_value, 1.0) # Maximaler p-Wert ist 1.0

    return lower_bound, upper_bound, p_value

def analyze_regional_deviations(
        results_df,
        save_dir,
        clinical_data_path, 
        volume_type,
        atlas_name,
        roi_names,
        norm_diagnosis,
        name,
        add_catatonia_subgroups=True,
        metadata_path=None,
        subgroup_columns=None,
        high_low_thresholds=None,
        merge_CAT_groups=True
    ):
    """
    Modified version with:
    - Heatmap 1: Top 30 CAT-affected regions (3 diagnoses)
    - Heatmap 2: Top 30 overall-affected regions (3 diagnoses)
    - Paper-style plots: Top 16 per diagnosis
    """
    
    print("\n[INFO] Starting MODIFIED regional deviation analysis...")
    
    # Import required libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import List
    import os
    
    # Helper functions for ROI naming (same as before)
    def get_atlas_abbreviations_local():
        return {
            "cobra": "[C]",
            "lpba40": "[L]",
            "neuromorphometrics": "[N]",
            "suit": "[S]",
            "thalamic_nuclei": "[TN]",
            "thalamus": "[T]",
        }

    def format_roi_name_for_plotting_local(original_roi_name: str, atlas_name_from_config: str | List[str] = None) -> str:
        """
        Format ROI name for plotting.
        NEW FORMAT: [V] RightHippocampus (Neurom)
        """
        
        atlas_abbreviations = {
            "cobra": "[C]",
            "lpba40": "[L]",
            "neuromorphometrics": "[N]",
            "Neurom": "[N]",
            "suit": "[S]",
            "SUIT": "[S]",
            "thalamic_nuclei": "[TN]",
            "thalamus": "[T]",
            "aal3": "[A]",
            "AAL3": "[AAL3]",
            "ibsr": "[I]",
            "IBSR": "[I]",
            "schaefer100": "[S100]",
            "Sch100": "[S100]",
            "schaefer200": "[S200]",
            "Sch200": "[S200]",
            "aparc_dk40": "[DK]",
            "DK40": "[DK]",
            "aparc_destrieux": "[DES]",
            "Destrieux": "[DES]",
        }
        
        # Split the original name
        parts = original_roi_name.split('_')
        
        if len(parts) < 3:
            # If format is unexpected, return as-is
            return original_roi_name
        
        # Extract components
        volume_type = parts[0]      # e.g., "Vgm", "G", "T"
        atlas_prefix = parts[1]     # e.g., "Neurom", "DK40", "lpba40"
        roi_name = "_".join(parts[2:])  # e.g., "RightHippocampus" or "Left_Amygdala"
        
        # Get volume type abbreviation
        if volume_type.startswith('V'):
            # Vgm → [GM], Vwm → [WM], Vcsf → [CSF]
            vtype_abbr = f"[{volume_type[1:].upper()}]"
        else:
            # G → [G], T → [T]
            vtype_abbr = f"[{volume_type}]"
        
        # ========== NEW FORMAT ==========
        # [V] RightHippocampus (Neurom)
        # NOT: [V] Neurom (RightHippocampus)
        return f"{vtype_abbr} {roi_name} ({atlas_prefix})"

    def format_roi_names_list_for_plotting_local(roi_names_list: List[str], atlas_name_from_config: str | List[str] = None) -> List[str]:
        return [format_roi_name_for_plotting_local(name, atlas_name_from_config) for name in roi_names_list]

    # Bootstrap parameters
    NUM_BOOTSTRAPS = 800
    CI_LEVEL = 0.95

    # Format ROI names
    if roi_names is not None:
        formatted_roi_names_for_plotting = format_roi_names_list_for_plotting_local(roi_names, atlas_name_from_config=atlas_name)
        print(f"[INFO] ROI names formatted for plotting. Example: {formatted_roi_names_for_plotting[0]}")
    else:
        print("[WARNING] No ROI names provided, using generic region_X labels.")
        region_cols_from_df = [col for col in results_df.columns if col.endswith("_z_score")]
        formatted_roi_names_for_plotting = [f"Region_{i+1}" for i in range(len(region_cols_from_df))]

    # Merge CAT groups if requested
    if merge_CAT_groups:
        results_df = results_df.copy()
        results_df.loc[results_df['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
        print("[INFO] Merged CAT-SSD and CAT-MDD into single CAT group")
    
    # ========== FILTER TO 4 MAIN DIAGNOSES ==========
    keep_diagnoses = ['HC', 'MDD', 'SSD', 'CAT']
    results_df = results_df[results_df['Diagnosis'].isin(keep_diagnoses)].copy()
    print(f"[INFO] Filtered to 4 main diagnoses: {keep_diagnoses}")
    print(f"[INFO] Sample sizes after filtering:")
    for diag in keep_diagnoses:
        n = (results_df['Diagnosis'] == diag).sum()
        if n > 0:
            print(f"       {diag}: {n}")

    # Find region columns
    region_cols = [col for col in results_df.columns if col.endswith("_z_score")]
    print(f"[INFO] Found {len(region_cols)} regional z-score columns")

    if len(formatted_roi_names_for_plotting) != len(region_cols):
        print(f"[WARNING] ROI name count mismatch. Using column names directly.")
        roi_mapping_for_internal = {col: col for col in region_cols}
        formatted_roi_names_for_plotting = [col.replace("_z_score", "") for col in region_cols]
    else:
        roi_mapping_for_internal = dict(zip(region_cols, formatted_roi_names_for_plotting))

    named_results_df = results_df.copy()
    named_results_df.rename(columns=roi_mapping_for_internal, inplace=True)

    # Get diagnoses and norm data
    diagnoses = results_df["Diagnosis"].unique()
    norm_data = results_df[results_df["Diagnosis"] == norm_diagnosis]

    if len(norm_data) == 0:
        print(f"[ERROR] No data found for normative diagnosis '{norm_diagnosis}'")
        return pd.DataFrame()

    effect_sizes = []

    # Catatonia subgroups (if requested)
    catatonia_subgroups = {}
    if add_catatonia_subgroups and metadata_path and subgroup_columns:
        try:
            from utils.dev_scores_utils import create_catatonia_subgroups
            metadata_df = pd.read_csv(metadata_path)
            if 'Diagnosis' in metadata_df.columns and merge_CAT_groups:
                metadata_df.loc[metadata_df['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'

            catatonia_subgroups = create_catatonia_subgroups(
                results_df, metadata_df, subgroup_columns, high_low_thresholds
            )
        except Exception as e:
            print(f"[WARNING] Could not create catatonia subgroups: {e}")

    # Helper function to process groups
    def process_group(group_name, group_data):
        nonlocal effect_sizes
        
        if len(group_data) == 0:
            print(f"[WARNING] No data for group: {group_name}")
            return

        print(f"[INFO] Analyzing {group_name} (n={len(group_data)}) vs {norm_diagnosis} (n={len(norm_data)})")

        for i, region_col in enumerate(region_cols):
            roi_name_for_output = formatted_roi_names_for_plotting[i] if i < len(formatted_roi_names_for_plotting) else f"Region_{i+1}"

            group_region_values = group_data[region_col].values
            norm_region_values = norm_data[region_col].values

            if len(group_region_values) == 0 or len(norm_region_values) == 0:
                continue

            group_mean = np.mean(group_region_values)
            group_std = np.std(group_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)

            mean_diff = group_mean - norm_mean
            
            # Import calculate_cliffs_delta and bootstrap_cliffs_delta_ci from utils
            from utils.dev_scores_utils import calculate_cliffs_delta, bootstrap_cliffs_delta_ci
            
            cliff_delta = calculate_cliffs_delta(group_region_values, norm_region_values)

            cliff_delta_ci_low, cliff_delta_ci_high, p_val_from_bootstrap = bootstrap_cliffs_delta_ci(
                group_region_values, norm_region_values, num_bootstraps=NUM_BOOTSTRAPS, ci_level=CI_LEVEL
            )

            is_significant_p05_uncorrected = False
            if not pd.isna(cliff_delta_ci_low) and not pd.isna(cliff_delta_ci_high):
                if (cliff_delta_ci_low > 0) or (cliff_delta_ci_high < 0):
                    is_significant_p05_uncorrected = True

            pooled_std = np.sqrt(((len(group_region_values) - 1) * group_std**2 +
                                (len(norm_region_values) - 1) * norm_std**2) /
                                (len(group_region_values) + len(norm_region_values) - 2))

            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

            effect_sizes.append({
                "Diagnosis": group_name,
                "Vs_Norm_Diagnosis": norm_diagnosis,
                "Region_Column": region_col,
                "ROI_Name": roi_name_for_output,
                "Diagnosis_Mean": group_mean,
                "Diagnosis_Std": group_std,
                "Norm_Mean": norm_mean,
                "Norm_Std": norm_std,
                "Mean_Difference": mean_diff,
                "Cliffs_Delta": cliff_delta,
                "Cliffs_Delta_CI_Low": cliff_delta_ci_low,
                "Cliffs_Delta_CI_High": cliff_delta_ci_high,
                "Significant_Bootstrap_p05_uncorrected": is_significant_p05_uncorrected,
                "Cohens_d": cohens_d,
                "P_Value_Uncorrected": p_val_from_bootstrap
            })

    # Process main diagnoses
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        process_group(diagnosis, dx_data)

    # Process catatonia subgroups
    for subgroup_name, subgroup_data in catatonia_subgroups.items():
        process_group(subgroup_name, subgroup_data)

    if len(effect_sizes) == 0:
        print("[ERROR] No effect sizes calculated")
        return pd.DataFrame()

    effect_sizes_df = pd.DataFrame(effect_sizes)
    effect_sizes_df["Abs_Cliffs_Delta"] = effect_sizes_df["Cliffs_Delta"].abs()
    effect_sizes_df["Abs_Cohens_d"] = effect_sizes_df["Cohens_d"].abs()
    
    # Save effect sizes
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    effect_sizes_df.to_csv(
        f"{save_dir}/effect_sizes_with_bootstrap_ci_and_significance_vs_{norm_diagnosis}.csv", 
        index=False
    )

    # ========================================================================
    # PAPER-STYLE PLOTS - FIXED PLOT AREA SIZE
    # ========================================================================

    print("\n[INFO] Creating paper-style plots (Top 16 per diagnosis)...")

    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue

        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        if dx_effect_sizes.empty:
            continue

        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        top_regions = dx_effect_sizes_sorted.head(16)

        # ========== FIXED: Control plot area size explicitly ==========
        fig = plt.figure(figsize=(10, 10))  # Total canvas size
        
        # Create axes with fixed dimensions for the PLOT AREA
        # [left, bottom, width, height] in figure coordinates (0-1)
        ax = fig.add_axes([0.45, 0.1, 0.50, 0.85])  
        # left=0.45 → Leaves space for long ROI names
        # width=0.50 → FIXED width for error bar area (50% of figure)
        # height=0.85 → FIXED height for plot area

        y_pos = np.arange(len(top_regions))

        for i, (idx, row) in enumerate(top_regions.iterrows()):
            effect = row["Cliffs_Delta"]
            ci_low = row["Cliffs_Delta_CI_Low"]
            ci_high = row["Cliffs_Delta_CI_High"]

            if pd.isna(ci_low) or pd.isna(ci_high):
                continue

            ax.plot([ci_low, ci_high], [i, i], 'k-', linewidth=1.5, alpha=0.8)
            ax.plot(effect, i, 'ko', markersize=4, markerfacecolor='black', markeredgecolor='black')

        # Format ROI names
        formatted_labels = []
        for roi_name in top_regions["ROI_Name"]:
            if '(' in roi_name and ')' in roi_name:
                formatted_labels.append(roi_name)
            else:
                formatted_labels.append(format_roi_name_for_plotting_local(roi_name, atlas_name))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(formatted_labels, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(x=0, color="blue", linestyle="--", linewidth=1, alpha=0.7)

        # Individual x-limits per diagnosis
        valid_ci_rows = top_regions.dropna(subset=["Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High"])
        if not valid_ci_rows.empty:
            min_value = valid_ci_rows["Cliffs_Delta_CI_Low"].min()
            max_value = valid_ci_rows["Cliffs_Delta_CI_High"].max()

            value_range = max_value - min_value
            buffer = value_range * 0.05

            ax.set_xlim(min_value - buffer, max_value + buffer)
        else:
            ax.set_xlim(-1, 1)

        ax.set_xlabel("Effect size", fontsize=10)
        ax.set_title(f"Top 16 Regions {diagnosis} vs. {norm_diagnosis}\n({name})", 
                    fontsize=11, fontweight='bold', pad=10)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(False)

        # NO tight_layout() or bbox_inches='tight' → keeps fixed dimensions
        plt.savefig(f"{save_dir}/figures/paper_style_{diagnosis}_vs_{norm_diagnosis}.png",
                    dpi=300, facecolor='white')  # Removed bbox_inches='tight'
        plt.close()
        
        print(f"  ✓ Created paper-style plot for {diagnosis}")

    # ========================================================================
    # HEATMAP 1 & 2 - FROM OLD SCRIPT
    # ========================================================================
    
    print("\n[INFO] Creating Heatmap 1 & 2...")
    
    # Determine main diagnoses
    if merge_CAT_groups:
        desired_diagnoses = ['MDD', 'SSD', 'CAT']
    else:
        desired_diagnoses = ['MDD', 'SSD', 'CAT-SSD', 'CAT-MDD']
    
    available_diagnoses_for_heatmap = [diag for diag in desired_diagnoses if diag in effect_sizes_df["Diagnosis"].unique()]
    
    # Get CAT top 30 regions
    if merge_CAT_groups:
        CAT_effects = effect_sizes_df[effect_sizes_df["Diagnosis"] == "CAT"]
    else:
        CAT_effects = effect_sizes_df[effect_sizes_df["Diagnosis"].isin(["CAT-SSD", "CAT-MDD"])]

    if not CAT_effects.empty:
        if merge_CAT_groups:
            CAT_top_regions_df = CAT_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)
            CAT_top_regions = CAT_top_regions_df["ROI_Name"].values
            print(f"[INFO] Selected top 30 CAT regions (n={len(CAT_effects)} total)")
        else:
            CAT_region_avg = CAT_effects.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
            CAT_top_regions_df = CAT_region_avg.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)
            CAT_top_regions = CAT_top_regions_df["ROI_Name"].values
            print(f"[INFO] Selected top 30 CAT regions (averaged across CAT-SSD and CAT-MDD)")
    else:
        print("[WARNING] No CAT data found for heatmaps")
        CAT_top_regions = []

    # Get overall top 30 regions
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    overall_top_regions = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
    print(f"[INFO] Selected top 30 overall regions")

    # Prepare heatmap data
    heatmap_data = []
    all_regions_for_heatmap = formatted_roi_names_for_plotting
    all_diagnoses_in_effects = effect_sizes_df["Diagnosis"].unique()
    
    significance_flags_matrix = pd.DataFrame(index=all_regions_for_heatmap, columns=all_diagnoses_in_effects)
    
    for region_formatted_name in all_regions_for_heatmap:
        row = {"ROI_Name": region_formatted_name}
        for diagnosis in all_diagnoses_in_effects:
            if diagnosis == norm_diagnosis:
                row[diagnosis] = np.nan
                significance_flags_matrix.loc[region_formatted_name, diagnosis] = False
                continue

            region_data = effect_sizes_df[(effect_sizes_df["ROI_Name"] == region_formatted_name) &
                                        (effect_sizes_df["Diagnosis"] == diagnosis)]
            if not region_data.empty:
                row[diagnosis] = region_data.iloc[0]["Cliffs_Delta"]
                significance_flags_matrix.loc[region_formatted_name, diagnosis] = region_data.iloc[0]["Significant_Bootstrap_p05_uncorrected"]
            else:
                row[diagnosis] = np.nan
                significance_flags_matrix.loc[region_formatted_name, diagnosis] = False
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index("ROI_Name", inplace=True)
    heatmap_df = heatmap_df.dropna(axis=1, how='all')

    significance_flags_matrix = significance_flags_matrix.loc[heatmap_df.index, heatmap_df.columns]

    def annotate_cell_with_significance(value, is_significant):
        if pd.isna(value):
            return ""
        stars = "*" if is_significant else ""
        return f"{value:.2f}{stars}"

    # ========== HEATMAP 1: Top 30 CAT-affected regions ==========
    print("\n[INFO] Creating Heatmap 1: Top 30 CAT-affected regions (3 diagnoses)...")
    
    if len(available_diagnoses_for_heatmap) > 0 and len(CAT_top_regions) > 0:
        heatmap_CAT_regions_data = heatmap_df.loc[CAT_top_regions, available_diagnoses_for_heatmap].copy()
        significance_CAT_regions = significance_flags_matrix.loc[CAT_top_regions, available_diagnoses_for_heatmap].copy()

        annot_combined_CAT = heatmap_CAT_regions_data.apply(
            lambda col: [annotate_cell_with_significance(val, significance_CAT_regions.loc[idx, col.name])
                        for idx, val in col.items()]
        )
        annot_combined_CAT = pd.DataFrame(annot_combined_CAT.values.tolist(),
                                        index=heatmap_CAT_regions_data.index,
                                        columns=heatmap_CAT_regions_data.columns)

        if not heatmap_CAT_regions_data.empty and not heatmap_CAT_regions_data.isna().all().all():
            fig_width = max(12, len(available_diagnoses_for_heatmap) * 3)
            plt.figure(figsize=(fig_width, 16))

            mask = heatmap_CAT_regions_data.isna()
            sns.heatmap(heatmap_CAT_regions_data, cmap="RdBu_r", center=0,
                    annot=annot_combined_CAT,
                    fmt="",
                    cbar_kws={"label": "Cliff's Delta"}, mask=mask,
                    square=False, linewidths=0.5)

            plt.title(f"Heatmap 1: Top 30 CAT-Affected Regions vs {norm_diagnosis}\n{name}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/heatmap_1_CAT_regions_3diagnoses_vs_{norm_diagnosis}.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            heatmap_CAT_regions_data.to_csv(f"{save_dir}/heatmap_1_CAT_regions_3diagnoses_vs_{norm_diagnosis}.csv")
            print(f"  ✓ Heatmap 1 created: {heatmap_CAT_regions_data.shape[0]} regions, {len(available_diagnoses_for_heatmap)} diagnoses")
        else:
            print("  [WARNING] No data available for Heatmap 1")
    else:
        print("  [WARNING] Cannot create Heatmap 1 - missing diagnoses or CAT regions")

    # ========== HEATMAP 2: Top 30 overall-affected regions ==========
    print("\n[INFO] Creating Heatmap 2: Top 30 overall-affected regions (3 diagnoses)...")
    
    if len(available_diagnoses_for_heatmap) > 0:
        heatmap_overall_regions_data = heatmap_df.loc[overall_top_regions, available_diagnoses_for_heatmap].copy()
        significance_overall_regions = significance_flags_matrix.loc[overall_top_regions, available_diagnoses_for_heatmap].copy()

        annot_combined_overall = heatmap_overall_regions_data.apply(
            lambda col: [annotate_cell_with_significance(val, significance_overall_regions.loc[idx, col.name])
                        for idx, val in col.items()]
        )
        annot_combined_overall = pd.DataFrame(annot_combined_overall.values.tolist(),
                                            index=heatmap_overall_regions_data.index,
                                            columns=heatmap_overall_regions_data.columns)

        if not heatmap_overall_regions_data.empty and not heatmap_overall_regions_data.isna().all().all():
            fig_width = max(12, len(available_diagnoses_for_heatmap) * 3)
            plt.figure(figsize=(fig_width, 16))

            mask = heatmap_overall_regions_data.isna()
            sns.heatmap(heatmap_overall_regions_data, cmap="RdBu_r", center=0,
                    annot=annot_combined_overall,
                    fmt="",
                    cbar_kws={"label": "Cliff's Delta"}, mask=mask,
                    square=False, linewidths=0.5)

            plt.title(f"Heatmap 2: Top 30 Overall-Affected Regions vs {norm_diagnosis}\n{name}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/heatmap_2_overall_regions_3diagnoses_vs_{norm_diagnosis}.png",
                    dpi=300, bbox_inches='tight')
            plt.close()

            heatmap_overall_regions_data.to_csv(f"{save_dir}/heatmap_2_overall_regions_3diagnoses_vs_{norm_diagnosis}.csv")
            print(f"  ✓ Heatmap 2 created: {heatmap_overall_regions_data.shape[0]} regions, {len(available_diagnoses_for_heatmap)} diagnoses")
        else:
            print("  [WARNING] No data available for Heatmap 2")
    else:
        print("  [WARNING] Cannot create Heatmap 2 - missing diagnoses")

    print("\n[INFO] Regional deviation analysis finished.")

    return effect_sizes_df


######################################################## CORRELATION ANALYSIS ################################################################


def create_corrected_correlation_heatmap(results_df, metadata_df, save_dir, name,
                                       correction_method='fdr_bh',
                                       alpha=0.05,
                                       merge_CAT_groups=True):
   
   # Erstellt eine Heatmap mit korrigierten Korrelationen zwischen Deviation Scores 
    
    metadata_df = pd.read_csv(metadata_df)
    # CAT Gruppen zusammenfassen falls gewünscht
    if merge_CAT_groups:
        results_df = results_df.copy()
        metadata_df = metadata_df.copy()
        results_df.loc[results_df['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
        metadata_df.loc[metadata_df['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
        print("CAT-SSD und CAT-MDD zu CAT zusammengefasst")
    
    # Merge der DataFrames
    merged_data = pd.merge(results_df, metadata_df, on='Filename', how='inner')
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex', 'Dataset_x': 'Dataset'})
    
    # Nur Patientengruppen (keine HC)
    patient_data = merged_data[merged_data['Diagnosis_x'] != 'HC']
    
    # Definiere Score-Spalten
    score_columns = ['GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                     'PANSS_General', 'PANSS_Total', 'BPRS_Total', 
                     'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 
                     'NCRS_Total', 'NSS_Motor', 'NSS_Total']
    
    # Filtere verfügbare Scores
    available_scores = [col for col in score_columns if col in patient_data.columns]
    
    # Patientengruppen identifizieren
    diagnoses = [d for d in patient_data['Diagnosis_x'].unique() if d != 'HC']
    
    print(f"Analysiere Korrelationen für {len(diagnoses)} Patientengruppen und {len(available_scores)} Scores")
    print(f"Patientengruppen: {diagnoses}")
    print(f"Verfügbare Scores: {available_scores}")
    
    # Korrelationen berechnen
    correlation_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
    p_value_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
    
    all_p_values = []
    correlation_info = []
    
    for i, diagnosis in enumerate(diagnoses):
        diag_data = patient_data[patient_data['Diagnosis_x'] == diagnosis]
        
        for j, score in enumerate(available_scores):
            valid_data = diag_data[['deviation_score', score]].dropna()
            
            if len(valid_data) >= 3: 
                r, p = pearsonr(valid_data['deviation_score'], valid_data[score])
                correlation_matrix[i, j] = r
                p_value_matrix[i, j] = p
                all_p_values.append(p)
                correlation_info.append((i, j, diagnosis, score, len(valid_data), r, p))
    
    # Multiple Testing Correction
    if len(all_p_values) > 0:
        rejected, corrected_p_values, _, _ = multipletests(
            all_p_values, alpha=alpha, method=correction_method
        )
        
        # Korrigierte p-Werte in Matrix einsetzen
        corrected_p_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
        significance_matrix = np.full((len(diagnoses), len(available_scores)), False)
        
        for idx, (i, j, diagnosis, score, n, r, p) in enumerate(correlation_info):
            corrected_p_matrix[i, j] = corrected_p_values[idx]
            significance_matrix[i, j] = rejected[idx]
    
    # Annotationen erstellen
    annotations = []
    for i in range(len(diagnoses)):
        row_annotations = []
        for j in range(len(available_scores)):
            if np.isnan(correlation_matrix[i, j]):
                row_annotations.append('')
            else:
                r_val = correlation_matrix[i, j]
                p_val = corrected_p_matrix[i, j]
                
                # Signifikanz-Sterne basierend auf korrigierten p-Werten
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < alpha:
                    stars = '*'
                else:
                    stars = ''
                
                annotation = f'{r_val:.2f}{stars}'
                row_annotations.append(annotation)
        annotations.append(row_annotations)
    
    # Heatmap erstellen
    plt.figure(figsize=(16, max(6, len(diagnoses) * 0.8)))
    
    mask = np.isnan(correlation_matrix)
    
    sns.heatmap(correlation_matrix,
                xticklabels=available_scores,
                yticklabels=diagnoses,
                annot=annotations,
                fmt='',
                cmap='RdBu_r',
                center=0,
                mask=mask,
                square=False,
                cbar_kws={'label': 'Pearson Correlation Coefficient'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title(f'Deviation Score Correlations - {name}\n'
              f'({correction_method.upper()} Corrected, α={alpha})\n'
              f'(* p<{alpha}, ** p<0.01, *** p<0.001)', 
              fontsize=14, pad=20)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    filename = f"{save_dir}/figures/patient_correlations_{correction_method}_corrected.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    total_tests = len([p for p in all_p_values if not np.isnan(p)])
    significant_corrected = np.sum(significance_matrix)
    
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"Gesamte Tests: {total_tests}")
    print(f"Signifikante Korrelationen (korrigiert): {significant_corrected}")
    print(f"Korrekturmethode: {correction_method}")
    print(f"Alpha-Level: {alpha}")
    print(f"Heatmap gespeichert: {filename}")
    
    print(f"\n=== SIGNIFIKANTE KORRELATIONEN ===")
    for idx, (i, j, diagnosis, score, n, r, p_orig) in enumerate(correlation_info):
        if significance_matrix[i, j]:
            p_corr = corrected_p_matrix[i, j]
            print(f"{diagnosis} - {score}: r={r:.3f}, p_orig={p_orig:.3f}, p_corr={p_corr:.3f}, n={n}")
    
    return correlation_matrix, corrected_p_matrix, significance_matrix