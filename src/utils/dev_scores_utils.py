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

# ============================================================================
# IMPROVED: P-VALUE CALCULATION WITH MULTIPLE TESTING CORRECTION
# ============================================================================

def calculate_group_pvalues_corrected(results_df, norm_diagnosis, correction='fdr_bh'):
    """Calculate p-values WITH multiple testing correction."""
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import mannwhitneyu
    import numpy as np
    
    print(f"\n[INFO] Calculating p-values with {correction.upper()} correction...")
    
    # ========== FIXED: Use SAME column names ==========
    metrics_map = {
        'reconstruction_error': 'deviation_score_recon',
        'kl_divergence': 'deviation_score_kl',
        'deviation_score': 'deviation_score'
    }
    
    # Get control group data
    control_mask = results_df["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"[ERROR] No control group '{norm_diagnosis}' found!")
        return {}
    
    control_data = results_df[control_mask]
    print(f"[INFO] Reference group: {norm_diagnosis} (n={len(control_data)})")
    
    # Get all diagnoses (exclude norm)
    diagnoses = [d for d in results_df["Diagnosis"].unique() if d != norm_diagnosis]
    
    # Collect ALL p-values first
    all_tests = []
    
    for display_name, metric_col in metrics_map.items():
        
        # Skip if column doesn't exist
        if metric_col not in results_df.columns:
            print(f"[WARNING] Column {metric_col} not found, skipping {display_name}")
            continue
        
        control_values = control_data[metric_col].values
        
        for diagnosis in diagnoses:
            group_data = results_df[results_df["Diagnosis"] == diagnosis]
            
            if len(group_data) > 0:
                group_values = group_data[metric_col].values
                
                try:
                    _, p_value = mannwhitneyu(
                        group_values, control_values,
                        alternative='two-sided'
                    )
                    
                    all_tests.append({
                        'metric': display_name,  # ← Store display name for lookup
                        'diagnosis': diagnosis,
                        'p_value': p_value,
                        'n_group': len(group_values),
                        'n_control': len(control_values)
                    })
                    
                except Exception as e:
                    print(f"[WARNING] Test failed for {diagnosis}, {metric_col}: {e}")
    
    # Apply multiple testing correction
    if len(all_tests) == 0:
        print("[ERROR] No tests to correct!")
        return {}
    
    p_values = [t['p_value'] for t in all_tests]
    rejected, p_corrected, _, _ = multipletests(
        p_values, 
        alpha=0.05, 
        method=correction
    )
    
    # Store corrected values back
    for i, test in enumerate(all_tests):
        test['p_corrected'] = p_corrected[i]
        test['significant'] = rejected[i]
    
    # Reorganize into nested dict
    group_pvalues = {}
    
    for metric in metrics_map.keys():
        group_pvalues[metric] = {}
        
        metric_tests = [t for t in all_tests if t['metric'] == metric]
        
        for test in metric_tests:
            group_pvalues[metric][test['diagnosis']] = {
                'p_uncorrected': test['p_value'],
                'p_corrected': test['p_corrected'],
                'significant': test['significant'],
                'n_group': test['n_group'],
                'n_control': test['n_control']
            }
    
    # Summary
    n_total = len(all_tests)
    n_sig_uncorrected = sum(1 for t in all_tests if t['p_value'] < 0.05)
    n_sig_corrected = sum(1 for t in all_tests if t['significant'])
    
    print(f"\n[INFO] Multiple Testing Summary:")
    print(f"  Total tests: {n_total}")
    print(f"  Significant (uncorrected α=0.05): {n_sig_uncorrected}/{n_total}")
    print(f"  Significant (corrected α=0.05): {n_sig_corrected}/{n_total}")
    print(f"  Correction method: {correction.upper()}")
    
    return group_pvalues

# ============================================================================
# IMPROVED: ERRORBAR PLOTS WITH CORRECTED P-VALUES
# ============================================================================
def plot_deviation_errorbar_improved(results_df, save_dir, norm_diagnosis='HC',
                                     correction='fdr_bh', custom_colors=None, name="Analysis"):
    """
    Create improved errorbar plots with multiple testing correction.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch
    
    print(f"\n[INFO] Creating improved errorbar plots with {correction.upper()} correction...")
    
    # Calculate corrected p-values
    group_pvalues = calculate_group_pvalues_corrected(results_df, norm_diagnosis, correction)
    
    if not group_pvalues:
        print("[ERROR] Could not calculate p-values. Skipping plots.")
        return
    
    # Metrics mapping
    metrics_map = {
        'reconstruction_error': 'deviation_score_recon',
        'kl_divergence': 'deviation_score_kl',
        'deviation_score': 'deviation_score'
    }
    
    # Nice labels
    label_map = {
        'reconstruction_error': 'Reconstruction Error (D_MSE)',
        'kl_divergence': 'KL Divergence (D_KL)',
        'deviation_score': 'Bootstrap Deviation Score'
    }
    
    # Merge CAT groups
    results_processed = results_df.copy()
    results_processed.loc[results_processed['Diagnosis'].isin(['CAT-SSD', 'CAT-MDD']), 'Diagnosis'] = 'CAT'
    
    # Filter to main diagnoses
    keep_diagnoses = ['HC', 'MDD', 'SSD', 'CAT']
    results_processed = results_processed[results_processed['Diagnosis'].isin(keep_diagnoses)]
    
    for display_name, metric_col in metrics_map.items():
        
        # Check if column exists
        if metric_col not in results_processed.columns:
            print(f"[WARNING] Column {metric_col} not found in results_df. Skipping {display_name}.")
            continue
        
        print(f"\n[INFO] Creating plot for {display_name} (using column: {metric_col})...")
        
        # Calculate summary statistics
        summary_df = (
            results_processed.groupby("Diagnosis")[metric_col]
            .agg(['mean', 'sem', 'count'])
            .reset_index()
        )
        
        # Add corrected p-values
        def get_pval_info(diag):
            if diag == norm_diagnosis:
                return {'p_corrected': np.nan, 'significant': False}
            return group_pvalues.get(display_name, {}).get(diag, {'p_corrected': np.nan, 'significant': False})
        
        summary_df["p_corrected"] = summary_df["Diagnosis"].map(lambda d: get_pval_info(d)['p_corrected'])
        summary_df["significant"] = summary_df["Diagnosis"].map(lambda d: get_pval_info(d)['significant'])
        
        # Sort (HC first, then by mean descending)
        diagnosis_order = [norm_diagnosis] + sorted(
            [d for d in summary_df["Diagnosis"] if d != norm_diagnosis],
            key=lambda d: summary_df[summary_df["Diagnosis"] == d]["mean"].values[0],
            reverse=True
        )
        diagnosis_order_plot = diagnosis_order[::-1]  # Bottom to top
        
        summary_df["Diagnosis"] = pd.Categorical(
            summary_df["Diagnosis"], 
            categories=diagnosis_order_plot, 
            ordered=True
        )
        summary_df = summary_df.sort_values("Diagnosis")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Errorbars (black)
        ax.errorbar(
            summary_df["mean"], 
            range(len(summary_df)),
            xerr=summary_df["sem"],
            fmt='none',
            color='black', 
            capsize=5, 
            linewidth=1.5,
            zorder=1
        )
        
        # Colored scatter points based on significance
        colors = []
        for _, row in summary_df.iterrows():
            diag = row["Diagnosis"]
            p_val = row["p_corrected"]
            
            if diag == norm_diagnosis:
                colors.append('#808080')  # Gray for reference
            elif np.isnan(p_val):
                colors.append('#FF6B6B')  # Red for missing
            elif p_val < 0.001:
                colors.append('#0066CC')  # Dark blue - highly significant
            elif p_val < 0.01:
                colors.append('#3399FF')  # Medium blue
            elif p_val < 0.05:
                colors.append('#FFD700')  # Gold - significant
            else:
                colors.append('#FF6B6B')  # Light red - not significant
        
        ax.scatter(
            summary_df["mean"], 
            range(len(summary_df)),
            c=colors, 
            s=150, 
            edgecolors='black', 
            linewidth=2, 
            zorder=3
        )
        
        # Add significance annotations
        for i, (_, row) in enumerate(summary_df.iterrows()):
            if row["Diagnosis"] != norm_diagnosis:
                p_val = row["p_corrected"]
                
                if np.isnan(p_val):
                    text = ''
                elif p_val < 0.001:
                    text = '***'
                elif p_val < 0.01:
                    text = '**'
                elif p_val < 0.05:
                    text = '*'
                else:
                    text = 'n.s.'
                
                if text:
                    x_pos = row["mean"] + row["sem"] * 1.3
                    ax.text(x_pos, i, text,
                           fontsize=12, fontweight='bold', 
                           va='center', ha='left')
        
        # Add sample size annotations
        for i, (_, row) in enumerate(summary_df.iterrows()):
            x_pos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
            ax.text(x_pos, i, f"n={int(row['count'])}",
                   fontsize=9, va='center', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Labels and formatting
        metric_label = label_map.get(display_name, display_name.replace('_', ' ').title())
        ax.set_xlabel(f"{metric_label} (Mean ± SEM)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Diagnosis", fontsize=12, fontweight='bold')
        ax.set_title(
            f"{metric_label} by Diagnosis\n"
            f"Multiple Testing: {correction.upper()} | Norm: {norm_diagnosis}\n"
            f"{name}",
            fontsize=13, fontweight='bold', pad=15
        )
        
        # Y-axis
        ax.set_yticks(range(len(summary_df)))
        ax.set_yticklabels(summary_df["Diagnosis"], fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='x', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        
        # Legend
        legend_elements = [
            Patch(facecolor='#0066CC', edgecolor='black', label='p < 0.001 ***'),
            Patch(facecolor='#3399FF', edgecolor='black', label='p < 0.01 **'),
            Patch(facecolor='#FFD700', edgecolor='black', label='p < 0.05 *'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='p ≥ 0.05 (n.s.)'),
            Patch(facecolor='#808080', edgecolor='black', label=f'Reference: {norm_diagnosis}')
        ]
        ax.legend(
            handles=legend_elements, 
            loc='lower right', 
            fontsize=9,
            framealpha=0.95,
            edgecolor='black'
        )
        
        plt.tight_layout()
        
        # Save - use display_name for filename
        filename = f"{display_name}_errorbar_corrected_{correction}.png"
        save_path = f"{save_dir}/figures/distributions/{filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    print(f"\n[INFO] All improved errorbar plots created!")

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
    """Run complete analysis with CAT splitting and color options"""
    
    print(f"Running analysis with CAT {'split' if split_CAT else 'combined'}")
    if custom_colors:
        print(f"Using custom colors: {custom_colors}")
    
    # Original plots
    summary_dict = plot_deviation_distributions(
        results_df=results_df,
        save_dir=save_dir,
        col_jitter=col_jitter,
        norm_diagnosis=norm_diagnosis,
        split_CAT=split_CAT,
        custom_colors=custom_colors,
        name=name
    )
    
    # ========== NEW: Improved errorbar plots ==========
    plot_deviation_errorbar_improved(
        results_df=results_df,
        save_dir=save_dir,
        norm_diagnosis=norm_diagnosis,
        correction='fdr_bh',  # Can also try 'bonferroni'
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
    Regional deviation analysis with:
    - 2 plot types per diagnosis:
      1. Color intensity plot (significance-based coloring)
      2. Dual-axis plot (Cliff's Delta + Volume Difference)
    - Heatmap 1: Top 30 CAT-affected regions
    - Heatmap 2: Top 30 overall-affected regions
    
    Direction colors based on RAW MRI values (median + Mann-Whitney test):
    - Red: Patient > HC (increased volume/thickness/etc)
    - Blue: Patient < HC (decreased volume/thickness/etc)
    - Gray: Not significant (p≥0.05)
    
    Color intensity indicates significance level:
    - Dark = p<0.001 (highly significant)
    - Light = p<0.05 (significant)
    """
    
    print("\n[INFO] Starting regional deviation analysis with advanced visualization...")
    print("[INFO] Will create 2 plot types per diagnosis:")
    print("      1. Color intensity plot (p-value based)")
    print("      2. Dual-axis plot (normative + anatomical)")
    
    # Import required libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import List
    import os
    from scipy.stats import mannwhitneyu
    from matplotlib.patches import Patch
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def format_roi_name_for_plotting_local(original_roi_name: str, atlas_name_from_config: str | List[str] = None) -> str:
        """Format ROI name for plotting: [V] RightHippocampus (Neurom)"""
        
        atlas_abbreviations = {
            "cobra": "[C]", "lpba40": "[L]", "neuromorphometrics": "[N]", "Neurom": "[N]",
            "suit": "[S]", "SUIT": "[S]", "thalamic_nuclei": "[TN]", "thalamus": "[T]",
            "aal3": "[A]", "AAL3": "[AAL3]", "ibsr": "[I]", "IBSR": "[I]",
            "schaefer100": "[S100]", "Sch100": "[S100]", "schaefer200": "[S200]", "Sch200": "[S200]",
            "aparc_dk40": "[DK]", "DK40": "[DK]", "aparc_destrieux": "[DES]", "Destrieux": "[DES]",
        }
        
        parts = original_roi_name.split('_')
        if len(parts) < 3:
            return original_roi_name
        
        volume_type = parts[0]
        atlas_prefix = parts[1]
        roi_name = "_".join(parts[2:])
        
        if volume_type.startswith('V'):
            vtype_abbr = f"[{volume_type[1:].upper()}]"
        else:
            vtype_abbr = f"[{volume_type}]"
        
        return f"{vtype_abbr} {roi_name} ({atlas_prefix})"

    def format_roi_names_list_for_plotting_local(roi_names_list: List[str], atlas_name_from_config: str | List[str] = None) -> List[str]:
        return [format_roi_name_for_plotting_local(name, atlas_name_from_config) for name in roi_names_list]

    def get_color_by_significance(direction, p_value):
        """
        Returns color with intensity based on significance level
        
        Args:
            direction: 'increase', 'decrease', or 'neutral'
            p_value: Mann-Whitney p-value
        
        Returns:
            RGB tuple or 'gray'
        """
        if direction == 'neutral' or p_value >= 0.05:
            return 'gray'
        
        # -log10 transformation (higher = more significant)
        if p_value > 0:
            sig_score = -np.log10(p_value)
        else:
            sig_score = 10
        
        # Clamp between 1.3 (p=0.05) and 4 (p<0.001)
        sig_score = np.clip(sig_score, 1.3, 4)
        intensity = (sig_score - 1.3) / (4 - 1.3)  # Normalize to 0-1
        
        if direction == 'increase':
            # Red: light → dark
            r = 1.0
            g = 0.5 - intensity * 0.5
            b = 0.5 - intensity * 0.5
            return (r, g, b)
        else:  # decrease
            # Blue: light → dark
            r = 0.5 - intensity * 0.5
            g = 0.5 - intensity * 0.5
            b = 1.0
            return (r, g, b)

    # ========================================================================
    # SETUP
    # ========================================================================
    
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
    
    # Filter to main diagnoses
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
    # ========================================================================
    # CREATE LONG FORMAT DATA FOR CLINICAL CORRELATIONS (Subject × ROI level)
    # ========================================================================

    print(f"\n[INFO] Creating subject-ROI level data for clinical correlations...")

    # Convert wide format to long format
    results_long_format = []

    for _, row in results_df.iterrows():
        subject = row.get('Subject', None)
        filename = row.get('Filename', None)
        diagnosis = row['Diagnosis']
        
        # Extract all other metadata columns (age, sex, dataset, etc.)
        metadata_cols = [col for col in results_df.columns 
                if col not in region_cols 
                and col not in ['Subject', 'Filename', 'Diagnosis', 'deviation_score']]
        metadata = {col: row[col] for col in metadata_cols}
        
        # For each ROI, create a row
        for i, region_col in enumerate(region_cols):
            roi_name = formatted_roi_names_for_plotting[i] if i < len(formatted_roi_names_for_plotting) else f"Region_{i+1}"
            z_score = row[region_col]
            
            results_long_format.append({
                'Subject': subject,
                'Filename': filename,
                'Diagnosis': diagnosis,
                'ROI_Name': roi_name,
                'Region_Column': region_col,
                'deviation_score': z_score,  # This is the regional z-score
                **metadata  # Add all other metadata
            })

    results_long_format = pd.DataFrame(results_long_format)

    print(f"  ✓ Created long format data:")
    print(f"    Shape: {results_long_format.shape}")
    print(f"    Unique subjects: {results_long_format['Filename'].nunique()}")
    print(f"    Unique ROIs: {results_long_format['ROI_Name'].nunique()}")
    print(f"    Total rows (Subject × ROI): {len(results_long_format)}")

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

    # ========================================================================
    # CALCULATE EFFECT SIZES
    # ========================================================================
    
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
    # LOAD RAW MRI DATA FOR DIRECTION ANALYSIS
    # ========================================================================
    
    print("\n[INFO] Loading original MRI data to determine direction of changes...")
    
    try:
        mri_data_df = pd.read_csv(clinical_data_path)
        print(f"  ✓ Loaded MRI data: {mri_data_df.shape}")
        
        raw_roi_cols = [col.replace("_z_score", "") for col in region_cols]
        available_raw_cols = [col for col in raw_roi_cols if col in mri_data_df.columns]
        
        if len(available_raw_cols) == 0:
            print("[WARNING] No matching raw ROI columns found. Using neutral colors.")
            use_direction_colors = False
        else:
            print(f"  ✓ Found {len(available_raw_cols)} matching raw ROI columns")
            use_direction_colors = True
            
            results_with_raw = results_df.merge(
                mri_data_df[['Filename'] + available_raw_cols],
                on='Filename',
                how='left'
            )
            
    except Exception as e:
        print(f"[WARNING] Could not load MRI data: {e}. Using neutral colors.")
        use_direction_colors = False

    # ========================================================================
    # CREATE PLOTS FOR EACH DIAGNOSIS
    # ========================================================================
    
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue

        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        if dx_effect_sizes.empty:
            continue

        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        top_regions = dx_effect_sizes_sorted.head(16)

        # ====================================================================
        # Compute direction for each region (ROBUST: Median + Mann-Whitney)
        # ====================================================================
        
        if use_direction_colors:
            region_directions = {}
            region_stats = {}
            
            print(f"\n  [INFO] Computing robust direction (median + Mann-Whitney) for {diagnosis}...")
            
            for idx, row in top_regions.iterrows():
                roi_name = row["ROI_Name"]
                region_col = row["Region_Column"]
                raw_col = region_col.replace("_z_score", "")
                
                if raw_col not in available_raw_cols:
                    region_directions[roi_name] = 'neutral'
                    region_stats[roi_name] = None
                    continue
                
                try:
                    dx_raw_values = results_with_raw[results_with_raw['Diagnosis'] == diagnosis][raw_col].dropna()
                    hc_raw_values = results_with_raw[results_with_raw['Diagnosis'] == norm_diagnosis][raw_col].dropna()
                    
                    if len(dx_raw_values) > 0 and len(hc_raw_values) > 0:
                        # ROBUST: Use median instead of mean
                        dx_median = dx_raw_values.median()
                        hc_median = hc_raw_values.median()
                        
                        # Mann-Whitney U test for significance (non-parametric)
                        try:
                            _, p_value = mannwhitneyu(dx_raw_values, hc_raw_values, alternative='two-sided')
                        except Exception as e:
                            print(f"    [WARNING] Mann-Whitney test failed for {roi_name}: {e}")
                            p_value = 1.0
                        
                        region_stats[roi_name] = {
                            'patient_median': dx_median,
                            'hc_median': hc_median,
                            'patient_mean': dx_raw_values.mean(),
                            'hc_mean': hc_raw_values.mean(),
                            'difference': dx_median - hc_median,
                            'percent_change': ((dx_median - hc_median) / hc_median * 100) if hc_median != 0 else 0,
                            'p_value_mw': p_value,
                            'n_patient': len(dx_raw_values),
                            'n_hc': len(hc_raw_values)
                        }
                        
                        # Only color if Mann-Whitney test is significant (p < 0.05)
                        if p_value < 0.05:
                            if dx_median > hc_median:
                                region_directions[roi_name] = 'increase'
                            else:
                                region_directions[roi_name] = 'decrease'
                        else:
                            region_directions[roi_name] = 'neutral'
                    else:
                        region_directions[roi_name] = 'neutral'
                        region_stats[roi_name] = None
                        
                except Exception as e:
                    print(f"    [WARNING] Could not compute direction for {roi_name}: {e}")
                    region_directions[roi_name] = 'neutral'
                    region_stats[roi_name] = None
            
            # Summary of direction results
            n_increase = sum(1 for d in region_directions.values() if d == 'increase')
            n_decrease = sum(1 for d in region_directions.values() if d == 'decrease')
            n_neutral = sum(1 for d in region_directions.values() if d == 'neutral')
            print(f"    ✓ Direction results: {n_increase} increased (red), {n_decrease} decreased (blue), {n_neutral} neutral (gray)")
        else:
            region_directions = {row["ROI_Name"]: 'neutral' for _, row in top_regions.iterrows()}
            region_stats = {row["ROI_Name"]: None for _, row in top_regions.iterrows()}

        # Format ROI names
        formatted_labels = []
        for roi_name in top_regions["ROI_Name"]:
            if '(' in roi_name and ')' in roi_name:
                formatted_labels.append(roi_name)
            else:
                formatted_labels.append(format_roi_name_for_plotting_local(roi_name, atlas_name))
        
        y_pos = np.arange(len(top_regions))

        # ====================================================================
        # PLOT 1: Color Intensity Plot (Significance-based)
        # ====================================================================
        
        print(f"  [INFO] Creating Plot 1: Color intensity plot for {diagnosis}...")
        
        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_axes([0.42, 0.1, 0.50, 0.80])
        
        legend_elements = []
        used_colors = set()

        for i, (idx, row) in enumerate(top_regions.iterrows()):
            effect = row["Cliffs_Delta"]
            ci_low = row["Cliffs_Delta_CI_Low"]
            ci_high = row["Cliffs_Delta_CI_High"]
            roi_name = row["ROI_Name"]

            if pd.isna(ci_low) or pd.isna(ci_high):
                continue
            
            direction = region_directions.get(roi_name, 'neutral')
            stats = region_stats.get(roi_name)
            p_value = stats['p_value_mw'] if stats else 1.0
            
            color = get_color_by_significance(direction, p_value)
            
            # Track unique colors for legend
            if direction == 'increase' and p_value < 0.001 and 'increase_high' not in used_colors:
                legend_elements.append((color, 'Increased (p<0.001)'))
                used_colors.add('increase_high')
            elif direction == 'increase' and p_value < 0.05 and 'increase_low' not in used_colors:
                legend_elements.append((color, 'Increased (p<0.05)'))
                used_colors.add('increase_low')
            elif direction == 'decrease' and p_value < 0.001 and 'decrease_high' not in used_colors:
                legend_elements.append((color, 'Decreased (p<0.001)'))
                used_colors.add('decrease_high')
            elif direction == 'decrease' and p_value < 0.05 and 'decrease_low' not in used_colors:
                legend_elements.append((color, 'Decreased (p<0.05)'))
                used_colors.add('decrease_low')

            ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=1.5, alpha=0.9)
            ax.plot(effect, i, 'o', markersize=4, markerfacecolor=color, markeredgecolor=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(formatted_labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        valid_ci_rows = top_regions.dropna(subset=["Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High"])
        if not valid_ci_rows.empty:
            min_value = valid_ci_rows["Cliffs_Delta_CI_Low"].min()
            max_value = valid_ci_rows["Cliffs_Delta_CI_High"].max()
            value_range = max_value - min_value
            buffer = value_range * 0.05
            ax.set_xlim(min_value - buffer, max_value + buffer)
        else:
            ax.set_xlim(-1, 1)

        ax.set_xlabel("Effect Size (Cliff's Delta)", fontsize=10, fontweight='bold')
        ax.set_title(f"Top 16 Regions: {diagnosis} vs. {norm_diagnosis}\n({name})\nColor intensity = significance level", 
                    fontsize=11, fontweight='bold', pad=15)

        if legend_elements:
            patches = [Patch(facecolor=color, label=label) for color, label in legend_elements]
            patches.append(Patch(facecolor='gray', label='Not significant (p≥0.05)'))
            ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(False)

        plt.savefig(f"{save_dir}/figures/paper_style_intensity_{diagnosis}_vs_{norm_diagnosis}.png",
                    dpi=300, facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: paper_style_intensity_{diagnosis}_vs_{norm_diagnosis}.png")

        # ====================================================================
        # PLOT 2: Dual-Axis Plot (Normative vs. Anatomical)
        # ====================================================================
        
        print(f"  [INFO] Creating Plot 2: Dual-axis plot for {diagnosis}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        
        # LEFT PANEL: Cliff's Delta (normative deviation)
        for i, (idx, row) in enumerate(top_regions.iterrows()):
            effect = row["Cliffs_Delta"]
            ci_low = row["Cliffs_Delta_CI_Low"]
            ci_high = row["Cliffs_Delta_CI_High"]

            if pd.isna(ci_low) or pd.isna(ci_high):
                continue
            
            ax1.plot([ci_low, ci_high], [i, i], 'k-', linewidth=1.5, alpha=0.8)
            ax1.plot(effect, i, 'ko', markersize=6, markerfacecolor='black', markeredgecolor='black')

        ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_xlabel("Normative Deviation\n(Cliff's Delta on Z-scores)", fontsize=10, fontweight='bold')
        ax1.set_title(f"Deviation from {norm_diagnosis} Norm", fontsize=11, fontweight='bold')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(formatted_labels, fontsize=9)
        ax1.invert_yaxis()
        ax1.grid(False)
        
        # RIGHT PANEL: Volume difference with significance colors
        legend_elements_dual = []
        used_colors_dual = set()
        
        for i, (idx, row) in enumerate(top_regions.iterrows()):
            roi_name = row["ROI_Name"]
            stats = region_stats.get(roi_name)
            
            if stats:
                diff = stats['difference']
                p_val = stats['p_value_mw']
                direction = region_directions.get(roi_name, 'neutral')
                
                color = get_color_by_significance(direction, p_val)
                alpha = 1.0 if p_val < 0.01 else (0.8 if p_val < 0.05 else 0.5)
                
                ax2.barh(i, diff, height=0.6, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
                
                # Track for legend
                if direction == 'increase' and p_val < 0.001 and 'inc_high' not in used_colors_dual:
                    legend_elements_dual.append((color, 'p<0.001: Increased'))
                    used_colors_dual.add('inc_high')
                elif direction == 'increase' and p_val < 0.05 and 'inc_low' not in used_colors_dual:
                    legend_elements_dual.append((color, 'p<0.05: Increased'))
                    used_colors_dual.add('inc_low')
                elif direction == 'decrease' and p_val < 0.001 and 'dec_high' not in used_colors_dual:
                    legend_elements_dual.append((color, 'p<0.001: Decreased'))
                    used_colors_dual.add('dec_high')
                elif direction == 'decrease' and p_val < 0.05 and 'dec_low' not in used_colors_dual:
                    legend_elements_dual.append((color, 'p<0.05: Decreased'))
                    used_colors_dual.add('dec_low')
            else:
                ax2.barh(i, 0, height=0.6, color='lightgray', alpha=0.3)

        ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_xlabel("Volume Difference (Median)\nPatient - HC", fontsize=10, fontweight='bold')
        ax2.set_title(f"Absolute Volume Change", fontsize=11, fontweight='bold')
        ax2.grid(False)
        
        if legend_elements_dual:
            patches = [Patch(facecolor=color, label=label) for color, label in legend_elements_dual]
            patches.append(Patch(facecolor='gray', alpha=0.5, label='p≥0.05: Not significant'))
            ax2.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)

        plt.suptitle(f"{diagnosis} vs. {norm_diagnosis}: Normative vs. Anatomical Changes\n({name})", 
                    fontsize=12, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_dir}/figures/paper_style_dualaxis_{diagnosis}_vs_{norm_diagnosis}.png",
                    dpi=300, facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: paper_style_dualaxis_{diagnosis}_vs_{norm_diagnosis}.png")
        
        # ====================================================================
        # Save direction statistics
        # ====================================================================
        
        if use_direction_colors:
            direction_stats = []
            for idx, row in top_regions.iterrows():
                roi_name = row["ROI_Name"]
                stats = region_stats.get(roi_name)
                
                if stats is not None:
                    direction_stats.append({
                        'ROI_Name': roi_name,
                        'Diagnosis': diagnosis,
                        'Direction': region_directions[roi_name],
                        'Patient_Median': stats['patient_median'],
                        'HC_Median': stats['hc_median'],
                        'Patient_Mean': stats['patient_mean'],
                        'HC_Mean': stats['hc_mean'],
                        'Absolute_Difference_Median': stats['difference'],
                        'Percent_Change_Median': stats['percent_change'],
                        'P_Value_MannWhitney': stats['p_value_mw'],
                        'N_Patient': stats['n_patient'],
                        'N_HC': stats['n_hc'],
                        'Cliffs_Delta': row['Cliffs_Delta'],
                        'Significant_CliffsDelta': row['Significant_Bootstrap_p05_uncorrected']
                    })
            
            if direction_stats:
                direction_df = pd.DataFrame(direction_stats)
                direction_df.to_csv(
                    f"{save_dir}/figures/direction_stats_{diagnosis}_vs_{norm_diagnosis}.csv",
                    index=False
                )
                print(f"    ✓ Saved direction statistics")
                
                # Print summary
                sig_changes = direction_df[direction_df['Direction'] != 'neutral']
                if len(sig_changes) > 0:
                    print(f"      Significant changes (p<0.05): {len(sig_changes)}/{len(direction_df)}")
                    print(f"        Increases: {(sig_changes['Direction'] == 'increase').sum()}")
                    print(f"        Decreases: {(sig_changes['Direction'] == 'decrease').sum()}")
    
    print("\n[INFO] Regional deviation analysis finished.")
    print(f"[INFO] Created 2 plot types per diagnosis in: {save_dir}/figures/")
    print(f"      - paper_style_intensity_*.png (significance-based coloring)")
    print(f"      - paper_style_dualaxis_*.png (normative vs. anatomical)")

    # ========================================================================
    # ADDITIONAL COMPARISON VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING ADDITIONAL COMPARISON PLOTS")
    print("="*80)
    
    try:
        # 1. Spider/Radar plots
        create_spider_plot(
            effect_sizes_df, 
            save_dir, 
            norm_diagnosis=norm_diagnosis,
            top_n=20,  # Top 20 per diagnosis
            custom_colors={'HC': '#125E8A', 'SSD': '#3E885B', 'MDD': '#BEDCFE', 'CAT': '#2F4B26'},
            separate_plots=True
        )
        
        # 2. Parallel coordinates
        pivot_data = create_parallel_coordinates_plot(
            effect_sizes_df,
            save_dir,
            norm_diagnosis=norm_diagnosis,
            top_n=20,
            custom_colors={'HC': '#125E8A', 'SSD': '#3E885B', 'MDD': '#BEDCFE', 'CAT': '#2F4B26'}
        )
        
        # 3. Cluster analysis (heatmap only with significance)
        cluster_data, roi_significance = create_cluster_analysis(
            effect_sizes_df,
            save_dir,
            norm_diagnosis=norm_diagnosis,
            min_rois=10,
            custom_colors={'HC': '#125E8A', 'SSD': '#3E885B', 'MDD': '#BEDCFE', 'CAT': '#2F4B26'}
        )
        
        # 4. UMAP based on effect sizes
        embedding, umap_distances = create_effect_size_umap(
            effect_sizes_df,
            save_dir,
            norm_diagnosis=norm_diagnosis,
            custom_colors={'HC': '#125E8A', 'SSD': '#3E885B', 'MDD': '#BEDCFE', 'CAT': '#2F4B26'},
            min_rois=10
        )
        
        print("\n✓ All comparison visualizations created!")
        
    except Exception as e:
        print(f"[WARNING] Could not create some comparison plots: {e}")
        import traceback
        traceback.print_exc()
    
    
    return effect_sizes_df, results_long_format

def create_dual_plots_for_diagnosis(diagnosis, top_regions, region_directions, region_stats,
                                   norm_diagnosis, name, save_dir, atlas_name, 
                                   format_roi_name_for_plotting_local):
    """
    Creates both plot types for a given diagnosis
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Helper function for color intensity
    def get_color_by_significance(direction, p_value):
        """Returns color with intensity based on significance level"""
        if direction == 'neutral' or p_value >= 0.05:
            return 'gray'
        
        # -log10 transformation
        if p_value > 0:
            sig_score = -np.log10(p_value)
        else:
            sig_score = 10
        
        # Clamp between 1.3 (p=0.05) and 4 (p<0.001)
        sig_score = np.clip(sig_score, 1.3, 4)
        intensity = (sig_score - 1.3) / (4 - 1.3)
        
        if direction == 'increase':
            r = 1.0
            g = 0.5 - intensity * 0.5
            b = 0.5 - intensity * 0.5
            return (r, g, b)
        else:
            r = 0.5 - intensity * 0.5
            g = 0.5 - intensity * 0.5
            b = 1.0
            return (r, g, b)
    
    # Format ROI names
    formatted_labels = []
    for roi_name in top_regions["ROI_Name"]:
        if '(' in roi_name and ')' in roi_name:
            formatted_labels.append(roi_name)
        else:
            formatted_labels.append(format_roi_name_for_plotting_local(roi_name, atlas_name))
    
    y_pos = np.arange(len(top_regions))
    
    # ========================================================================
    # PLOT 1: Color Intensity Plot
    # ========================================================================
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_axes([0.42, 0.1, 0.50, 0.80])
    
    legend_elements = []
    used_colors = set()

    for i, (idx, row) in enumerate(top_regions.iterrows()):
        effect = row["Cliffs_Delta"]
        ci_low = row["Cliffs_Delta_CI_Low"]
        ci_high = row["Cliffs_Delta_CI_High"]
        roi_name = row["ROI_Name"]

        if pd.isna(ci_low) or pd.isna(ci_high):
            continue
        
        direction = region_directions.get(roi_name, 'neutral')
        stats = region_stats.get(roi_name)
        p_value = stats['p_value_mw'] if stats else 1.0
        
        color = get_color_by_significance(direction, p_value)
        
        # Track unique colors for legend
        if direction == 'increase' and p_value < 0.001 and 'increase_high' not in used_colors:
            legend_elements.append((color, 'Increased (p<0.001)'))
            used_colors.add('increase_high')
        elif direction == 'increase' and p_value < 0.05 and 'increase_low' not in used_colors:
            legend_elements.append((color, 'Increased (p<0.05)'))
            used_colors.add('increase_low')
        elif direction == 'decrease' and p_value < 0.001 and 'decrease_high' not in used_colors:
            legend_elements.append((color, 'Decreased (p<0.001)'))
            used_colors.add('decrease_high')
        elif direction == 'decrease' and p_value < 0.05 and 'decrease_low' not in used_colors:
            legend_elements.append((color, 'Decreased (p<0.05)'))
            used_colors.add('decrease_low')

        ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=1.5, alpha=0.9)
        ax.plot(effect, i, 'o', markersize=4, markerfacecolor=color, markeredgecolor=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(formatted_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    valid_ci_rows = top_regions.dropna(subset=["Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High"])
    if not valid_ci_rows.empty:
        min_value = valid_ci_rows["Cliffs_Delta_CI_Low"].min()
        max_value = valid_ci_rows["Cliffs_Delta_CI_High"].max()
        value_range = max_value - min_value
        buffer = value_range * 0.05
        ax.set_xlim(min_value - buffer, max_value + buffer)
    else:
        ax.set_xlim(-1, 1)

    ax.set_xlabel("Effect Size (Cliff's Delta)", fontsize=10, fontweight='bold')
    ax.set_title(f"Top 16 Regions: {diagnosis} vs. {norm_diagnosis}\n({name})\nColor intensity = significance level", 
                fontsize=11, fontweight='bold', pad=15)

    if legend_elements:
        patches = [Patch(facecolor=color, label=label) for color, label in legend_elements]
        patches.append(Patch(facecolor='gray', label='Not significant (p≥0.05)'))
        ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(False)

    plt.savefig(f"{save_dir}/figures/paper_style_intensity_{diagnosis}_vs_{norm_diagnosis}.png",
                dpi=300, facecolor='white')
    plt.close()
    
    # ========================================================================
    # PLOT 2: Dual-Axis Plot
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    
    # LEFT: Cliff's Delta
    for i, (idx, row) in enumerate(top_regions.iterrows()):
        effect = row["Cliffs_Delta"]
        ci_low = row["Cliffs_Delta_CI_Low"]
        ci_high = row["Cliffs_Delta_CI_High"]

        if pd.isna(ci_low) or pd.isna(ci_high):
            continue
        
        ax1.plot([ci_low, ci_high], [i, i], 'k-', linewidth=1.5, alpha=0.8)
        ax1.plot(effect, i, 'ko', markersize=6)

    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Normative Deviation\n(Cliff's Delta on Z-scores)", fontsize=10, fontweight='bold')
    ax1.set_title(f"Deviation from {norm_diagnosis} Norm", fontsize=11, fontweight='bold')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(formatted_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(False)
    
    # RIGHT: Volume difference
    legend_elements_dual = []
    used_colors_dual = set()
    
    for i, (idx, row) in enumerate(top_regions.iterrows()):
        roi_name = row["ROI_Name"]
        stats = region_stats.get(roi_name)
        
        if stats:
            diff = stats['difference']
            p_val = stats['p_value_mw']
            direction = region_directions.get(roi_name, 'neutral')
            
            color = get_color_by_significance(direction, p_val)
            alpha = 1.0 if p_val < 0.01 else (0.8 if p_val < 0.05 else 0.5)
            
            ax2.barh(i, diff, height=0.6, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            
            if direction == 'increase' and p_val < 0.001 and 'inc_high' not in used_colors_dual:
                legend_elements_dual.append((color, 'p<0.001: Increased'))
                used_colors_dual.add('inc_high')
            elif direction == 'increase' and p_val < 0.05 and 'inc_low' not in used_colors_dual:
                legend_elements_dual.append((color, 'p<0.05: Increased'))
                used_colors_dual.add('inc_low')
            elif direction == 'decrease' and p_val < 0.001 and 'dec_high' not in used_colors_dual:
                legend_elements_dual.append((color, 'p<0.001: Decreased'))
                used_colors_dual.add('dec_high')
            elif direction == 'decrease' and p_val < 0.05 and 'dec_low' not in used_colors_dual:
                legend_elements_dual.append((color, 'p<0.05: Decreased'))
                used_colors_dual.add('dec_low')
        else:
            ax2.barh(i, 0, height=0.6, color='lightgray', alpha=0.3)

    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Volume Difference (Median)\nPatient - HC", fontsize=10, fontweight='bold')
    ax2.set_title(f"Absolute Volume Change", fontsize=11, fontweight='bold')
    ax2.grid(False)
    
    if legend_elements_dual:
        patches = [Patch(facecolor=color, label=label) for color, label in legend_elements_dual]
        patches.append(Patch(facecolor='gray', alpha=0.5, label='p≥0.05: Not significant'))
        ax2.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)

    plt.suptitle(f"{diagnosis} vs. {norm_diagnosis}: Normative vs. Anatomical Changes\n({name})", 
                fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_dir}/figures/paper_style_dualaxis_{diagnosis}_vs_{norm_diagnosis}.png",
                dpi=300, facecolor='white')
    plt.close()
    
    return True
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

#-------NEW FUNCTIONS FOR MULTIVARIATE COMPARISON-------------------

def create_spider_plot(effect_sizes_df, save_dir, norm_diagnosis='HC', 
                       top_n=20, custom_colors=None, separate_plots=True):
    """
    Create spider/radar plots comparing effect sizes across diagnoses.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        save_dir: Output directory
        norm_diagnosis: Reference diagnosis
        top_n: Number of top ROIs per diagnosis to include
        custom_colors: Color dict for diagnoses
        separate_plots: If True, create individual plots + combined plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from math import pi
    
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A", "SSD": "#3E885B", "MDD": "#BEDCFE",
            "CAT": "#2F4B26", "CAT-SSD": "#A67DB8", "CAT-MDD": "#160C28"
        }
    
    print(f"\n[INFO] Creating spider/radar plots...")
    
    # Get diagnoses (exclude norm)
    diagnoses = [d for d in effect_sizes_df['Diagnosis'].unique() if d != norm_diagnosis]
    
    # ========== NEW: Get union of top N ROIs from ALL diagnoses ==========
    effect_sizes_df['Abs_Cliffs_Delta'] = effect_sizes_df['Cliffs_Delta'].abs()
    
    all_top_rois = set()
    for diagnosis in diagnoses:
        diag_data = effect_sizes_df[effect_sizes_df['Diagnosis'] == diagnosis]
        top_rois_diag = diag_data.nlargest(top_n, 'Abs_Cliffs_Delta')['ROI_Name'].tolist()
        all_top_rois.update(top_rois_diag)
    
    top_rois = sorted(list(all_top_rois))  # Sort for consistency
    
    print(f"  Union of top {top_n} ROIs per diagnosis: {len(top_rois)} unique ROIs")
    print(f"  Breakdown:")
    for diagnosis in diagnoses:
        diag_data = effect_sizes_df[effect_sizes_df['Diagnosis'] == diagnosis]
        top_for_diag = diag_data.nlargest(top_n, 'Abs_Cliffs_Delta')['ROI_Name'].tolist()
        print(f"    {diagnosis}: {len(top_for_diag)} ROIs (contributed {len(set(top_for_diag))} unique)")
    
    # Filter to top ROIs
    plot_data = effect_sizes_df[effect_sizes_df['ROI_Name'].isin(top_rois)].copy()
    
    # Shorten ROI names for readability
    roi_short_names = [roi.split('(')[0].strip()[:20] for roi in top_rois]
    
    # Setup radar chart
    num_vars = len(top_rois)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    if separate_plots:
        # ========== INDIVIDUAL PLOTS FOR EACH DIAGNOSIS ==========
        for diagnosis in diagnoses:
            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
            
            # Get values for this diagnosis
            values = []
            for roi in top_rois:
                roi_data = plot_data[(plot_data['Diagnosis'] == diagnosis) & 
                                    (plot_data['ROI_Name'] == roi)]
                if len(roi_data) > 0:
                    values.append(roi_data['Cliffs_Delta'].iloc[0])
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            # Plot with DARKER lines
            color = custom_colors.get(diagnosis, '#888888')
            ax.plot(angles, values, 'o-', linewidth=3, label=diagnosis, 
                   color=color, alpha=1.0, markersize=6)  # ← Thicker, opaque
            ax.fill(angles, values, alpha=0.25, color=color)
            
            # Styling
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(roi_short_names, size=8)
            ax.set_ylim(-1, 1)
            ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
            ax.set_yticklabels(['-0.8', '-0.4', '0', '0.4', '0.8'], size=9)
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
            
            # ========== REMOVE OUTER RING ==========
            ax.spines['polar'].set_visible(False)
            
            # Add reference circle at 0 (darker)
            ax.plot(angles, [0]*len(angles), 'k-', linewidth=1.5, alpha=0.7)
            
            # Title
            plt.title(f"{diagnosis} vs. {norm_diagnosis}\nRegional Effect Sizes (Cliff's Delta)\n"
                     f"Union of Top {top_n} ROIs per Diagnosis ({len(top_rois)} total)",
                     size=14, weight='bold', y=1.08)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/spider_plot_{diagnosis}_vs_{norm_diagnosis}.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ Saved: spider_plot_{diagnosis}_vs_{norm_diagnosis}.png")
    
    # ========== ALWAYS CREATE COMBINED PLOT ==========
    print(f"\n  [INFO] Creating combined plot with all diagnoses...")
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    for diagnosis in diagnoses:
        # Get values for this diagnosis
        values = []
        for roi in top_rois:
            roi_data = plot_data[(plot_data['Diagnosis'] == diagnosis) & 
                                (plot_data['ROI_Name'] == roi)]
            if len(roi_data) > 0:
                values.append(roi_data['Cliffs_Delta'].iloc[0])
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        
        # Plot with DARKER lines and different colors
        color = custom_colors.get(diagnosis, '#888888')
        ax.plot(angles, values, 'o-', linewidth=2.5, label=diagnosis, 
               color=color, markersize=5, alpha=0.9)  # ← Thicker, more opaque
        ax.fill(angles, values, alpha=0.12, color=color)  # ← Less transparent fill
    
    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(roi_short_names, size=8)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticklabels(['-0.8', '-0.4', '0', '0.4', '0.8'], size=10)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    
    # ========== REMOVE OUTER RING ==========
    ax.spines['polar'].set_visible(False)
    
    # Add reference circle at 0 (darker)
    ax.plot(angles, [0]*len(angles), 'k-', linewidth=1.5, alpha=0.7)
    
    # Legend with larger font and better positioning
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), 
             fontsize=11, framealpha=0.95, edgecolor='black')
    
    # Title
    plt.title(f"Regional Effect Sizes Across Diagnoses\nvs. {norm_diagnosis}\n"
             f"Union of Top {top_n} ROIs per Diagnosis ({len(top_rois)} total)",
             size=15, weight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/spider_plot_combined_vs_{norm_diagnosis}.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: spider_plot_combined_vs_{norm_diagnosis}.png")
    
    print(f"[INFO] Spider plots complete!")

def create_parallel_coordinates_plot(effect_sizes_df, save_dir, norm_diagnosis='HC',
                                    top_n=20, custom_colors=None):
    """
    Create parallel coordinates plot comparing effect sizes across ROIs.
    Uses union of top N ROIs from all diagnoses.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A", "SSD": "#3E885B", "MDD": "#BEDCFE",
            "CAT": "#2F4B26", "CAT-SSD": "#A67DB8", "CAT-MDD": "#160C28"
        }
    
    print(f"\n[INFO] Creating parallel coordinates plot...")
    
    # Get diagnoses (exclude norm)
    diagnoses = [d for d in effect_sizes_df['Diagnosis'].unique() if d != norm_diagnosis]
    
    # ========== NEW: Get union of top N ROIs from ALL diagnoses ==========
    effect_sizes_df['Abs_Cliffs_Delta'] = effect_sizes_df['Cliffs_Delta'].abs()
    
    all_top_rois = set()
    for diagnosis in diagnoses:
        diag_data = effect_sizes_df[effect_sizes_df['Diagnosis'] == diagnosis]
        top_rois_diag = diag_data.nlargest(top_n, 'Abs_Cliffs_Delta')['ROI_Name'].tolist()
        all_top_rois.update(top_rois_diag)
    
    top_rois = sorted(list(all_top_rois))
    
    print(f"  Union of top {top_n} ROIs per diagnosis: {len(top_rois)} unique ROIs")
    
    # Filter to top ROIs
    plot_data = effect_sizes_df[effect_sizes_df['ROI_Name'].isin(top_rois)].copy()
    
    # Pivot data: rows = diagnoses, columns = ROIs
    pivot_data = plot_data.pivot(index='Diagnosis', columns='ROI_Name', values='Cliffs_Delta')
    pivot_data = pivot_data[top_rois]  # Ensure consistent ROI order
    
    # Shorten ROI names
    roi_short_names = [roi.split('(')[0].strip()[:25] for roi in top_rois]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(18, len(top_rois)*0.6), 9))
    
    # X positions for ROIs
    x_positions = np.arange(len(top_rois))
    
    # Plot each diagnosis as a line with DARKER, THICKER lines
    for diagnosis in diagnoses:
        if diagnosis not in pivot_data.index:
            continue
            
        values = pivot_data.loc[diagnosis].values
        color = custom_colors.get(diagnosis, '#888888')
        
        # Plot line with markers - THICKER and MORE OPAQUE
        ax.plot(x_positions, values, 'o-', 
               linewidth=2.5, markersize=7,  # ← Thicker
               label=diagnosis, color=color, alpha=0.95)  # ← More opaque
    
    # Add horizontal line at 0 (darker)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(roi_short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Effect Size (Cliff's Delta)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Brain Regions", fontsize=13, fontweight='bold')
    ax.set_title(f"Regional Effect Sizes Across Diagnoses\nvs. {norm_diagnosis}\n"
                f"Union of Top {top_n} ROIs per Diagnosis ({len(top_rois)} total)",
                fontsize=15, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    
    # Adjust y-limits for better visualization
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/parallel_coordinates_vs_{norm_diagnosis}.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: parallel_coordinates_vs_{norm_diagnosis}.png")
    print(f"[INFO] Parallel coordinates plot complete!")
    
    return pivot_data
def create_cluster_analysis(effect_sizes_df, save_dir, norm_diagnosis='HC',
                           min_rois=10, custom_colors=None):
    """
    Create clustered heatmap with significance markers showing which regions
    differ significantly between diagnoses.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        save_dir: Output directory
        norm_diagnosis: Reference diagnosis
        min_rois: Minimum number of ROIs to include
        custom_colors: Color dict for diagnoses
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    from matplotlib.patches import Rectangle
    
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A", "SSD": "#3E885B", "MDD": "#BEDCFE",
            "CAT": "#2F4B26", "CAT-SSD": "#A67DB8", "CAT-MDD": "#160C28"
        }
    
    print(f"\n[INFO] Performing cluster analysis with significance testing...")
    
    # Get ROIs that appear in all diagnoses
    roi_counts = effect_sizes_df.groupby('ROI_Name')['Diagnosis'].nunique()
    n_diagnoses = effect_sizes_df['Diagnosis'].nunique()
    complete_rois = roi_counts[roi_counts == n_diagnoses].index.tolist()
    
    if len(complete_rois) < min_rois:
        print(f"[WARNING] Only {len(complete_rois)} ROIs have data for all diagnoses. Using all available ROIs.")
        complete_rois = effect_sizes_df['ROI_Name'].unique().tolist()
    
    print(f"  Using {len(complete_rois)} ROIs for clustering")
    
    # Filter data
    cluster_data = effect_sizes_df[effect_sizes_df['ROI_Name'].isin(complete_rois)].copy()
    
    # Get diagnoses (exclude norm)
    diagnoses = [d for d in cluster_data['Diagnosis'].unique() if d != norm_diagnosis]
    
    # Pivot data: rows = diagnoses, columns = ROIs
    pivot_data = cluster_data.pivot(index='Diagnosis', columns='ROI_Name', values='Cliffs_Delta')
    pivot_data = pivot_data.fillna(0)
    pivot_data = pivot_data.loc[diagnoses]
    
    # ========================================================================
    # SIGNIFICANCE TESTING: Which ROIs differ between diagnosis pairs?
    # ========================================================================
    
    print(f"\n  [INFO] Testing for significant differences between diagnosis pairs...")
    
    # We need the original regional z-scores, not just Cliff's Delta
    # Load from results_df if available, or use Cliff's Delta as proxy
    
    # Create significance matrix: ROIs × Diagnosis pairs
    from itertools import combinations
    diagnosis_pairs = list(combinations(diagnoses, 2))
    
    print(f"    Testing {len(diagnosis_pairs)} diagnosis pairs: {diagnosis_pairs}")
    
    # For each ROI, test if it differs significantly between any pair
    roi_significance = {}  # ROI → list of significant pairs
    
    for roi in complete_rois:
        roi_data = cluster_data[cluster_data['ROI_Name'] == roi]
        
        significant_pairs = []
        p_values = []
        
        for diag1, diag2 in diagnosis_pairs:
            d1_cliff = roi_data[roi_data['Diagnosis'] == diag1]['Cliffs_Delta'].values
            d2_cliff = roi_data[roi_data['Diagnosis'] == diag2]['Cliffs_Delta'].values
            
            if len(d1_cliff) > 0 and len(d2_cliff) > 0:
                # Simple test: are the effect sizes substantially different?
                diff = abs(d1_cliff[0] - d2_cliff[0])
                
                # Use CI bounds if available for more robust test
                d1_ci_low = roi_data[roi_data['Diagnosis'] == diag1]['Cliffs_Delta_CI_Low'].values
                d1_ci_high = roi_data[roi_data['Diagnosis'] == diag1]['Cliffs_Delta_CI_High'].values
                d2_ci_low = roi_data[roi_data['Diagnosis'] == diag2]['Cliffs_Delta_CI_Low'].values
                d2_ci_high = roi_data[roi_data['Diagnosis'] == diag2]['Cliffs_Delta_CI_High'].values
                
                if len(d1_ci_low) > 0 and len(d2_ci_low) > 0:
                    # Test if CIs don't overlap
                    ci_overlap = not (d1_ci_high[0] < d2_ci_low[0] or d2_ci_high[0] < d1_ci_low[0])
                    
                    if not ci_overlap and diff > 0.2:  # Non-overlapping CIs + substantial difference
                        significant_pairs.append(f"{diag1}-{diag2}")
                        p_values.append(0.01)  # Proxy p-value
                    else:
                        p_values.append(1.0)
                else:
                    # Fallback: just use effect size difference
                    if diff > 0.3:  # Substantial difference threshold
                        significant_pairs.append(f"{diag1}-{diag2}")
                        p_values.append(0.05)
                    else:
                        p_values.append(1.0)
        
        roi_significance[roi] = significant_pairs
    
    # Count how many ROIs show differences for each pair
    print(f"\n    Significant differences by diagnosis pair:")
    for pair in diagnosis_pairs:
        pair_str = f"{pair[0]}-{pair[1]}"
        n_sig = sum(1 for pairs in roi_significance.values() if pair_str in pairs)
        print(f"      {pair_str}: {n_sig}/{len(complete_rois)} ROIs")
    
    # ========================================================================
    # CLUSTERED HEATMAP WITH SIGNIFICANCE MARKERS
    # ========================================================================
    
    print(f"\n  [INFO] Creating clustered heatmap with significance markers...")
    
    # Create clustermap
    g = sns.clustermap(
        pivot_data.T,  # Transpose: ROIs as rows, diagnoses as columns
        method='ward',
        metric='euclidean',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(10, max(16, len(complete_rois)*0.3)),
        cbar_kws={'label': "Cliff's Delta", 'shrink': 0.5},
        dendrogram_ratio=0.1,
        linewidths=0.5,
        linecolor='lightgray',
        yticklabels=True,
        xticklabels=True,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)  # Position colorbar better
    )
    
    # ========================================================================
    # ADD SIGNIFICANCE MARKERS
    # ========================================================================
    
    # Get the reordered ROI order after clustering
    reordered_rois = [complete_rois[i] for i in g.dendrogram_row.reordered_ind]
    
    # Add asterisks or boxes for significant ROIs
    ax = g.ax_heatmap
    
    for row_idx, roi in enumerate(reordered_rois):
        if roi in roi_significance and len(roi_significance[roi]) > 0:
            # This ROI shows significant differences
            n_pairs = len(roi_significance[roi])
            
            # Add a colored box around the entire row
            if n_pairs >= 2:  # Differs in 2+ pairs
                rect = Rectangle((0, row_idx), len(diagnoses), 1, 
                               fill=False, edgecolor='gold', linewidth=2.5)
                ax.add_patch(rect)
            elif n_pairs == 1:  # Differs in 1 pair
                rect = Rectangle((0, row_idx), len(diagnoses), 1, 
                               fill=False, edgecolor='orange', linewidth=1.5)
                ax.add_patch(rect)
    
    # Styling
    g.ax_heatmap.set_xlabel("Diagnosis", fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel("Brain Regions", fontsize=12, fontweight='bold')
    g.fig.suptitle(
        f"Clustered Effect Sizes: Diagnoses vs. {norm_diagnosis}\n"
        f"(Ward Linkage, Euclidean Distance)\n"
        f"Gold border = differs in 2+ pairs | Orange border = differs in 1 pair",
        fontsize=13, fontweight='bold', x=0.5, y=0.98
    )
    
    plt.savefig(f"{save_dir}/figures/cluster_heatmap_with_significance_vs_{norm_diagnosis}.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: cluster_heatmap_with_significance_vs_{norm_diagnosis}.png")
    
    # ========================================================================
    # SAVE SIGNIFICANCE RESULTS
    # ========================================================================
    
    sig_results = []
    for roi, pairs in roi_significance.items():
        if len(pairs) > 0:
            sig_results.append({
                'ROI_Name': roi,
                'N_Significant_Pairs': len(pairs),
                'Significant_Pairs': ', '.join(pairs)
            })
    
    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        sig_df = sig_df.sort_values('N_Significant_Pairs', ascending=False)
        sig_df.to_csv(f"{save_dir}/roi_significance_between_diagnoses.csv", index=False)
        print(f"    ✓ Saved: roi_significance_between_diagnoses.csv")
    
    print(f"\n[INFO] Cluster analysis complete!")
    
    return pivot_data, roi_significance

def create_effect_size_umap(effect_sizes_df, save_dir, norm_diagnosis='HC',
                           custom_colors=None, min_rois=10):
    """
    Create visualization based on regional effect size patterns.
    Uses UMAP if n >= 5, otherwise creates distance plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    
    if custom_colors is None:
        custom_colors = {
            "HC": "#125E8A", "SSD": "#3E885B", "MDD": "#BEDCFE",
            "CAT": "#2F4B26", "CAT-SSD": "#A67DB8", "CAT-MDD": "#160C28"
        }
    
    print(f"\n[INFO] Creating effect size pattern visualization...")
    
    # Get ROIs that appear in all diagnoses
    roi_counts = effect_sizes_df.groupby('ROI_Name')['Diagnosis'].nunique()
    n_diagnoses = effect_sizes_df['Diagnosis'].nunique()
    complete_rois = roi_counts[roi_counts == n_diagnoses].index.tolist()
    
    if len(complete_rois) < min_rois:
        print(f"[WARNING] Only {len(complete_rois)} ROIs available. Using all.")
        complete_rois = effect_sizes_df['ROI_Name'].unique().tolist()
    
    print(f"  Using {len(complete_rois)} ROIs")
    
    # Filter data
    umap_data = effect_sizes_df[effect_sizes_df['ROI_Name'].isin(complete_rois)].copy()
    
    # Get diagnoses (exclude norm)
    diagnoses = [d for d in umap_data['Diagnosis'].unique() if d != norm_diagnosis]
    
    # Pivot: rows = diagnoses, columns = ROIs (effect sizes)
    pivot_data = umap_data.pivot(index='Diagnosis', columns='ROI_Name', values='Cliffs_Delta')
    pivot_data = pivot_data.fillna(0)
    pivot_data = pivot_data.loc[diagnoses]
    
    print(f"  Data shape: {pivot_data.shape} (diagnoses × ROIs)")
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pivot_data.values)
    
    # Compute distances
    distances = pdist(data_scaled, metric='euclidean')
    distance_matrix = squareform(distances)
    distance_df = pd.DataFrame(
        distance_matrix,
        index=diagnoses,
        columns=diagnoses
    )
    
    distance_df.to_csv(f"{save_dir}/effect_size_distances_vs_{norm_diagnosis}.csv")
    
    print(f"\n  Pairwise distances between diagnoses:")
    for i, diag1 in enumerate(diagnoses):
        for diag2 in diagnoses[i+1:]:
            dist = distance_df.loc[diag1, diag2]
            print(f"    {diag1} ↔ {diag2}: {dist:.3f}")
    
    # ========================================================================
    # DECIDE: UMAP or SIMPLE PLOT?
    # ========================================================================
    
    if len(diagnoses) < 4:
        print(f"\n  [INFO] Only {len(diagnoses)} diagnoses - creating distance bar plot instead of UMAP")
        
        # Create distance bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pairs = []
        dists = []
        for i, diag1 in enumerate(diagnoses):
            for diag2 in diagnoses[i+1:]:
                pairs.append(f"{diag1} - {diag2}")
                dists.append(distance_df.loc[diag1, diag2])
        
        colors_bar = [custom_colors.get(p.split(' - ')[0], '#1f77b4') for p in pairs]
        bars = ax.barh(pairs, dists, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Euclidean Distance\n(Standardized Effect Sizes)', 
                     fontsize=12, fontweight='bold')
        ax.set_title(
            f'Pairwise Distances Between Diagnoses\n'
            f'Based on Regional Effect Size Patterns vs. {norm_diagnosis}\n'
            f'({len(complete_rois)} ROIs, Standardized)',
            fontsize=13, fontweight='bold', pad=15
        )
        ax.grid(axis='x', alpha=0.3)
        
        # Add distance values on bars
        for bar, dist in zip(bars, dists):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {dist:.2f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distance_plot_effect_sizes_vs_{norm_diagnosis}.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: distance_plot_effect_sizes_vs_{norm_diagnosis}.png")
        
        return None, distance_df
    
    # ========================================================================
    # UMAP (for n >= 4)
    # ========================================================================
    
    try:
        import umap
        
        print(f"\n  [INFO] Running UMAP...")
        
        # Critical fix: adjust n_neighbors for small sample size
        n_neighbors = max(2, min(len(diagnoses) - 1, 15))
        
        print(f"    Using n_neighbors={n_neighbors} (adjusted for {len(diagnoses)} samples)")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42,
            init='random'  # Use random instead of spectral for small n
        )
        
        embedding = reducer.fit_transform(data_scaled)
        
        print(f"    ✓ UMAP complete")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, diagnosis in enumerate(diagnoses):
            color = custom_colors.get(diagnosis, '#888888')
            
            ax.scatter(
                embedding[i, 0],
                embedding[i, 1],
                c=color,
                s=400,
                alpha=0.8,
                edgecolors='black',
                linewidth=2,
                label=diagnosis,
                zorder=3
            )
            
            ax.text(
                embedding[i, 0],
                embedding[i, 1],
                diagnosis,
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=4
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.set_title(
            f'UMAP: Diagnoses Based on Regional Effect Size Patterns\n'
            f'vs. {norm_diagnosis} ({len(complete_rois)} ROIs)\n'
            f'Closer points = more similar regional deviation patterns',
            fontsize=13, fontweight='bold', pad=15
        )
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.95, edgecolor='black')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/umap_effect_sizes_vs_{norm_diagnosis}.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: umap_effect_sizes_vs_{norm_diagnosis}.png")
        
        return embedding, distance_df
        
    except Exception as e:
        print(f"  [WARNING] UMAP failed: {e}")
        print(f"  [INFO] Creating distance plot instead...")
        
        # Fallback to distance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pairs = []
        dists = []
        for i, diag1 in enumerate(diagnoses):
            for diag2 in diagnoses[i+1:]:
                pairs.append(f"{diag1} - {diag2}")
                dists.append(distance_df.loc[diag1, diag2])
        
        colors_bar = [custom_colors.get(p.split(' - ')[0], '#1f77b4') for p in pairs]
        bars = ax.barh(pairs, dists, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Euclidean Distance', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Pairwise Distances Between Diagnoses\n'
            f'Based on Regional Effect Size Patterns',
            fontsize=13, fontweight='bold', pad=15
        )
        ax.grid(axis='x', alpha=0.3)
        
        for bar, dist in zip(bars, dists):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {dist:.2f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distance_plot_effect_sizes_vs_{norm_diagnosis}.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: distance_plot_effect_sizes_vs_{norm_diagnosis}.png")
        
        return None, distance_df
    
def analyze_clinical_correlations(results_df, clinical_data_path, save_dir,
                                  datasets_with_clinical=['NSS', 'whiteCAT'],
                                  min_subjects=20,
                                  apply_fdr=True,
                                  skip_ml=True,  # ← NEW
                                  alpha_uncorrected=0.05):
    """
    Analyze correlations between deviation scores and clinical scores.
    Works with both subject-level and regional (ROI-level) deviation data.
    
    Args:
        results_df: DataFrame with deviation scores per subject (or per subject/ROI)
        clinical_data_path: Path to complete_metadata.csv
        save_dir: Output directory
        datasets_with_clinical: Datasets that have clinical data
        min_subjects: Minimum subjects needed for correlation
        apply_fdr: Whether to apply FDR correction
        skip_ml: Skip machine learning analysis (default: True)
    
    Returns:
        correlation_results: DataFrame with all correlations
        diag_corr_df: DataFrame with diagnosis-stratified correlations
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr, pearsonr
    from statsmodels.stats.multitest import multipletests
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*80)
    print("ANALYZING BRAIN-CLINICAL CORRELATIONS")
    print("="*80)
    
    # ========================================================================
    # 0. DETERMINE ANALYSIS LEVEL (subject-level vs regional)
    # ========================================================================
    
    has_roi_data = 'ROI_Name' in results_df.columns
    
    if has_roi_data:
        print("\n[INFO] Detected REGIONAL (ROI-level) deviation data")
        analysis_level = "regional"
    else:
        print("\n[INFO] Detected SUBJECT-LEVEL deviation data")
        analysis_level = "subject"
    
    # ========================================================================
    # 1. LOAD AND MERGE CLINICAL DATA
    # ========================================================================
    
    print(f"\n[INFO] Loading clinical data from: {clinical_data_path}")
    clinical_df = pd.read_csv(clinical_data_path)
    
    # Clinical score columns
    clinical_scores = [
        'GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 'PANSS_General', 
        'PANSS_Total', 'BPRS_Total', 'NCRS_Motor', 'NCRS_Affective', 
        'NCRS_Behavioral', 'NCRS_Total', 'NSS_Motor', 'NSS_Total'
    ]
    
    # Filter to datasets with clinical data
    clinical_df = clinical_df[clinical_df['Dataset'].isin(datasets_with_clinical)]
    
    print(f"  Clinical data available for {len(clinical_df)} subjects from datasets: {datasets_with_clinical}")
    
    # Check data availability
    print(f"\n  Clinical scores availability:")
    for score in clinical_scores:
        n_available = clinical_df[score].notna().sum()
        pct = (n_available / len(clinical_df)) * 100
        print(f"    {score:20s}: {n_available:4d} / {len(clinical_df)} ({pct:.1f}%)")
    
    # ========================================================================
    # 2. MERGE WITH DEVIATION SCORES
    # ========================================================================
    
    print(f"\n[INFO] Merging clinical data with deviation scores...")
    
    if 'Filename' not in results_df.columns:
        print("[ERROR] results_df must have 'Filename' column")
        return None, None
    
    # Merge on Filename
    merged_df = results_df.merge(
        clinical_df[['Filename', 'Dataset', 'Diagnosis'] + clinical_scores],
        on='Filename',
        how='inner',
        suffixes=('', '_clinical')
    )
    
    print(f"  Merged data: {len(merged_df)} records")
    print(f"  Unique subjects: {merged_df['Filename'].nunique()}")
    
    if has_roi_data:
        print(f"  Unique ROIs: {merged_df['ROI_Name'].nunique()}")
    
    # Filter to patients only (exclude HC)
    merged_df = merged_df[merged_df['Diagnosis'] != 'HC']
    
    print(f"  After filtering to patients: {merged_df['Filename'].nunique()} subjects")
    
    # Check diagnoses
    print(f"\n  Diagnoses in clinical sample:")
    for diag, count in merged_df['Diagnosis'].value_counts().items():
        n_subj = merged_df[merged_df['Diagnosis']==diag]['Filename'].nunique()
        if has_roi_data:
            print(f"    {diag}: {count} ROI measurements ({n_subj} subjects)")
        else:
            print(f"    {diag}: {n_subj} subjects")
    
    # ========================================================================
    # 3. CORRELATION ANALYSIS
    # ========================================================================
    
    if analysis_level == "regional":
        print(f"\n[INFO] Computing ROI-wise correlations...")
        correlation_results = _analyze_regional_correlations(
            merged_df, clinical_scores, min_subjects
        )
    else:
        print(f"\n[INFO] Computing subject-level correlations...")
        correlation_results = _analyze_subject_level_correlations(
            merged_df, clinical_scores, min_subjects
        )
    
    if not correlation_results:
        print("[WARNING] No correlations could be computed!")
        return pd.DataFrame(), pd.DataFrame()
    
    corr_df = pd.DataFrame(correlation_results)
    print(f"    Computed {len(corr_df)} correlations")
    
    # ========================================================================
    # 4. MULTIPLE TESTING CORRECTION
    # ========================================================================
    
    print(f"\n[INFO] Applying FDR correction...")
    
    # FDR correction on Spearman p-values
    reject, pvals_corrected, _, _ = multipletests(
        corr_df['Spearman_p'].values,
        alpha=0.05,
        method='fdr_bh'
    )
    
    corr_df['Spearman_p_corrected'] = pvals_corrected
    corr_df['Significant_FDR'] = reject
    
    n_sig = corr_df['Significant_FDR'].sum()
    print(f"    Significant correlations (FDR < 0.05): {n_sig} / {len(corr_df)}")
    
    # Save results
    corr_df_sorted = corr_df.sort_values('Spearman_p', ascending=True)
    corr_df_sorted.to_csv(f"{save_dir}/clinical_correlations_all.csv", index=False)
    
    # Save only significant
    sig_corr = corr_df_sorted[corr_df_sorted['Significant_FDR']]
    if len(sig_corr) > 0:
        sig_corr.to_csv(f"{save_dir}/clinical_correlations_significant.csv", index=False)
        print(f"    ✓ Saved significant correlations to: clinical_correlations_significant.csv")
    
    # ========================================================================
    # 5. VISUALIZATIONS
    # ========================================================================
    
    if analysis_level == "regional" and n_sig > 0:
        _create_regional_visualizations(
            merged_df, corr_df, sig_corr, clinical_scores, save_dir, has_roi_data
        )
    elif analysis_level == "subject" and n_sig > 0:
        _create_subject_level_visualizations(
            merged_df, corr_df, sig_corr, clinical_scores, save_dir
        )
    
    # ========================================================================
    # 6. DIAGNOSIS-STRATIFIED ANALYSIS
    # ========================================================================
    
    print(f"\n[INFO] Performing diagnosis-stratified analysis...")
    
    diag_corr_df = _analyze_by_diagnosis(
        merged_df, clinical_scores, min_subjects, has_roi_data
    )
    
    if not diag_corr_df.empty:
        diag_corr_df.to_csv(f"{save_dir}/clinical_correlations_by_diagnosis.csv", index=False)
        
        # Summary
        print(f"\n  Significant correlations by diagnosis (FDR < 0.05):")
        for diagnosis in diag_corr_df['Diagnosis'].unique():
            n_sig = diag_corr_df[
                (diag_corr_df['Diagnosis'] == diagnosis) & 
                (diag_corr_df['Significant_FDR'])
            ].shape[0]
            print(f"    {diagnosis}: {n_sig}")
    
    print(f"\n[INFO] Clinical correlation analysis complete!")
    print(f"  Results saved to: {save_dir}/")
    
    return corr_df, diag_corr_df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _analyze_regional_correlations(merged_df, clinical_scores, min_subjects):
    """ROI-wise correlation analysis"""
    from scipy.stats import spearmanr, pearsonr
    
    correlation_results = []
    rois = merged_df['ROI_Name'].unique()
    
    for roi in rois:
        roi_data = merged_df[merged_df['ROI_Name'] == roi].copy()
        
        for score in clinical_scores:
            valid_data = roi_data[roi_data[score].notna()].copy()
            
            if len(valid_data) < min_subjects:
                continue
            
            x = valid_data['deviation_score'].values
            y = valid_data[score].values
            
            rho, p_val = spearmanr(x, y)
            r, p_val_pearson = pearsonr(x, y)
            
            correlation_results.append({
                'ROI_Name': roi,
                'Clinical_Score': score,
                'N_Subjects': len(valid_data),
                'Spearman_rho': rho,
                'Spearman_p': p_val,
                'Pearson_r': r,
                'Pearson_p': p_val_pearson
            })
    
    return correlation_results


def _analyze_subject_level_correlations(merged_df, clinical_scores, min_subjects):
    """Subject-level correlation analysis (no ROI dimension)"""
    from scipy.stats import spearmanr, pearsonr
    
    correlation_results = []
    
    # One correlation per clinical score (using global deviation score)
    for score in clinical_scores:
        valid_data = merged_df[merged_df[score].notna()].copy()
        
        if len(valid_data) < min_subjects:
            continue
        
        x = valid_data['deviation_score'].values
        y = valid_data[score].values
        
        rho, p_val = spearmanr(x, y)
        r, p_val_pearson = pearsonr(x, y)
        
        correlation_results.append({
            'Analysis_Level': 'Global',
            'Clinical_Score': score,
            'N_Subjects': len(valid_data),
            'Spearman_rho': rho,
            'Spearman_p': p_val,
            'Pearson_r': r,
            'Pearson_p': p_val_pearson
        })
    
    return correlation_results


def _analyze_by_diagnosis(merged_df, clinical_scores, min_subjects, has_roi_data):
    """Diagnosis-stratified analysis"""
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    
    diag_corr_results = []
    
    for diagnosis in merged_df['Diagnosis'].unique():
        diag_data = merged_df[merged_df['Diagnosis'] == diagnosis]
        n_subjects = diag_data['Filename'].nunique()
        print(f"\n  {diagnosis}: {n_subjects} subjects")
        
        if has_roi_data:
            rois = diag_data['ROI_Name'].unique()
            for roi in rois:
                roi_diag_data = diag_data[diag_data['ROI_Name'] == roi]
                
                for score in clinical_scores:
                    valid_data = roi_diag_data[roi_diag_data[score].notna()]
                    
                    if len(valid_data) < min(10, min_subjects):
                        continue
                    
                    x = valid_data['deviation_score'].values
                    y = valid_data[score].values
                    rho, p_val = spearmanr(x, y)
                    
                    diag_corr_results.append({
                        'Diagnosis': diagnosis,
                        'ROI_Name': roi,
                        'Clinical_Score': score,
                        'N_Subjects': len(valid_data),
                        'Spearman_rho': rho,
                        'Spearman_p': p_val
                    })
        else:
            # Subject-level analysis
            for score in clinical_scores:
                valid_data = diag_data[diag_data[score].notna()]
                
                if len(valid_data) < min(10, min_subjects):
                    continue
                
                x = valid_data['deviation_score'].values
                y = valid_data[score].values
                rho, p_val = spearmanr(x, y)
                
                diag_corr_results.append({
                    'Diagnosis': diagnosis,
                    'Analysis_Level': 'Global',
                    'Clinical_Score': score,
                    'N_Subjects': len(valid_data),
                    'Spearman_rho': rho,
                    'Spearman_p': p_val
                })
    
    if not diag_corr_results:
        return pd.DataFrame()
    
    diag_corr_df = pd.DataFrame(diag_corr_results)
    
    # FDR correction per diagnosis
    for diagnosis in diag_corr_df['Diagnosis'].unique():
        diag_mask = diag_corr_df['Diagnosis'] == diagnosis
        pvals = diag_corr_df.loc[diag_mask, 'Spearman_p'].values
        
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        
        diag_corr_df.loc[diag_mask, 'Spearman_p_corrected'] = pvals_corrected
        diag_corr_df.loc[diag_mask, 'Significant_FDR'] = reject
    
    return diag_corr_df


def _create_subject_level_visualizations(merged_df, corr_df, sig_corr, clinical_scores, save_dir):
    """Create visualizations for subject-level correlations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    figures_dir = f"{save_dir}/figures/clinical_correlations"
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\n[INFO] Creating subject-level correlation visualizations...")
    
    # Bar plot of correlations
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute correlation
    plot_data = corr_df.copy()
    plot_data['abs_rho'] = plot_data['Spearman_rho'].abs()
    plot_data = plot_data.sort_values('abs_rho', ascending=True)
    
    colors = ['#d62728' if (rho > 0 and sig) else '#1f77b4' if (rho < 0 and sig) else 'gray' 
              for rho, sig in zip(plot_data['Spearman_rho'], plot_data['Significant_FDR'])]
    
    y_pos = np.arange(len(plot_data))
    ax.barh(y_pos, plot_data['Spearman_rho'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['Clinical_Score'], fontsize=10)
    ax.set_xlabel("Spearman's ρ", fontsize=12, fontweight='bold')
    ax.set_title(
        "Global Deviation Score × Clinical Score Correlations\n"
        "(Red/Blue = FDR significant, Gray = non-significant)",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/subject_level_correlations_barplot.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: subject_level_correlations_barplot.png")
    
    # Scatter plots for significant correlations
    if len(sig_corr) > 0:
        n_plots = min(6, len(sig_corr))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(sig_corr.head(n_plots).iterrows()):
            ax = axes[idx]
            score = row['Clinical_Score']
            
            plot_data = merged_df[merged_df[score].notna()].copy()
            
            # Plot by diagnosis
            for diag in plot_data['Diagnosis'].unique():
                diag_data = plot_data[plot_data['Diagnosis'] == diag]
                ax.scatter(
                    diag_data['deviation_score'],
                    diag_data[score],
                    label=diag,
                    alpha=0.6,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Regression line
            from scipy.stats import linregress
            x = plot_data['deviation_score'].values
            y = plot_data[score].values
            slope, intercept, _, _, _ = linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'k--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Global Deviation Score', fontsize=10, fontweight='bold')
            ax.set_ylabel(score, fontsize=10, fontweight='bold')
            ax.set_title(
                f"{score}\n"
                f"ρ = {row['Spearman_rho']:.3f}, p = {row['Spearman_p']:.1e}",
                fontsize=10, fontweight='bold'
            )
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for idx in range(n_plots, 6):
            axes[idx].axis('off')
        
        plt.suptitle(
            "Top Clinical Correlations (Subject-Level)",
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/scatter_subject_level_correlations.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: scatter_subject_level_correlations.png")
    
def _create_regional_visualizations(merged_df, corr_df, sig_corr, clinical_scores, save_dir, has_roi_data):
    """Create visualizations for ROI-level correlations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from scipy.stats import linregress
    
    figures_dir = f"{save_dir}/figures/clinical_correlations"
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\n[INFO] Creating ROI-level correlation visualizations...")
    
    n_sig = len(sig_corr)
    
    # --- 5.1 Heatmap: ROIs × Clinical Scores ---
    print(f"  Creating correlation heatmap...")
    
    # Pivot for heatmap
    heatmap_data = corr_df.pivot(
        index='ROI_Name',
        columns='Clinical_Score',
        values='Spearman_rho'
    )
    
    # Keep only ROIs/scores with sufficient data
    heatmap_data = heatmap_data.dropna(how='all', axis=0)
    heatmap_data = heatmap_data.dropna(how='all', axis=1)
    
    # Create mask for non-significant correlations
    pval_pivot = corr_df.pivot(
        index='ROI_Name',
        columns='Clinical_Score',
        values='Spearman_p_corrected'
    )
    mask = pval_pivot >= 0.05
    
    fig, ax = plt.subplots(figsize=(12, max(10, len(heatmap_data)*0.3)))
    
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        cbar_kws={'label': "Spearman's ρ"},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        mask=mask.reindex_like(heatmap_data),
        annot=False
    )
    
    ax.set_xlabel("Clinical Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Brain Region (ROI)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Brain-Clinical Correlations\n"
        f"(Only FDR-corrected significant correlations shown, n={n_sig})",
        fontsize=13, fontweight='bold', pad=15
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/heatmap_brain_clinical_correlations.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: heatmap_brain_clinical_correlations.png")
    
    # --- 5.2 Top Correlations Bar Plot ---
    print(f"  Creating top correlations plot...")
    
    if len(sig_corr) > 0:
        top_n = min(20, len(sig_corr))
        top_corr = sig_corr.nlargest(top_n, 'Spearman_rho')
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n*0.4)))
        
        # Create labels
        labels = [f"{row['ROI_Name'][:30]} × {row['Clinical_Score']}" 
                 for _, row in top_corr.iterrows()]
        
        # Color by strength
        colors = ['#d62728' if rho > 0 else '#1f77b4' 
                 for rho in top_corr['Spearman_rho']]
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, top_corr['Spearman_rho'], color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Spearman's ρ", fontsize=12, fontweight='bold')
        ax.set_title(
            f"Top {top_n} Brain-Clinical Correlations (FDR < 0.05)",
            fontsize=13, fontweight='bold', pad=15
        )
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/top_correlations_barplot.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: top_correlations_barplot.png")
    
    # --- 5.3 Scatter Plots for Top Correlations ---
    print(f"  Creating scatter plots for top correlations...")
    
    if len(sig_corr) > 0:
        n_scatter = min(6, len(sig_corr))
        sig_corr['abs_rho'] = sig_corr['Spearman_rho'].abs()
        top_scatter = sig_corr.nlargest(n_scatter, 'abs_rho')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_scatter.iterrows()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            roi = row['ROI_Name']
            score = row['Clinical_Score']
            
            # Get data
            plot_data = merged_df[
                (merged_df['ROI_Name'] == roi) & 
                (merged_df[score].notna())
            ].copy()
            
            # Plot by diagnosis
            for diag in plot_data['Diagnosis'].unique():
                diag_data = plot_data[plot_data['Diagnosis'] == diag]
                ax.scatter(
                    diag_data['deviation_score'],
                    diag_data[score],
                    label=diag,
                    alpha=0.6,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Regression line
            x = plot_data['deviation_score'].values
            y = plot_data[score].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'k--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Regional Deviation Score', fontsize=10, fontweight='bold')
            ax.set_ylabel(score, fontsize=10, fontweight='bold')
            ax.set_title(
                f"{roi[:30]}\n"
                f"ρ = {row['Spearman_rho']:.3f}, p = {row['Spearman_p']:.1e}",
                fontsize=10, fontweight='bold'
            )
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for idx in range(n_scatter, 6):
            axes[idx].axis('off')
        
        plt.suptitle(
            "Top Brain-Clinical Correlations (Scatter Plots)",
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/scatter_top_correlations.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: scatter_top_correlations.png")
    
    # --- 5.4 Clinical Score Summary ---
    print(f"  Creating clinical score summary...")
    
    score_summary = []
    
    for score in clinical_scores:
        score_corr = corr_df[corr_df['Clinical_Score'] == score]
        sig_score_corr = score_corr[score_corr['Significant_FDR']]
        
        if len(sig_score_corr) > 0:
            # Top positive
            top_pos = sig_score_corr[sig_score_corr['Spearman_rho'] > 0].nlargest(3, 'Spearman_rho')
            # Top negative
            top_neg = sig_score_corr[sig_score_corr['Spearman_rho'] < 0].nsmallest(3, 'Spearman_rho')
            
            score_summary.append({
                'Clinical_Score': score,
                'N_Significant_ROIs': len(sig_score_corr),
                'Top_Positive_ROIs': ', '.join(top_pos['ROI_Name'].tolist()),
                'Top_Negative_ROIs': ', '.join(top_neg['ROI_Name'].tolist())
            })
    
    if score_summary:
        import pandas as pd
        summary_df = pd.DataFrame(score_summary)
        summary_df = summary_df.sort_values('N_Significant_ROIs', ascending=False)
        summary_df.to_csv(f"{save_dir}/clinical_score_summary.csv", index=False)
        
        print(f"\n  Clinical Scores with Significant Brain Correlations:")
        for _, row in summary_df.iterrows():
            print(f"    {row['Clinical_Score']:20s}: {row['N_Significant_ROIs']} ROIs")

def _compute_ml_feature_importance(merged_df, clinical_scores, save_dir):
    """Compute ML feature importance (only works for regional data)"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    
    ml_results = []
    
    for score in clinical_scores:
        score_data = merged_df[merged_df[score].notna()].copy()
        
        if score_data['Filename'].nunique() < 30:
            continue
        
        pivot_ml = score_data.pivot_table(
            index='Filename',
            columns='ROI_Name',
            values='deviation_score',
            aggfunc='first'
        )
        
        clinical_vals = score_data.drop_duplicates('Filename').set_index('Filename')[score]
        clinical_vals = clinical_vals.reindex(pivot_ml.index)
        
        valid_idx = clinical_vals.notna() & pivot_ml.notna().all(axis=1)
        X = pivot_ml.loc[valid_idx].fillna(0).values
        y = clinical_vals.loc[valid_idx].values
        
        if len(y) < 20:
            continue
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        feature_importance = pd.DataFrame({
            'ROI_Name': pivot_ml.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        ml_results.append({
            'Clinical_Score': score,
            'N_Subjects': len(y),
            'CV_R2_Mean': cv_scores.mean(),
            'CV_R2_Std': cv_scores.std(),
            'Top_5_ROIs': ', '.join(feature_importance.head(5)['ROI_Name'].tolist())
        })
        
        feature_importance.to_csv(
            f"{save_dir}/feature_importance_{score}.csv",
            index=False
        )
    
    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        ml_df = ml_df.sort_values('CV_R2_Mean', ascending=False)
        ml_df.to_csv(f"{save_dir}/ml_prediction_performance.csv", index=False)
        
        print(f"\n  ML Prediction Performance (R² from 5-fold CV):")
        for _, row in ml_df.iterrows():
            print(f"    {row['Clinical_Score']:20s}: R² = {row['CV_R2_Mean']:.3f} ± {row['CV_R2_Std']:.3f}")
    
    return ml_results

def analyze_subgroups(results_df, save_dir, norm_diagnosis='HC'):
    """
    Analyze deviation scores stratified by demographic subgroups.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu
    
    print(f"\n[INFO] Performing subgroup analyses...")
    
    figures_dir = f"{save_dir}/figures/subgroups"
    os.makedirs(figures_dir, exist_ok=True)
    
    # ========================================================================
    # PREPARE SEX LABELS UPFRONT
    # ========================================================================
    
    # Convert Sex to string labels if needed
    if 'Sex' in results_df.columns:
        # Check if Sex is numeric (0/1) or string (M/F)
        unique_sex = results_df['Sex'].dropna().unique()
        
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in unique_sex):
            # Numeric - convert to labels
            results_df['Sex_Label'] = results_df['Sex'].map({
                0: 'M', 1: 'F', 
                0.0: 'M', 1.0: 'F',
                '0': 'M', '1': 'F'
            })
            print(f"  Converted numeric Sex values to labels (M/F)")
        else:
            # Already string
            results_df['Sex_Label'] = results_df['Sex']
            print(f"  Sex already in string format")
    else:
        print(f"[WARNING] Sex column not found. Skipping sex stratification.")
        results_df['Sex_Label'] = None
    
    # ========================================================================
    # AGE STRATIFICATION
    # ========================================================================
    
    print(f"\n  [1/3] Age stratification...")
    
    # Define age bins
    results_df['Age_Group'] = pd.cut(
        results_df['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(['deviation_score_recon', 'deviation_score_kl', 'deviation_score']):
        ax = axes[idx]
        
        # Prepare data for grouped boxplot
        plot_data = results_df[results_df['Diagnosis'] != norm_diagnosis].copy()
        
        sns.boxplot(
            data=plot_data,
            x='Age_Group',
            y=metric,
            hue='Diagnosis',
            ax=ax,
            palette={'MDD': '#BEDCFE', 'SSD': '#3E885B', 'CAT': '#2F4B26',
                    'CAT-SSD': '#A67DB8', 'CAT-MDD': '#160C28'}
        )
        
        metric_names = {
            'deviation_score_recon': 'Reconstruction Error',
            'deviation_score_kl': 'KL Divergence',
            'deviation_score': 'Combined Score'
        }
        
        ax.set_title(metric_names[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax.set_ylabel('Deviation Score', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.suptitle('Deviation Scores by Age Group and Diagnosis',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/age_stratification.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: age_stratification.png")
    
    # ========================================================================
    # SEX STRATIFICATION
    # ========================================================================
    
    print(f"\n  [2/3] Sex stratification...")
    
    # Check if Sex_Label is available and has data
    if results_df['Sex_Label'].notna().sum() == 0:
        print(f"[WARNING] No valid Sex data available. Creating placeholder plot.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Sex data not available', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(f"{figures_dir}/sex_stratification.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    ✓ Saved placeholder plot")
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(['deviation_score_recon', 'deviation_score_kl', 'deviation_score']):
            ax = axes[idx]
            
            plot_data = results_df[
                (results_df['Diagnosis'] != norm_diagnosis) &
                (results_df['Sex_Label'].notna())
            ].copy()
            
            # Check if we have both sexes
            sex_counts = plot_data['Sex_Label'].value_counts()
            
            if len(sex_counts) < 2:
                ax.text(0.5, 0.5, f'Only one sex in data: {sex_counts.index[0]}',
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Create violin plot with split by sex
            sns.violinplot(
                data=plot_data,
                x='Diagnosis',
                y=metric,
                hue='Sex_Label',
                split=True,
                ax=ax,
                palette={'M': '#4A90E2', 'F': '#E24A90'}
            )
            
            metric_names = {
                'deviation_score_recon': 'Reconstruction Error',
                'deviation_score_kl': 'KL Divergence',
                'deviation_score': 'Combined Score'
            }
            
            ax.set_title(metric_names[metric], fontsize=12, fontweight='bold')
            ax.set_xlabel('Diagnosis', fontsize=11, fontweight='bold')
            ax.set_ylabel('Deviation Score', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(title='Sex', fontsize=9)
        
        plt.suptitle('Deviation Scores by Sex and Diagnosis',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/sex_stratification.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: sex_stratification.png")
    
    # ========================================================================
    # DATASET STRATIFICATION
    # ========================================================================
    
    print(f"\n  [3/3] Dataset stratification...")
    
    # Only for datasets with sufficient samples
    dataset_counts = results_df['Dataset'].value_counts()
    datasets_to_plot = dataset_counts[dataset_counts >= 30].index.tolist()
    
    if len(datasets_to_plot) == 0:
        print(f"[WARNING] No datasets with ≥30 samples. Skipping dataset stratification.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Insufficient data per dataset (need ≥30 per dataset)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(f"{figures_dir}/dataset_stratification.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    ✓ Saved placeholder plot")
        
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        plot_data = results_df[
            (results_df['Dataset'].isin(datasets_to_plot)) &
            (results_df['Diagnosis'] != norm_diagnosis)
        ].copy()
        
        sns.boxplot(
            data=plot_data,
            x='Dataset',
            y='deviation_score',
            hue='Diagnosis',
            ax=ax,
            palette={'MDD': '#BEDCFE', 'SSD': '#3E885B', 'CAT': '#2F4B26',
                    'CAT-SSD': '#A67DB8', 'CAT-MDD': '#160C28'}
        )
        
        ax.set_title('Combined Deviation Score by Dataset and Diagnosis',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Combined Deviation Score', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/dataset_stratification.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: dataset_stratification.png")
    
    print(f"\n[INFO] Subgroup analyses complete!")

def analyze_ncrs_predictions(results_df, clinical_data_path, save_dir,
                             min_subjects=10,
                             apply_fdr=True,
                             alpha_uncorrected=0.05):
    """
    Analyze NCRS (Northoff Catatonia Rating Scale) scores specifically.
    
    Focus on motor symptoms and catatonia:
    1. Correlate regional deviations with NCRS subscales
    2. Predict NCRS scores from regional patterns
    3. Identify brain regions most predictive of catatonic symptoms
    
    Args:
        results_df: DataFrame with regional deviation scores
        clinical_data_path: Path to complete_metadata.csv
        save_dir: Output directory
        min_subjects: Minimum subjects needed for analysis
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr, pearsonr
    from statsmodels.stats.multitest import multipletests
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*80)
    print("ANALYZING NCRS (CATATONIA) SYMPTOM CORRELATIONS")
    print("="*80)
    
    # ========================================================================
    # 1. LOAD AND MERGE CLINICAL DATA
    # ========================================================================
    
    print(f"\n[INFO] Loading clinical data...")
    clinical_df = pd.read_csv(clinical_data_path)
    
    # NCRS scores
    ncrs_scores = ['NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total']
    
    # Check availability
    print(f"\n  NCRS scores availability:")
    for score in ncrs_scores:
        if score in clinical_df.columns:
            n_available = clinical_df[score].notna().sum()
            print(f"    {score:20s}: {n_available:4d} subjects")
        else:
            print(f"    {score:20s}: NOT FOUND in data")
    
    # Merge with deviation scores
    if 'Filename' not in results_df.columns:
        print("[ERROR] results_df must have 'Filename' column")
        return None
    
    merged_df = results_df.merge(
        clinical_df[['Filename', 'Dataset', 'Diagnosis'] + ncrs_scores],
        on='Filename',
        how='inner',
        suffixes=('', '_clinical')
    )
    
    # Filter to subjects with at least one NCRS score
    has_ncrs = merged_df[ncrs_scores].notna().any(axis=1)
    merged_df = merged_df[has_ncrs]
    
    print(f"\n  Merged data: {merged_df['Filename'].nunique()} subjects with NCRS data")
    
    print(f"  Before HC filtering: {merged_df['Filename'].nunique()} subjects")
    merged_df = merged_df[merged_df['Diagnosis'] != 'HC']
    print(f"  After HC filtering: {merged_df['Filename'].nunique()} patients")

    # Diagnosis breakdown
    print(f"\n  Diagnoses with NCRS data:")
    for diag in merged_df['Diagnosis'].unique():
        n = merged_df[merged_df['Diagnosis'] == diag]['Filename'].nunique()
        print(f"    {diag}: {n} subjects")
    
    if merged_df['Filename'].nunique() < min_subjects:
        print(f"[WARNING] Only {merged_df['Filename'].nunique()} subjects with NCRS data. Need at least {min_subjects}.")
        return None
    
    # ========================================================================
    # 2. ROI-WISE CORRELATIONS WITH NCRS SCORES
    # ========================================================================
    
    print(f"\n[INFO] Computing ROI-wise correlations with NCRS scores...")
    
    correlation_results = []
    rois = merged_df['ROI_Name'].unique()
    
    for roi in rois:
        roi_data = merged_df[merged_df['ROI_Name'] == roi].copy()
        
        for score in ncrs_scores:
            if score not in roi_data.columns:
                continue
                
            valid_data = roi_data[roi_data[score].notna()].copy()
            
            if len(valid_data) < min_subjects:
                continue
            
            x = valid_data['deviation_score'].values
            y = valid_data[score].values
            
            # Spearman correlation
            rho, p_val = spearmanr(x, y)
            
            # Pearson for comparison
            r, p_val_pearson = pearsonr(x, y)
            
            correlation_results.append({
                'ROI_Name': roi,
                'NCRS_Score': score,
                'N_Subjects': len(valid_data),
                'Spearman_rho': rho,
                'Spearman_p': p_val,
                'Pearson_r': r,
                'Pearson_p': p_val_pearson
            })
    
    if len(correlation_results) == 0:
        print("[WARNING] No correlations could be computed. Insufficient data.")
        return None
    
    corr_df = pd.DataFrame(correlation_results)
    
    print(f"    Computed {len(corr_df)} ROI × NCRS correlations")
    
    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        corr_df['Spearman_p'].values,
        alpha=0.05,
        method='fdr_bh'
    )
    
    corr_df['Spearman_p_corrected'] = pvals_corrected
    corr_df['Significant_FDR'] = reject
    
    n_sig = corr_df['Significant_FDR'].sum()
    print(f"    Significant correlations (FDR < 0.05): {n_sig} / {len(corr_df)}")
    
    # Save results
    corr_df_sorted = corr_df.sort_values('Spearman_p', ascending=True)
    corr_df_sorted.to_csv(f"{save_dir}/ncrs_correlations_all.csv", index=False)
    
    sig_corr = corr_df_sorted[corr_df_sorted['Significant_FDR']]
    if len(sig_corr) > 0:
        sig_corr.to_csv(f"{save_dir}/ncrs_correlations_significant.csv", index=False)
        print(f"    ✓ Saved: ncrs_correlations_significant.csv")
    
    # ========================================================================
    # 3. VISUALIZATIONS
    # ========================================================================
    
    figures_dir = f"{save_dir}/figures/ncrs_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- 3.1 Heatmap: ROIs × NCRS Scores ---
    print(f"\n[INFO] Creating NCRS correlation heatmap...")
    
    heatmap_data = corr_df.pivot(
        index='ROI_Name',
        columns='NCRS_Score',
        values='Spearman_rho'
    )
    
    heatmap_data = heatmap_data.dropna(how='all', axis=0)
    heatmap_data = heatmap_data.dropna(how='all', axis=1)
    
    # Mask non-significant
    pval_pivot = corr_df.pivot(
        index='ROI_Name',
        columns='NCRS_Score',
        values='Spearman_p_corrected'
    )
    mask = pval_pivot >= 0.05
    
    fig, ax = plt.subplots(figsize=(10, max(12, len(heatmap_data)*0.3)))
    
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        cbar_kws={'label': "Spearman's ρ"},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        mask=mask.reindex_like(heatmap_data),
        annot=False
    )
    
    ax.set_xlabel("NCRS Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Brain Region (ROI)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Brain-NCRS Correlations (Catatonia Symptoms)\n"
        f"(Only FDR-corrected significant correlations shown, n={n_sig})",
        fontsize=13, fontweight='bold', pad=15
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/heatmap_ncrs_correlations.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: heatmap_ncrs_correlations.png")
    
    # --- 3.2 Top Correlations Bar Plot ---
    if len(sig_corr) > 0:
        print(f"\n[INFO] Creating top NCRS correlations plot...")
        
        top_n = min(15, len(sig_corr))
        
        # Get top positive and negative
        top_pos = sig_corr[sig_corr['Spearman_rho'] > 0].nlargest(8, 'Spearman_rho')
        top_neg = sig_corr[sig_corr['Spearman_rho'] < 0].nsmallest(7, 'Spearman_rho')
        top_corr = pd.concat([top_pos, top_neg]).sort_values('Spearman_rho', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_corr)*0.4)))
        
        labels = [f"{row['ROI_Name'][:35]} × {row['NCRS_Score']}" 
                 for _, row in top_corr.iterrows()]
        
        colors = ['#d62728' if rho > 0 else '#1f77b4' 
                 for rho in top_corr['Spearman_rho']]
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, top_corr['Spearman_rho'], color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Spearman's ρ", fontsize=12, fontweight='bold')
        ax.set_title(
            f"Top {len(top_corr)} Brain-NCRS Correlations (FDR < 0.05)\n"
            f"Red = Positive correlation | Blue = Negative correlation",
            fontsize=13, fontweight='bold', pad=15
        )
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/ncrs_top_correlations.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: ncrs_top_correlations.png")
    
    # --- 3.3 Scatter Plots for Top Correlations ---
    if len(sig_corr) > 0:
        print(f"\n[INFO] Creating scatter plots for top NCRS correlations...")
        
        n_scatter = min(6, len(sig_corr))
        top_scatter = sig_corr.nlargest(n_scatter, key=lambda x: abs(x['Spearman_rho']))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_scatter.iterrows()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            roi = row['ROI_Name']
            score = row['NCRS_Score']
            
            plot_data = merged_df[
                (merged_df['ROI_Name'] == roi) & 
                (merged_df[score].notna())
            ].copy()
            
            # Plot by diagnosis
            for diag in plot_data['Diagnosis'].unique():
                diag_data = plot_data[plot_data['Diagnosis'] == diag]
                ax.scatter(
                    diag_data['deviation_score'],
                    diag_data[score],
                    label=diag,
                    alpha=0.6,
                    s=80,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Regression line
            from scipy.stats import linregress
            x = plot_data['deviation_score'].values
            y = plot_data[score].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'k--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Regional Deviation Score', fontsize=10, fontweight='bold')
            ax.set_ylabel(score.replace('_', ' '), fontsize=10, fontweight='bold')
            ax.set_title(
                f"{roi[:35]}\n"
                f"ρ = {row['Spearman_rho']:.3f}, p = {row['Spearman_p']:.1e}",
                fontsize=10, fontweight='bold'
            )
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for idx in range(len(top_scatter), 6):
            axes[idx].axis('off')
        
        plt.suptitle(
            "Top Brain-NCRS Correlations (Scatter Plots)",
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/ncrs_scatter_plots.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: ncrs_scatter_plots.png")
    
    # ========================================================================
    # 4. MACHINE LEARNING: PREDICT NCRS SCORES FROM REGIONAL PATTERNS
    # ========================================================================
    
    print(f"\n[INFO] Training ML models to predict NCRS scores from regional deviations...")
    
    ml_results = []
    
    for score in ncrs_scores:
        if score not in merged_df.columns:
            continue
            
        # Get subjects with this score
        score_data = merged_df[merged_df[score].notna()].copy()
        
        if score_data['Filename'].nunique() < 15:  # Need reasonable sample
            print(f"    Skipping {score}: only {score_data['Filename'].nunique()} subjects")
            continue
        
        # Pivot: rows = subjects, columns = ROIs
        pivot_ml = score_data.pivot_table(
            index='Filename',
            columns='ROI_Name',
            values='deviation_score',
            aggfunc='first'
        )
        
        # Get clinical scores
        clinical_vals = score_data.drop_duplicates('Filename').set_index('Filename')[score]
        clinical_vals = clinical_vals.reindex(pivot_ml.index)
        
        # Remove NaNs
        valid_idx = clinical_vals.notna() & pivot_ml.notna().all(axis=1)
        X = pivot_ml.loc[valid_idx].fillna(0).values
        y = clinical_vals.loc[valid_idx].values
        
        if len(y) < 10:
            print(f"    Skipping {score}: only {len(y)} complete cases")
            continue
        
        print(f"\n    [{score}] Training on {len(y)} subjects...")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ====================================================================
        # Model 1: Random Forest
        # ====================================================================
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # 5-fold CV
        cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
        
        # Train on all data for feature importance
        rf.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'ROI_Name': pivot_ml.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_importance.to_csv(
            f"{save_dir}/ncrs_feature_importance_{score}.csv",
            index=False
        )
        
        # ====================================================================
        # Model 2: Ridge Regression (interpretable coefficients)
        # ====================================================================
        
        ridge = Ridge(alpha=1.0)
        cv_scores_ridge = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
        
        # Train on all data for coefficients
        ridge.fit(X_scaled, y)
        
        ridge_coefs = pd.DataFrame({
            'ROI_Name': pivot_ml.columns,
            'Coefficient': ridge.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        ridge_coefs.to_csv(
            f"{save_dir}/ncrs_ridge_coefficients_{score}.csv",
            index=False
        )
        
        # ====================================================================
        # Model 3: Leave-One-Out CV (for small samples)
        # ====================================================================
        
        if len(y) <= 50:
            loo = LeaveOneOut()
            cv_scores_loo = cross_val_score(ridge, X_scaled, y, cv=loo, scoring='r2')
            loo_r2 = cv_scores_loo.mean()
        else:
            loo_r2 = None
        
        # Store results
        ml_results.append({
            'NCRS_Score': score,
            'N_Subjects': len(y),
            'RF_R2_Mean': cv_scores_rf.mean(),
            'RF_R2_Std': cv_scores_rf.std(),
            'Ridge_R2_Mean': cv_scores_ridge.mean(),
            'Ridge_R2_Std': cv_scores_ridge.std(),
            'LOO_R2': loo_r2,
            'Top_5_ROIs_RF': ', '.join(feature_importance.head(5)['ROI_Name'].tolist()),
            'Top_5_ROIs_Ridge': ', '.join(ridge_coefs.head(5)['ROI_Name'].tolist())
        })
        
        print(f"      Random Forest: R² = {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
        print(f"      Ridge:         R² = {cv_scores_ridge.mean():.3f} ± {cv_scores_ridge.std():.3f}")
        if loo_r2 is not None:
            print(f"      LOO CV:        R² = {loo_r2:.3f}")
    
    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        ml_df = ml_df.sort_values('Ridge_R2_Mean', ascending=False)
        ml_df.to_csv(f"{save_dir}/ncrs_prediction_performance.csv", index=False)
        
        print(f"\n  ML Prediction Summary:")
        print(f"    ✓ Saved: ncrs_prediction_performance.csv")
        print(f"    ✓ Saved: ncrs_feature_importance_*.csv for each score")
        print(f"    ✓ Saved: ncrs_ridge_coefficients_*.csv for each score")
    
    # ========================================================================
    # 5. VISUALIZE FEATURE IMPORTANCE
    # ========================================================================
    
    if ml_results:
        print(f"\n[INFO] Creating feature importance plots...")
        
        for score in [r['NCRS_Score'] for r in ml_results]:
            # Load feature importance
            try:
                feat_imp = pd.read_csv(f"{save_dir}/ncrs_feature_importance_{score}.csv")
                top_feats = feat_imp.head(20)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                y_pos = np.arange(len(top_feats))
                ax.barh(y_pos, top_feats['Importance'], alpha=0.7, 
                       edgecolor='black', linewidth=1.5, color='#2E86AB')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_feats['ROI_Name'], fontsize=9)
                ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
                ax.set_title(
                    f'Top 20 Brain Regions Predicting {score.replace("_", " ")}\n'
                    f'(Random Forest Feature Importance)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{figures_dir}/feature_importance_{score}.png",
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
            except:
                continue
        
        print(f"    ✓ Saved feature importance plots")
    
    print(f"\n[INFO] NCRS analysis complete!")
    print(f"  Results saved to: {save_dir}/")
    
    return corr_df, ml_results

def create_paper_figure_significant_correlations(
    clinical_correlations_csv,
    output_dir,
    figsize=(14, 10),
    dpi=300
):
    """
    Create paper-ready figure showing only significant correlations.
    
    Args:
        clinical_correlations_csv: Path to clinical_correlations_significant.csv
        output_dir: Where to save figures
        figsize: Figure size (width, height)
        dpi: Resolution for saving
    
    Returns:
        None (saves figures to disk)
    """
    
    # Load data
    sig_corr = pd.read_csv(clinical_correlations_csv)
    
    print(f"\n[INFO] Creating paper figure from {len(sig_corr)} significant correlations")
    
    if len(sig_corr) == 0:
        print("[WARNING] No significant correlations found! Cannot create figure.")
        return
    
    # ========================================================================
    # FIGURE 1: Heatmap of Significant Correlations ONLY
    # ========================================================================
    
    print("\n[1/4] Creating focused heatmap (significant only)...")
    
    # Pivot to matrix format
    heatmap_data = sig_corr.pivot(
        index='ROI_Name',
        columns='Clinical_Score',
        values='Spearman_rho'
    )
    
    # Drop empty rows/columns
    heatmap_data = heatmap_data.dropna(how='all', axis=0)
    heatmap_data = heatmap_data.dropna(how='all', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        cbar_kws={'label': "Spearman's ρ", 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot=False,
        fmt='.2f'
    )
    
    ax.set_xlabel("Clinical Symptom Score", fontsize=13, fontweight='bold')
    ax.set_ylabel("Brain Region (ROI)", fontsize=13, fontweight='bold')
    ax.set_title(
        f"Significant Brain-Clinical Correlations (n={len(sig_corr)})\n"
        f"FDR-corrected, q < 0.05",
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "paper_figure_heatmap_significant.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")
    
    # ========================================================================
    # FIGURE 2: Bar Plot - All Significant Correlations
    # ========================================================================
    
    print("\n[2/4] Creating comprehensive barplot...")
    
    # Sort by absolute correlation strength
    sig_corr_sorted = sig_corr.copy()
    sig_corr_sorted['abs_rho'] = sig_corr_sorted['Spearman_rho'].abs()
    sig_corr_sorted = sig_corr_sorted.sort_values('abs_rho', ascending=True)
    
    # Create figure (height scales with number of correlations)
    fig_height = max(8, len(sig_corr_sorted) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Create labels
    labels = [
        f"{row['ROI_Name'][:40]} × {row['Clinical_Score']}" 
        for _, row in sig_corr_sorted.iterrows()
    ]
    
    # Color by direction
    colors = [
        '#d62728' if rho > 0 else '#1f77b4' 
        for rho in sig_corr_sorted['Spearman_rho']
    ]
    
    # Plot
    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, 
        sig_corr_sorted['Spearman_rho'], 
        color=colors, 
        alpha=0.7, 
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Spearman's ρ", fontsize=12, fontweight='bold')
    ax.set_title(
        f"All Significant Brain-Clinical Correlations (FDR < 0.05)\n"
        f"Red = Positive | Blue = Negative | n = {len(sig_corr_sorted)}",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add effect size regions
    ax.axvspan(0.3, 0.6, alpha=0.05, color='red', label='Moderate-Strong (|ρ| > 0.3)')
    ax.axvspan(-0.6, -0.3, alpha=0.05, color='blue')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "paper_figure_barplot_all_significant.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")
    
    # ========================================================================
    # FIGURE 3: Grouped by Clinical Score
    # ========================================================================
    
    print("\n[3/4] Creating grouped barplot by clinical score...")
    
    # Count significant correlations per clinical score
    score_counts = sig_corr['Clinical_Score'].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(score_counts) * 0.4)))
    
    colors_grouped = ['#2E86AB' if count > 5 else '#A23B72' for count in score_counts]
    
    y_pos = np.arange(len(score_counts))
    ax.barh(y_pos, score_counts.values, color=colors_grouped, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(score_counts.index, fontsize=11)
    ax.set_xlabel('Number of Significant ROIs', fontsize=12, fontweight='bold')
    ax.set_title(
        'Brain-Wide Impact of Clinical Symptoms\n'
        '(How many brain regions correlate with each symptom?)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (score, count) in enumerate(score_counts.items()):
        ax.text(count + 0.3, i, str(count), va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "paper_figure_grouped_by_score.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")
    
    # ========================================================================
    # FIGURE 4: Top 10 Strongest Correlations with Details
    # ========================================================================
    
    print("\n[4/4] Creating detailed figure for top 10 correlations...")
    
    # Get top 10 by absolute strength
    top_10 = sig_corr.copy()
    top_10['abs_rho'] = top_10['Spearman_rho'].abs()
    top_10 = top_10.nlargest(10, 'abs_rho')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create detailed labels with p-values
    labels_detailed = []
    for _, row in top_10.sort_values('Spearman_rho', ascending=True).iterrows():
        roi = row['ROI_Name'][:35]
        score = row['Clinical_Score']
        rho = row['Spearman_rho']
        p = row['Spearman_p']
        n = row['N_Subjects']
        
        # Format p-value
        if p < 0.0001:
            p_str = "p < 0.0001"
        elif p < 0.001:
            p_str = f"p = {p:.4f}"
        else:
            p_str = f"p = {p:.3f}"
        
        labels_detailed.append(f"{roi} × {score}\n(ρ={rho:.3f}, {p_str}, n={n})")
    
    y_pos = np.arange(len(labels_detailed))
    colors_top = [
        '#d62728' if rho > 0 else '#1f77b4' 
        for rho in top_10.sort_values('Spearman_rho')['Spearman_rho']
    ]
    
    bars = ax.barh(y_pos, top_10.sort_values('Spearman_rho')['Spearman_rho'], 
                   color=colors_top, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_detailed, fontsize=9)
    ax.set_xlabel("Spearman's ρ", fontsize=13, fontweight='bold')
    ax.set_title(
        "Top 10 Strongest Brain-Clinical Correlations\n"
        "FDR-corrected significant findings (q < 0.05)",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add effect size reference lines
    ax.axvline(0.3, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(-0.3, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(0.31, len(labels_detailed)-0.5, 'Moderate', fontsize=8, alpha=0.6)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "paper_figure_top10_detailed.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal significant correlations: {len(sig_corr)}")
    print(f"  Positive correlations: {(sig_corr['Spearman_rho'] > 0).sum()}")
    print(f"  Negative correlations: {(sig_corr['Spearman_rho'] < 0).sum()}")
    
    print(f"\nEffect size distribution:")
    print(f"  Small (|ρ| < 0.3):    {(sig_corr['Spearman_rho'].abs() < 0.3).sum()}")
    print(f"  Moderate (|ρ| ≥ 0.3): {(sig_corr['Spearman_rho'].abs() >= 0.3).sum()}")
    print(f"  Strong (|ρ| ≥ 0.5):   {(sig_corr['Spearman_rho'].abs() >= 0.5).sum()}")
    
    print(f"\nClinical scores with most correlations:")
    top_scores = sig_corr['Clinical_Score'].value_counts().head(5)
    for score, count in top_scores.items():
        print(f"  {score:20s}: {count} ROIs")
    
    print(f"\nROIs with most correlations:")
    top_rois = sig_corr['ROI_Name'].value_counts().head(5)
    for roi, count in top_rois.items():
        print(f"  {roi[:40]:40s}: {count} scores")
    
    print(f"\nStrongest correlations:")
    strongest = sig_corr.nlargest(3, key=lambda x: abs(x['Spearman_rho']))
    for _, row in strongest.iterrows():
        print(f"  {row['ROI_Name'][:35]:35s} × {row['Clinical_Score']:20s}: ρ = {row['Spearman_rho']:+.3f}")
    
    print("\n" + "="*80)
    print("PAPER FIGURES COMPLETE!")
    print("="*80)
    print(f"\nGenerated 4 publication-ready figures:")
    print(f"  1. paper_figure_heatmap_significant.png")
    print(f"  2. paper_figure_barplot_all_significant.png")
    print(f"  3. paper_figure_grouped_by_score.png")
    print(f"  4. paper_figure_top10_detailed.png")


def create_combined_paper_figure(
    clinical_correlations_csv,
    output_dir,
    figsize=(18, 12),
    dpi=300
):
    """
    Create ONE comprehensive figure with 4 panels (for paper).
    
    Perfect for: Main figure in paper showing all aspects
    """
    
    sig_corr = pd.read_csv(clinical_correlations_csv)
    
    if len(sig_corr) == 0:
        print("[WARNING] No significant correlations found!")
        return
    
    print(f"\n[INFO] Creating combined 4-panel figure...")
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # PANEL A: Heatmap
    # ========================================================================
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    heatmap_data = sig_corr.pivot(
        index='ROI_Name',
        columns='Clinical_Score',
        values='Spearman_rho'
    ).dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        cbar_kws={'label': "Spearman's ρ"},
        linewidths=0.5,
        linecolor='white',
        ax=ax1
    )
    
    ax1.set_xlabel("Clinical Score", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Brain Region", fontsize=10, fontweight='bold')
    ax1.set_title("A) Correlation Matrix", fontsize=12, fontweight='bold', loc='left')
    ax1.tick_params(axis='both', labelsize=8)
    
    # ========================================================================
    # PANEL B: Bar plot by clinical score
    # ========================================================================
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    score_counts = sig_corr['Clinical_Score'].value_counts().sort_values(ascending=True)
    colors_b = ['#2E86AB' if count > 5 else '#A23B72' for count in score_counts]
    
    y_pos = np.arange(len(score_counts))
    ax2.barh(y_pos, score_counts.values, color=colors_b, alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(score_counts.index, fontsize=9)
    ax2.set_xlabel('Number of ROIs', fontsize=10, fontweight='bold')
    ax2.set_title("B) Impact by Clinical Score", fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='x', alpha=0.3)
    
    # ========================================================================
    # PANEL C: Top 15 correlations
    # ========================================================================
    
    ax3 = fig.add_subplot(gs[1, :])
    
    top_15 = sig_corr.copy()
    top_15['abs_rho'] = top_15['Spearman_rho'].abs()
    top_15 = top_15.nlargest(15, 'abs_rho').sort_values('Spearman_rho', ascending=True)
    
    labels_c = [
        f"{row['ROI_Name'][:30]} × {row['Clinical_Score']}" 
        for _, row in top_15.iterrows()
    ]
    
    colors_c = ['#d62728' if rho > 0 else '#1f77b4' for rho in top_15['Spearman_rho']]
    
    y_pos = np.arange(len(labels_c))
    ax3.barh(y_pos, top_15['Spearman_rho'], color=colors_c, alpha=0.7, edgecolor='black')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_c, fontsize=9)
    ax3.set_xlabel("Spearman's ρ", fontsize=11, fontweight='bold')
    ax3.set_title("C) Top 15 Strongest Correlations", fontsize=12, fontweight='bold', loc='left')
    ax3.axvline(0, color='black', linewidth=0.8)
    ax3.grid(axis='x', alpha=0.3)
    
    # ========================================================================
    # Overall title
    # ========================================================================
    
    fig.suptitle(
        f"Significant Brain-Clinical Correlations (n={len(sig_corr)}, FDR < 0.05)",
        fontsize=15, fontweight='bold', y=0.98
    )
    
    output_path = os.path.join(output_dir, "paper_figure_combined_4panel.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved combined figure: {output_path}")