"""
Deviation Score Utilities for Conditional VAE (CVAE)

Complete analysis suite with all functions needed for CVAE-based
normative modeling analysis.

For standard VAE, use dev_scores_utils.py instead.
"""

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

print("[INFO] Loading dev_scores_utils_CVAE.py - Conditional VAE version")

# ============================================================================
# CORE DEVIATION CALCULATION - CVAE VERSION
# ============================================================================

def calculate_deviations_cvae(normative_models, data_tensor, conditions_tensor, 
                               norm_diagnosis, annotations_df, device="cuda", 
                               roi_names=None):
    """
    Calculate deviation scores using bootstrap CVAE models.
    
    CVAE VERSION: Requires conditions_tensor parameter.
    
    Args:
        normative_models: List of trained ConditionalVAE_2D models
        data_tensor: Tensor of clinical data (all subjects)
        conditions_tensor: Tensor of conditions [Age, Sex, IQR, Dataset_onehot]
        norm_diagnosis: Normative diagnosis group (e.g., 'HC')
        annotations_df: DataFrame with metadata
        device: Computing device ('cuda' or 'cpu')
        roi_names: Optional list of ROI names
    
    Returns:
        results_df: DataFrame with deviation scores normalized relative to HC
    """
    
    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
    # Alignment check
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch: {total_subjects} samples vs {len(annotations_df)} annotations")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        annotations_df = annotations_df.iloc[valid_indices].reset_index(drop=True)
        data_tensor = data_tensor[:len(annotations_df)]
        conditions_tensor = conditions_tensor[:len(annotations_df)]
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Initialize arrays
    all_recon_errors = np.zeros((total_subjects, total_models))
    all_kl_divs = np.zeros((total_subjects, total_models))
    all_z_scores = np.zeros((total_subjects, data_tensor.shape[1], total_models))
    
    # Process each bootstrap model
    print(f"[INFO] Processing {total_models} bootstrap CVAE models...")
    
    for i, model in enumerate(normative_models):
        model.eval()
        model.to(device)
        with torch.no_grad():
            batch_data = data_tensor.to(device)
            batch_conditions = conditions_tensor.to(device)
            
            # CVAE forward pass with conditions
            recon, mu, log_var = model(batch_data, batch_conditions)
            
            # Reconstruction error (MSE per subject)
            recon_error = torch.mean((batch_data - recon) ** 2, dim=1).cpu().numpy()
            all_recon_errors[:, i] = recon_error
            
            # KL divergence (per subject)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()
            all_kl_divs[:, i] = kl_div
            
            # Region-wise squared errors
            z_scores = ((batch_data - recon) ** 2).cpu().numpy()
            all_z_scores[:, :, i] = z_scores
        
        torch.cuda.empty_cache()
    
    print(f"[INFO] Finished processing models")
    
    # Average across bootstrap models
    mean_recon_error = np.mean(all_recon_errors, axis=1)
    std_recon_error = np.std(all_recon_errors, axis=1)
    mean_kl_div = np.mean(all_kl_divs, axis=1)
    std_kl_div = np.std(all_kl_divs, axis=1)
    mean_region_z_scores = np.mean(all_z_scores, axis=2)
    
    # Create base dataframe
    results_df = annotations_df[["Filename", "Diagnosis", "Age", "Sex", "Dataset"]].copy()
    results_df["reconstruction_error"] = mean_recon_error
    results_df["reconstruction_error_std"] = std_recon_error
    results_df["kl_divergence"] = mean_kl_div
    results_df["kl_divergence_std"] = std_kl_div
    
    # Add region-wise z-scores
    if roi_names is not None and len(roi_names) == mean_region_z_scores.shape[1]:
        column_names = [f"{name}_z_score" for name in roi_names]
    else:
        column_names = [f"region_{i}_z_score" for i in range(mean_region_z_scores.shape[1])]
    
    new_columns = pd.DataFrame(mean_region_z_scores, columns=column_names)
    results_df = pd.concat([results_df, new_columns], axis=1)
    
    # Normalization relative to HC
    print(f"\n[INFO] Normalizing deviation scores relative to {norm_diagnosis}...")
    
    hc_mask = annotations_df["Diagnosis"] == norm_diagnosis
    n_hc = hc_mask.sum()
    
    if n_hc == 0:
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
    
    # Z-score normalization
    z_norm_recon = (mean_recon_error - recon_mean_hc) / (recon_std_hc + 1e-8)
    z_norm_kl = (mean_kl_div - kl_mean_hc) / (kl_std_hc + 1e-8)
    results_df["deviation_score_zscore"] = (z_norm_recon + z_norm_kl) / 2
    
    # Percentile-based scoring
    recon_percentiles = np.array([
        scipy_stats.percentileofscore(hc_recon, x, kind='rank') / 100 
        for x in mean_recon_error
    ])
    kl_percentiles = np.array([
        scipy_stats.percentileofscore(hc_kl, x, kind='rank') / 100 
        for x in mean_kl_div
    ])
    results_df["deviation_score_percentile"] = (recon_percentiles + kl_percentiles) / 2
    
    # Robust min-max
    min_recon_hc = np.percentile(hc_recon, 5)
    max_recon_hc = np.percentile(hc_recon, 95)
    min_kl_hc = np.percentile(hc_kl, 5)
    max_kl_hc = np.percentile(hc_kl, 95)
    
    norm_recon = np.clip(mean_recon_error, min_recon_hc, max_recon_hc)
    norm_recon = (norm_recon - min_recon_hc) / (max_recon_hc - min_recon_hc + 1e-8)
    
    norm_kl = np.clip(mean_kl_div, min_kl_hc, max_kl_hc)
    norm_kl = (norm_kl - min_kl_hc) / (max_kl_hc - min_kl_hc + 1e-8)
    
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2
    
    # Summary
    print(f"\n[INFO] Deviation Score Summary by Diagnosis:")
    print("="*60)
    for diagnosis in sorted(results_df["Diagnosis"].unique()):
        diag_mask = results_df["Diagnosis"] == diagnosis
        n = diag_mask.sum()
        mean_score = results_df[diag_mask]["deviation_score"].mean()
        std_score = results_df[diag_mask]["deviation_score"].std()
        print(f"{diagnosis:10s} (n={n:3d}): score={mean_score:.3f}±{std_score:.3f}")
    print("="*60)
    
    return results_df


# ============================================================================
# ADDITIONAL DEVIATION METRICS - CVAE VERSION
# ============================================================================

def calculate_reconstruction_deviation_cvae(model, data, conditions, device='cuda'):
    """D_MSE - Reconstruction-based deviation for CVAE"""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        cond_tensor = torch.FloatTensor(conditions).to(device)
        reconstructed, _, _ = model(data_tensor, cond_tensor)
        mse = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
    return mse.cpu().numpy()


def calculate_kl_divergence_deviation_cvae(model, data, conditions, device='cuda'):
    """D_KL - KL Divergence as deviation metric for CVAE"""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        cond_tensor = torch.FloatTensor(conditions).to(device)
        _, mu, logvar = model(data_tensor, cond_tensor)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_div.cpu().numpy()


def compute_hc_latent_stats_cvae(model, hc_data, hc_conditions, device='cuda'):
    """Compute HC population statistics in latent space for Aguila method (CVAE)"""
    model.eval()
    with torch.no_grad():
        hc_tensor = torch.FloatTensor(hc_data).to(device)
        hc_cond_tensor = torch.FloatTensor(hc_conditions).to(device)
        _, mu, _ = model(hc_tensor, hc_cond_tensor)
        hc_mean = mu.mean(dim=0)
        hc_std = mu.std(dim=0)
    return {
        'mean': hc_mean.cpu().numpy(),
        'std': hc_std.cpu().numpy()
    }


def calculate_latent_deviation_aguila_cvae(model, data, conditions, hc_latent_stats, device='cuda'):
    """D_L - Latent-based deviation (Aguila et al. 2022) for CVAE"""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        cond_tensor = torch.FloatTensor(conditions).to(device)
        _, mu, logvar = model(data_tensor, cond_tensor)
        
        sigma_kj = torch.exp(0.5 * logvar)
        hc_mean = torch.FloatTensor(hc_latent_stats['mean']).to(device)
        hc_std = torch.FloatTensor(hc_latent_stats['std']).to(device)
        
        numerator = torch.abs(mu - hc_mean)
        denominator = torch.sqrt(hc_std**2 + sigma_kj**2)
        per_dim_deviations = numerator / denominator
        deviation_scores = torch.mean(per_dim_deviations, dim=1)
    
    return deviation_scores.cpu().numpy(), per_dim_deviations.cpu().numpy()


def calculate_combined_deviation(recon_dev, kl_dev, alpha=0.7, beta=0.3):
    """D_combined - Weighted combination (no model needed)"""
    recon_norm = (recon_dev - recon_dev.min()) / (recon_dev.max() - recon_dev.min() + 1e-8)
    kl_norm = (kl_dev - kl_dev.min()) / (kl_dev.max() - kl_dev.min() + 1e-8)
    return alpha * recon_norm + beta * kl_norm


# ============================================================================
# VISUALIZATION - CVAE VERSION
# ============================================================================

def visualize_embeddings_multiple_cvae(normative_models, data_tensor, conditions_tensor,
                                       annotations_df, columns_to_plot=None, 
                                       device="cuda", figsize=(12, 10)):
    """Visualizes the latent space colored by metadata for CVAE"""
    
    total_subjects = data_tensor.shape[0]
    
    # Alignment check
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch: {total_subjects} vs {len(annotations_df)}")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        annotations_df = annotations_df.iloc[valid_indices].reset_index(drop=True)
        data_tensor = data_tensor[:len(annotations_df)]
        conditions_tensor = conditions_tensor[:len(annotations_df)]
    
    model = normative_models[0]
    model.eval()
    model.to(device)
    
    all_embeddings = []
    batch_size = 16
    
    data_loader = DataLoader(
        TensorDataset(data_tensor, conditions_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("Generating CVAE embeddings...")
    with torch.no_grad():
        for batch_data, batch_cond in data_loader:
            batch_data = batch_data.to(device)
            batch_cond = batch_cond.to(device)
            _, mu, _ = model(batch_data, batch_cond)
            all_embeddings.append(mu.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    if columns_to_plot is None:
        columns_to_plot = []
        for col in annotations_df.columns:
            if annotations_df[col].dtype == 'object' or annotations_df[col].nunique() <= 20:
                columns_to_plot.append(col)
    
    results = {}
    custom_palette = ["#125E8A", "#3E885B", "#BEDCFE", "#2F4B26", "#A67DB8", "#160C28"]
    
    for col in columns_to_plot:
        if col not in annotations_df.columns:
            continue
        
        plot_df = annotations_df[[col]].copy()
        plot_df["umap_1"] = umap_embeddings[:, 0]
        plot_df["umap_2"] = umap_embeddings[:, 1]
        plot_df = plot_df.dropna(subset=[col])
        
        plt.figure(figsize=figsize)
        unique_values = plot_df[col].nunique()
        
        if unique_values <= len(custom_palette):
            palette = custom_palette[:unique_values]
        else:
            palette = sns.color_palette("viridis", n_colors=unique_values)
        
        if plot_df[col].dtype in ['object', 'category'] or unique_values <= 20:
            sns.scatterplot(data=plot_df, x="umap_1", y="umap_2", hue=col,
                          palette=palette, s=40, alpha=0.7)
        else:
            scatter = plt.scatter(plot_df["umap_1"], plot_df["umap_2"],
                                c=plot_df[col], cmap='viridis', s=40, alpha=0.7)
            plt.colorbar(scatter, label=col)
        
        plt.title(f"CVAE UMAP - Colored by {col}", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=13)
        plt.ylabel("UMAP 2", fontsize=13)
        
        if plt.gca().get_legend() is not None:
            plt.legend(title=col, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        results[col] = (plt.gcf(), plot_df.copy())
        plt.show()
    
    return results


def save_latent_visualizations(results, output_dir, dpi=300):
    """Save latent visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    for col_name, (fig, plot_df) in results.items():
        clean_name = col_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        fig.savefig(
            os.path.join(output_dir, f"cvae_umap_{clean_name}.png"),
            dpi=dpi, bbox_inches='tight', facecolor='white'
        )
        print(f"Saved CVAE visualization for '{col_name}'")


# ============================================================================
# STATISTICAL HELPERS - SHARED (NO MODEL INTERACTION)
# ============================================================================

def calculate_cliffs_delta(x, y):
    """Calculate Cliff's Delta effect size"""
    x = np.asarray(x)
    y = np.asarray(y)
    dominance = np.zeros((len(x), len(y)))
    for i, x_i in enumerate(x):
        dominance[i] = np.sign(x_i - y)
    delta = np.mean(dominance)
    return delta


def bootstrap_cliffs_delta_ci(data1, data2, num_bootstraps=100, ci_level=0.95):
    """Bootstrap confidence interval for Cliff's Delta"""
    bootstrapped_deltas = []
    n1 = len(data1)
    n2 = len(data2)

    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    original_delta = calculate_cliffs_delta(data1, data2)
    if np.isnan(original_delta):
        return np.nan, np.nan, np.nan

    for _ in range(num_bootstraps):
        sample1 = np.random.choice(data1, n1, replace=True)
        sample2 = np.random.choice(data2, n2, replace=True)
        delta = calculate_cliffs_delta(sample1, sample2)
        if not np.isnan(delta):
            bootstrapped_deltas.append(delta)

    if not bootstrapped_deltas:
        return np.nan, np.nan, np.nan

    sorted_deltas = np.sort(bootstrapped_deltas)
    lower_bound_idx = int(num_bootstraps * (1 - ci_level) / 2)
    upper_bound_idx = int(num_bootstraps * (1 - (1 - ci_level) / 2))

    lower_bound = sorted_deltas[lower_bound_idx] if lower_bound_idx < len(sorted_deltas) else np.nan
    upper_bound = sorted_deltas[upper_bound_idx] if upper_bound_idx < len(sorted_deltas) else np.nan
    
    count_extreme = np.sum(np.abs(sorted_deltas) >= np.abs(original_delta))
    p_value = count_extreme / num_bootstraps
    p_value = max(p_value, 1.0 / num_bootstraps)
    p_value = min(p_value, 1.0)

    return lower_bound, upper_bound, p_value


# ============================================================================
# ROI NAME FORMATTING - SHARED
# ============================================================================

def format_roi_name_for_plotting(original_roi_name: str, atlas_name_from_config=None) -> str:
    """Format ROI name for plotting: [V] RightHippocampus (Neurom)"""
    
    atlas_abbreviations = {
        "cobra": "[C]", "lpba40": "[L]", "neuromorphometrics": "[N]",
        "Neurom": "[N]", "suit": "[S]", "SUIT": "[S]",
        "thalamic_nuclei": "[TN]", "thalamus": "[T]",
        "aal3": "[A]", "AAL3": "[AAL3]", "ibsr": "[I]", "IBSR": "[I]",
        "schaefer100": "[S100]", "Sch100": "[S100]",
        "schaefer200": "[S200]", "Sch200": "[S200]",
        "aparc_dk40": "[DK]", "DK40": "[DK]",
        "aparc_destrieux": "[DES]", "Destrieux": "[DES]",
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


def format_roi_names_list_for_plotting(roi_names_list: List[str], atlas_name_from_config=None) -> List[str]:
    """Format list of ROI names"""
    return [format_roi_name_for_plotting(name, atlas_name_from_config) for name in roi_names_list]


# ============================================================================
# IMPORT SHARED PLOTTING/ANALYSIS FUNCTIONS FROM ORIGINAL
# ============================================================================

# These don't interact with models, so they work for both VAE and CVAE
from utils.dev_scores_utils import (
    plot_all_deviation_metrics_errorbar,
    plot_deviation_distributions,
    create_diagnosis_palette,
    calculate_group_pvalues,
    run_analysis_with_options,
    analyze_regional_deviations,
    create_corrected_correlation_heatmap,
)

print("[INFO] Loaded all CVAE deviation utilities")
print("[INFO] For standard VAE, use dev_scores_utils.py instead")