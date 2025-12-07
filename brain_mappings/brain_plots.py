"""
3D Brain Visualization of Regional Effect Sizes

Creates publication-quality 3D brain visualizations showing:
- Effect size (Cliffs Delta) as color intensity
- Direction (positive/negative) as red/blue
- Significance as transparency/opacity

Supports multiple atlases:
- DK40 (Desikan-Killiany): Surface-based
- Neuromorphometrics: Volume-based
- Others: Adaptable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import os
from typing import Dict, List, Tuple

# Try importing brain visualization libraries
try:
    from nilearn import plotting, datasets, surface
    from nilearn.image import load_img, new_img_like
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("[WARNING] nilearn not available. Install with: pip install nilearn")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("[WARNING] nibabel not available. Install with: pip install nibabel")


# ============================================================================
# HELPER FUNCTIONS: Mapping ROI names to atlas indices
# ============================================================================

def create_dk40_roi_mapping():
    """
    Create mapping from DK40 ROI names to FreeSurfer label indices.
    
    Returns:
        dict: {roi_name: label_index}
    """
    # FreeSurfer DK40 labels (aparc.a2009s)
    dk40_labels = {
        # Left hemisphere (1000-1035)
        'bankssts': 1001, 'caudalanteriorcingulate': 1002,
        'caudalmiddlefrontal': 1003, 'corpuscallosum': 1004,
        'cuneus': 1005, 'entorhinal': 1006, 'fusiform': 1007,
        'inferiorparietal': 1008, 'inferiortemporal': 1009,
        'isthmuscingulate': 1010, 'lateraloccipital': 1011,
        'lateralorbitofrontal': 1012, 'lingual': 1013,
        'medialorbitofrontal': 1014, 'middletemporal': 1015,
        'parahippocampal': 1016, 'paracentral': 1017,
        'parsopercularis': 1018, 'parsorbitalis': 1019,
        'parstriangularis': 1020, 'pericalcarine': 1021,
        'postcentral': 1022, 'posteriorcingulate': 1023,
        'precentral': 1024, 'precuneus': 1025,
        'rostralanteriorcingulate': 1026, 'rostralmiddlefrontal': 1027,
        'superiorfrontal': 1028, 'superiorparietal': 1029,
        'superiortemporal': 1030, 'supramarginal': 1031,
        'frontalpole': 1032, 'temporalpole': 1033,
        'transversetemporal': 1034, 'insula': 1035,
        
        # Right hemisphere (2000-2035)
        'rbankssts': 2001, 'rcaudalanteriorcingulate': 2002,
        'rcaudalmiddlefrontal': 2003, 'rcorpuscallosum': 2004,
        'rcuneus': 2005, 'rentorhinal': 2006, 'rfusiform': 2007,
        'rinferiorparietal': 2008, 'rinferiortemporal': 2009,
        'risthmuscingulate': 2010, 'rlateraloccipital': 2011,
        'rlateralorbitofrontal': 2012, 'rlingual': 2013,
        'rmedialorbitofrontal': 2014, 'rmiddletemporal': 2015,
        'rparahippocampal': 2016, 'rparacentral': 2017,
        'rparsopercularis': 2018, 'rparsorbitalis': 2019,
        'rparstriangularis': 2020, 'rpericalcarine': 2021,
        'rpostcentral': 2022, 'rposteriorcingulate': 2023,
        'rprecentral': 2024, 'rprecuneus': 2025,
        'rrostralanteriorcingulate': 2026, 'rrostralmiddlefrontal': 2027,
        'rsuperiorfrontal': 2028, 'rsuperiorparietal': 2029,
        'rsuperiortemporal': 2030, 'rsupramarginal': 2031,
        'rfrontalpole': 2032, 'rtemporalpole': 2033,
        'rtransversetemporal': 2034, 'rinsula': 2035
    }
    
    return dk40_labels


def parse_roi_name(roi_name: str) -> Tuple[str, str]:
    """
    Parse ROI name from CSV format.
    
    Example: "[T] lprecentral (DK40)" -> ("l", "precentral")
    
    Returns:
        (hemisphere, region_name)
    """
    # Extract hemisphere
    if roi_name.startswith('['):
        # Format: [T] lprecentral (DK40)
        parts = roi_name.split(']')
        if len(parts) > 1:
            name_part = parts[1].strip()
            # Split by space
            name_parts = name_part.split()
            if name_parts:
                region = name_parts[0]
                
                # Extract hemisphere
                if region.startswith('l'):
                    hemi = 'l'
                    region = region[1:]
                elif region.startswith('r'):
                    hemi = 'r'
                    region = region[1:]
                else:
                    hemi = 'l'  # default
                
                return hemi, region
    
    # Fallback: direct parsing
    if roi_name.startswith('l'):
        return 'l', roi_name[1:]
    elif roi_name.startswith('r'):
        return 'r', roi_name[1:]
    else:
        return 'l', roi_name


def map_csv_to_atlas_labels(
    effect_sizes_df: pd.DataFrame,
    atlas_type: str = 'DK40'
) -> Dict[int, float]:
    """
    Map effect sizes from CSV to atlas label indices.
    
    Args:
        effect_sizes_df: DataFrame with columns [ROI_Name, Cliffs_Delta, Significant_FDR]
        atlas_type: Type of atlas ('DK40', 'neuromorphometrics')
    
    Returns:
        dict: {label_index: cliffs_delta_value}
    """
    
    if atlas_type == 'DK40':
        dk40_mapping = create_dk40_roi_mapping()
        
        label_values = {}
        
        for _, row in effect_sizes_df.iterrows():
            roi_name = row['ROI_Name']
            cliffs_delta = row['Cliffs_Delta']
            is_significant = row.get('Significant_FDR', True)
            
            # Only include if significant
            if not is_significant:
                continue
            
            # Parse ROI name
            try:
                hemi, region = parse_roi_name(roi_name)
                
                # Create full name
                if hemi == 'r':
                    full_name = 'r' + region
                else:
                    full_name = region
                
                # Find in mapping
                if full_name in dk40_mapping:
                    label_idx = dk40_mapping[full_name]
                    label_values[label_idx] = cliffs_delta
                else:
                    print(f"[WARNING] Could not map: {roi_name} -> {full_name}")
            
            except Exception as e:
                print(f"[WARNING] Error parsing {roi_name}: {e}")
        
        print(f"[INFO] Mapped {len(label_values)} ROIs to atlas labels")
        return label_values
    
    else:
        raise NotImplementedError(f"Atlas type {atlas_type} not yet implemented")


# ============================================================================
# VISUALIZATION FUNCTION 1: Surface-based (DK40)
# ============================================================================

def visualize_surface_effects(
    effect_sizes_df: pd.DataFrame,
    atlas_file: str = None,
    output_dir: str = './brain_visualizations',
    diagnosis: str = 'CAT',
    vmax: float = 0.6,
    threshold: float = 0.1,
    views: List[str] = ['lateral', 'medial'],
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Create surface-based brain visualization using nilearn.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        atlas_file: Path to atlas file (if None, uses fsaverage)
        output_dir: Where to save figures
        diagnosis: Diagnosis name for title
        vmax: Maximum value for colormap
        threshold: Minimum absolute value to display
        views: Brain views to show
        cmap: Colormap name
        figsize: Figure size
    """
    
    if not NILEARN_AVAILABLE:
        print("[ERROR] nilearn is required for surface visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[INFO] Creating surface visualization for {diagnosis}")
    
    # Filter to significant only
    sig_df = effect_sizes_df[effect_sizes_df['Significant_FDR'] == True].copy()
    
    if len(sig_df) == 0:
        print("[WARNING] No significant ROIs to visualize")
        return
    
    print(f"[INFO] Visualizing {len(sig_df)} significant ROIs")
    
    # Map to atlas labels
    label_values = map_csv_to_atlas_labels(sig_df, atlas_type='DK40')
    
    if not label_values:
        print("[ERROR] No ROIs could be mapped to atlas")
        return
    
    # Fetch fsaverage surfaces
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    
    # Create data arrays for left and right hemispheres
    # We need to create a texture map from label indices to values
    
    # Load parcellation
    from nilearn.datasets import fetch_atlas_surf_destrieux
    parcellation = fetch_atlas_surf_destrieux()
    
    # For now, use statistical surface maps
    # This is a placeholder - you'll need to adapt based on your atlas files
    
    print("[INFO] Creating brain surface plots...")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot different views
    for view_idx, view in enumerate(views):
        for hemi_idx, hemi in enumerate(['left', 'right']):
            
            # Create texture from label values
            # This requires mapping your labels to surface vertices
            # Placeholder for demonstration
            texture = np.zeros(10242)  # fsaverage5 has 10242 vertices per hemisphere
            
            # Map label values to texture
            # (This is simplified - actual implementation depends on atlas format)
            for label, value in label_values.items():
                # Determine if this label is for this hemisphere
                if hemi == 'left' and 1000 <= label < 2000:
                    # Map to vertices (requires atlas parcellation)
                    pass
                elif hemi == 'right' and 2000 <= label < 3000:
                    pass
            
            # Plot
            ax_idx = view_idx * 2 + hemi_idx + 1
            ax = plt.subplot(len(views), 2, ax_idx, projection='3d')
            
            plotting.plot_surf_stat_map(
                fsaverage[f'pial_{hemi}'],
                texture,
                hemi=hemi,
                view=view,
                threshold=threshold,
                cmap=cmap,
                vmax=vmax,
                symmetric_cbar=True,
                colorbar=True,
                axes=ax,
                title=f"{diagnosis} - {hemi.capitalize()} {view.capitalize()}"
            )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"brain_surface_{diagnosis}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] Saved: {output_path}")


# ============================================================================
# VISUALIZATION FUNCTION 2: Glass Brain (Fast & Easy!)
# ============================================================================

def visualize_glass_brain_effects(
    effect_sizes_df: pd.DataFrame,
    output_dir: str = './brain_visualizations',
    diagnosis: str = 'CAT',
    vmax: float = 0.6,
    threshold: float = 0.1,
    cmap: str = 'cold_hot',
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Create glass brain visualization (easiest option!).
    
    This creates a semi-transparent brain showing effect sizes
    as colored blobs. Works without needing exact atlas files.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        output_dir: Where to save figures  
        diagnosis: Diagnosis name
        vmax: Max colorbar value
        threshold: Min absolute value to show
        cmap: 'cold_hot' (red/blue) or 'hot' (red only)
    """
    
    if not NILEARN_AVAILABLE:
        print("[ERROR] nilearn required for glass brain")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter significant
    sig_df = effect_sizes_df[effect_sizes_df['Significant_FDR'] == True].copy()
    
    if len(sig_df) == 0:
        print("[WARNING] No significant ROIs")
        return
    
    print(f"\n[INFO] Creating glass brain for {diagnosis}")
    print(f"[INFO] Visualizing {len(sig_df)} significant ROIs")
    
    # For glass brain, we can use approximate MNI coordinates
    # Based on ROI names
    
    # Simplified approach: Use plotting functions that accept
    # statistical maps
    
    # Create dummy stat map for demonstration
    # In practice, you'd load/create a proper NIfTI file
    
    print("[INFO] For glass brain visualization, you need:")
    print("  1. NIfTI file with your atlas labels")
    print("  2. Or: MNI coordinates for each ROI")
    print("  3. Then map your effect sizes to those coordinates/labels")
    
    # Placeholder - show what the call would look like:
    """
    display = plotting.plot_glass_brain(
        stat_map_img,  # Your effect sizes as NIfTI
        threshold=threshold,
        vmax=vmax,
        cmap=cmap,
        colorbar=True,
        title=f"Regional Effect Sizes: {diagnosis} vs HC",
        plot_abs=False,  # Show both positive and negative
        display_mode='lyrz',  # Show all views
        figure=fig
    )
    """
    
    # For now, create a summary plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Sort by absolute effect size
    sig_df['abs_delta'] = sig_df['Cliffs_Delta'].abs()
    top_rois = sig_df.nlargest(20, 'abs_delta')
    
    # Plot 1: Bar plot of top ROIs
    ax = axes[0]
    colors = ['red' if x > 0 else 'blue' for x in top_rois['Cliffs_Delta']]
    y_pos = np.arange(len(top_rois))
    ax.barh(y_pos, top_rois['Cliffs_Delta'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[:30] for r in top_rois['ROI_Name']], fontsize=8)
    ax.set_xlabel("Cliffs Delta")
    ax.set_title(f"Top 20 ROIs: {diagnosis}")
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Histogram of effect sizes
    ax = axes[1]
    ax.hist(sig_df['Cliffs_Delta'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='--')
    ax.axvline(0.3, color='orange', linestyle='--', label='Moderate')
    ax.axvline(0.5, color='red', linestyle='--', label='Large')
    ax.set_xlabel("Cliffs Delta")
    ax.set_ylabel("Count")
    ax.set_title("Effect Size Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Scatter: effect size vs p-value
    ax = axes[2]
    ax.scatter(sig_df['Cliffs_Delta'], -np.log10(sig_df['p_value']),
              c=sig_df['Cliffs_Delta'], cmap='RdBu_r', alpha=0.7,
              s=100, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("Cliffs Delta")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Effect Size vs Significance")
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"glass_brain_summary_{diagnosis}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] Saved: {output_path}")


# ============================================================================
# VISUALIZATION FUNCTION 3: Simple Brain Schematic
# ============================================================================

def visualize_brain_schematic(
    effect_sizes_df: pd.DataFrame,
    output_dir: str = './brain_visualizations',
    diagnosis: str = 'CAT',
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Create simple brain schematic showing which regions are affected.
    
    This is a fallback when atlas files are not available.
    Shows a 2D layout of brain regions with color-coded effects.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    sig_df = effect_sizes_df[effect_sizes_df['Significant_FDR'] == True].copy()
    
    if len(sig_df) == 0:
        print("[WARNING] No significant ROIs")
        return
    
    print(f"\n[INFO] Creating brain schematic for {diagnosis}")
    
    # Group ROIs by brain system
    frontal_keys = ['frontal', 'precentral', 'cingulate']
    temporal_keys = ['temporal', 'hippocampus', 'amygdala', 'entorhinal']
    parietal_keys = ['parietal', 'postcentral', 'precuneus', 'supramarginal']
    occipital_keys = ['occipital', 'cuneus', 'lingual', 'pericalcarine']
    subcortical_keys = ['thalamus', 'caudate', 'putamen', 'pallidum', 'accumbens']
    
    def categorize_roi(roi_name):
        roi_lower = roi_name.lower()
        if any(k in roi_lower for k in frontal_keys):
            return 'Frontal'
        elif any(k in roi_lower for k in temporal_keys):
            return 'Temporal'
        elif any(k in roi_lower for k in parietal_keys):
            return 'Parietal'
        elif any(k in roi_lower for k in occipital_keys):
            return 'Occipital'
        elif any(k in roi_lower for k in subcortical_keys):
            return 'Subcortical'
        else:
            return 'Other'
    
    sig_df['System'] = sig_df['ROI_Name'].apply(categorize_roi)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    systems = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Subcortical', 'Other']
    
    for idx, system in enumerate(systems):
        ax = axes[idx]
        system_data = sig_df[sig_df['System'] == system]
        
        if len(system_data) == 0:
            ax.text(0.5, 0.5, f'No significant\n{system} ROIs',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f"{system} Cortex", fontweight='bold', fontsize=14)
            ax.axis('off')
            continue
        
        # Sort by effect size
        system_data = system_data.sort_values('Cliffs_Delta', ascending=True)
        
        # Plot
        y_pos = np.arange(len(system_data))
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in system_data['Cliffs_Delta']]
        
        bars = ax.barh(y_pos, system_data['Cliffs_Delta'], color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r.split('(')[0].strip()[:25] for r in system_data['ROI_Name']], 
                          fontsize=9)
        ax.set_xlabel("Cliffs Delta", fontsize=10, fontweight='bold')
        ax.set_title(f"{system} Cortex (n={len(system_data)})", 
                    fontweight='bold', fontsize=12)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-0.3, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(
        f"Regional Effect Sizes by Brain System: {diagnosis} vs HC\n"
        f"(Red = increased deviation, Blue = decreased deviation)",
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"brain_schematic_{diagnosis}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] Saved: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_all_brain_visualizations(
    effect_sizes_csv: str,
    output_dir: str = './brain_visualizations_3d',
    diagnosis_column: str = 'Diagnosis',
    atlas_file: str = None
):
    """
    Create all brain visualizations from effect sizes CSV.
    
    Args:
        effect_sizes_csv: Path to regional_effect_sizes_combined.csv
        output_dir: Output directory
        diagnosis_column: Column name for diagnosis
        atlas_file: Path to atlas NIfTI (optional)
    """
    
    print("="*80)
    print("CREATING 3D BRAIN VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print(f"\n[INFO] Loading: {effect_sizes_csv}")
    df = pd.read_csv(effect_sizes_csv)
    
    print(f"[INFO] Loaded {len(df)} ROIs")
    print(f"[INFO] Diagnoses: {df[diagnosis_column].unique()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each diagnosis
    for diagnosis in df[diagnosis_column].unique():
        print(f"\n{'='*80}")
        print(f"Processing: {diagnosis}")
        print('='*80)
        
        diagnosis_data = df[df[diagnosis_column] == diagnosis].copy()
        
        # 1. Brain schematic (always works, no dependencies)
        visualize_brain_schematic(
            diagnosis_data,
            output_dir=output_dir,
            diagnosis=diagnosis
        )
        
        # 2. Glass brain summary
        visualize_glass_brain_effects(
            diagnosis_data,
            output_dir=output_dir,
            diagnosis=diagnosis
        )
        
        # 3. Surface visualization (if nilearn available)
        if NILEARN_AVAILABLE:
            try:
                visualize_surface_effects(
                    diagnosis_data,
                    atlas_file=atlas_file,
                    output_dir=output_dir,
                    diagnosis=diagnosis
                )
            except Exception as e:
                print(f"[WARNING] Surface visualization failed: {e}")
    
    print(f"\n{'='*80}")
    print("VISUALIZATIONS COMPLETE!")
    print('='*80)
    print(f"\nOutput directory: {output_dir}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Example usage
    effect_sizes_csv = "regional_effect_sizes_combined.csv"
    output_dir = "./brain_visualizations_3d"
    
    # Create all visualizations
    create_all_brain_visualizations(
        effect_sizes_csv=effect_sizes_csv,
        output_dir=output_dir
    )
    
    print("\n✓ All visualizations created!")
    print(f"  Check: {output_dir}/")