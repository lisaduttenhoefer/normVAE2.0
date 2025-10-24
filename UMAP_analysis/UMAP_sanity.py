#!/usr/bin/env python3
"""
UMAP comparison of old vs new CAT12 processed data
Compares grey matter volumes (Vgm) across all atlases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from umap import UMAP
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

NEW_DATA_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
OLD_DATA_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/catatonia_VAE-main_bq/data/all_csv_data/csv_files"
METADATA_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/complete_metadata.csv"
OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/UMAP_analysis"

# UMAP parameters
UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42
}

# ============================================================================
# Helper Functions
# ============================================================================

def normalize_patient_id(patient_id):
    """Remove 'sub-' prefix from patient ID for matching"""
    if isinstance(patient_id, str):
        return patient_id.replace('sub-', '')
    return patient_id

def load_new_data(filepath):
    """Load new CAT12 data and extract Vgm columns"""
    print(f"Loading new data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Get all Vgm columns (these start with "Vgm_")
    vgm_cols = [col for col in df.columns if col.startswith('Vgm_')]
    print(f"  Found {len(vgm_cols)} Vgm columns")
    
    # Assuming first column or a column named 'Filename' contains patient IDs
    id_col = None
    for possible_id in ['Filename', 'filename', 'ID', 'id', 'Subject', 'subject']:
        if possible_id in df.columns:
            id_col = possible_id
            break
    
    if id_col is None:
        # Use first column as ID
        id_col = df.columns[0]
        print(f"  Using first column '{id_col}' as patient ID")
    else:
        print(f"  Using column '{id_col}' as patient ID")
    
    # Extract patient IDs and Vgm data
    df_new = df[[id_col] + vgm_cols].copy()
    df_new['patient_id_normalized'] = df_new[id_col].apply(normalize_patient_id)
    
    # Set normalized ID as index
    df_new = df_new.set_index('patient_id_normalized')
    df_new = df_new[vgm_cols]
    
    print(f"  Loaded {len(df_new)} patients with {len(vgm_cols)} features")
    return df_new

def load_old_data(directory):
    """Load old CAT12 data from multiple atlas CSV files"""
    print(f"\nLoading old data from {directory}...")
    
    directory = Path(directory)
    csv_files = list(directory.glob("Aggregated_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No 'Aggregated_*.csv' files found in {directory}")
    
    print(f"  Found {len(csv_files)} atlas files: {[f.name for f in csv_files]}")
    
    all_data = []
    
    for csv_file in csv_files:
        atlas_name = csv_file.stem.replace('Aggregated_', '')
        print(f"  Processing {atlas_name}...")
        
        # Read CSV with MultiIndex columns
        df = pd.read_csv(csv_file, header=[0, 1], index_col=0)
        
        # Extract only Vgm rows (second level of MultiIndex)
        # The MultiIndex is (patient_id, volume_type)
        vgm_data = df.loc[:, (slice(None), 'Vgm')]
        
        # Get patient IDs and region names
        patient_ids = vgm_data.columns.get_level_values(0).tolist()
        region_names = vgm_data.index.tolist()
        
        # Convert to numpy array and transpose
        data_array = vgm_data.values.T  # Now: rows=patients, cols=regions
        
        # Create DataFrame with patients as rows
        vgm_data_t = pd.DataFrame(
            data_array,
            index=patient_ids,
            columns=[f"{atlas_name}_{region}" for region in region_names]
        )
        
        all_data.append(vgm_data_t)
        print(f"    Added {len(vgm_data_t)} patients, {len(vgm_data_t.columns)} regions")
    
    # Concatenate all atlases
    df_old = pd.concat(all_data, axis=1, join='outer')
    
    # Normalize patient IDs
    df_old['patient_id_normalized'] = df_old.index.map(normalize_patient_id)
    df_old = df_old.set_index('patient_id_normalized')
    
    print(f"  Total: {len(df_old)} patients with {len(df_old.columns)} features")
    return df_old

def load_metadata(filepath):
    """Load metadata with dataset information"""
    print(f"\nLoading metadata from {filepath}...")
    df_meta = pd.read_csv(filepath)
    
    # Find the relevant columns
    filename_col = None
    dataset_col = None
    
    for col in df_meta.columns:
        if col.lower() in ['filename', 'file_name', 'id', 'subject']:
            filename_col = col
        if col.lower() in ['dataset', 'data_set', 'cohort']:
            dataset_col = col
    
    if filename_col is None or dataset_col is None:
        print(f"  Warning: Could not find Filename or Dataset columns")
        print(f"  Available columns: {df_meta.columns.tolist()}")
        return None
    
    print(f"  Using '{filename_col}' and '{dataset_col}' columns")
    
    # Create normalized mapping
    df_meta['patient_id_normalized'] = df_meta[filename_col].apply(normalize_patient_id)
    metadata_dict = df_meta.set_index('patient_id_normalized')[dataset_col].to_dict()
    
    print(f"  Loaded metadata for {len(metadata_dict)} patients")
    print(f"  Datasets: {df_meta[dataset_col].unique()}")
    return metadata_dict

def match_and_align_data(df_new, df_old):
    """Match patients and align features between old and new data"""
    print("\nMatching patients between old and new data...")
    
    # Find common patients
    common_patients = df_new.index.intersection(df_old.index)
    print(f"  Common patients: {len(common_patients)}")
    print(f"  New only: {len(df_new.index.difference(df_old.index))}")
    print(f"  Old only: {len(df_old.index.difference(df_new.index))}")
    
    if len(common_patients) == 0:
        raise ValueError("No common patients found! Check patient ID normalization.")
    
    # Filter to common patients
    df_new_matched = df_new.loc[common_patients].copy()
    df_old_matched = df_old.loc[common_patients].copy()
    
    # ========================================================================
    # IMPORTANT: Align features between old and new data
    # ========================================================================
    print("\nAligning features between datasets...")
    
    # Normalize atlas names to match between datasets
    ATLAS_MAPPING = {
        'neurom': 'neuromorphometrics',
        'neuromorphometrics': 'neuromorphometrics',
        'suit': 'suit',
        'cobra': 'cobra',
        'lpba40': 'lpba40',
        'thalamus': 'thalamus',
        'thalamic': 'thalamus',  # thalamic_nuclei maps to thalamus
    }
    
    def normalize_atlas_name(atlas):
        """Normalize atlas name to standard form"""
        atlas_lower = atlas.lower()
        return ATLAS_MAPPING.get(atlas_lower, atlas_lower)
    
    def normalize_region_name(region):
        """Normalize region name by removing spaces and standardizing"""
        # Remove spaces and convert to lowercase for matching
        region = region.replace(' ', '').replace('-', '').lower()
        return region
    
    # Show example column names for debugging
    print("\n  Example NEW column names:")
    for col in list(df_new_matched.columns)[:5]:
        print(f"    {col}")
    
    print("\n  Example OLD column names:")
    for col in list(df_old_matched.columns)[:5]:
        print(f"    {col}")
    
    # Extract atlas and region information from column names
    def parse_new_column_name(col):
        """Parse new data column name: Vgm_{atlas}_{region}"""
        if col.startswith('Vgm_'):
            parts = col[4:].split('_', 1)  # Remove 'Vgm_' and split into atlas and region
            if len(parts) == 2:
                atlas = normalize_atlas_name(parts[0])
                region = normalize_region_name(parts[1])
                return f"{atlas}_{region}"
            else:
                # Only one part after Vgm_, treat as region without atlas
                return col[4:]
        return col
    
    def parse_old_column_name(col):
        """Parse old data column name: {atlas}_{region}"""
        parts = col.split('_', 1)
        if len(parts) == 2:
            atlas = normalize_atlas_name(parts[0])
            region = normalize_region_name(parts[1])
            return f"{atlas}_{region}"
        return col
    
    # Normalize column names for both datasets
    new_cols_normalized = {}
    for col in df_new_matched.columns:
        normalized = parse_new_column_name(col)
        new_cols_normalized[col] = normalized
    
    old_cols_normalized = {}
    for col in df_old_matched.columns:
        normalized = parse_old_column_name(col)
        old_cols_normalized[col] = normalized
    
    # Show normalized examples
    print("\n  Example NORMALIZED NEW column names:")
    for i, (orig, norm) in enumerate(list(new_cols_normalized.items())[:5]):
        print(f"    {orig} -> {norm}")
    
    print("\n  Example NORMALIZED OLD column names:")
    for i, (orig, norm) in enumerate(list(old_cols_normalized.items())[:5]):
        print(f"    {orig} -> {norm}")
    
    # Find common features
    new_features = set(new_cols_normalized.values())
    old_features = set(old_cols_normalized.values())
    common_features = new_features.intersection(old_features)
    
    # Analyze by atlas
    print("\n  Feature count by atlas:")
    atlases_new = {}
    atlases_old = {}
    
    for feat in new_features:
        if '_' in feat:
            atlas = feat.split('_')[0]
            atlases_new[atlas] = atlases_new.get(atlas, 0) + 1
    
    for feat in old_features:
        if '_' in feat:
            atlas = feat.split('_')[0]
            atlases_old[atlas] = atlases_old.get(atlas, 0) + 1
    
    print("\n    NEW data atlases:")
    for atlas, count in sorted(atlases_new.items()):
        print(f"      {atlas}: {count} regions")
    
    print("\n    OLD data atlases:")
    for atlas, count in sorted(atlases_old.items()):
        print(f"      {atlas}: {count} regions")
    
    print(f"\n  New dataset features: {len(new_features)}")
    print(f"  Old dataset features: {len(old_features)}")
    print(f"  Common features: {len(common_features)}")
    print(f"  New only features: {len(new_features - old_features)}")
    print(f"  Old only features: {len(old_features - new_features)}")
    
    # Show some examples of non-matching features
    if len(new_features - old_features) > 0:
        print("\n  Example features only in NEW:")
        for feat in list(new_features - old_features)[:5]:
            print(f"    {feat}")
    
    if len(old_features - new_features) > 0:
        print("\n  Example features only in OLD:")
        for feat in list(old_features - new_features)[:5]:
            print(f"    {feat}")
    
    if len(common_features) == 0:
        raise ValueError("No common features found between datasets!")
    
    # Create reverse mapping to get original column names
    new_cols_reverse = {v: k for k, v in new_cols_normalized.items()}
    old_cols_reverse = {v: k for k, v in old_cols_normalized.items()}
    
    # Select only common features
    common_features_sorted = sorted(common_features)
    
    new_common_cols = [new_cols_reverse[feat] for feat in common_features_sorted]
    old_common_cols = [old_cols_reverse[feat] for feat in common_features_sorted]
    
    df_new_aligned = df_new_matched[new_common_cols].copy()
    df_old_aligned = df_old_matched[old_common_cols].copy()
    
    # Rename to common names for consistency
    df_new_aligned.columns = common_features_sorted
    df_old_aligned.columns = common_features_sorted
    
    print(f"\n  Final aligned shape: {df_new_aligned.shape}")
    
    # Handle missing values
    print("\nHandling missing values...")
    print(f"  New data NaNs: {df_new_aligned.isna().sum().sum()}")
    print(f"  Old data NaNs: {df_old_aligned.isna().sum().sum()}")
    
    # Fill NaNs with column mean
    df_new_aligned = df_new_aligned.fillna(df_new_aligned.mean())
    df_old_aligned = df_old_aligned.fillna(df_old_aligned.mean())
    
    return df_new_aligned, df_old_aligned, common_patients

def create_combined_umap(df_new, df_old, common_patients, metadata_dict, output_dir, metadata_path):
    """Create UMAP visualizations comparing old and new data"""
    print("\nCreating UMAP embeddings...")
    
    # Combine data
    df_new_combined = df_new.copy()
    df_new_combined['data_source'] = 'New'
    
    df_old_combined = df_old.copy()
    df_old_combined['data_source'] = 'Old'
    
    # Standardize features separately for each dataset
    from sklearn.preprocessing import StandardScaler
    
    scaler_new = StandardScaler()
    scaler_old = StandardScaler()
    
    new_scaled = scaler_new.fit_transform(df_new.values)
    old_scaled = scaler_old.fit_transform(df_old.values)
    
    # Combine scaled data
    combined_data = np.vstack([new_scaled, old_scaled])
    
    # Create labels
    labels_source = ['New'] * len(df_new) + ['Old'] * len(df_old)
    patient_ids = list(df_new.index) + list(df_old.index)
    
    # Add dataset information from metadata
    labels_dataset = []
    for pid in patient_ids:
        if metadata_dict and pid in metadata_dict:
            labels_dataset.append(metadata_dict[pid])
        else:
            labels_dataset.append('Unknown')
    
    # Fit UMAP
    print(f"  Fitting UMAP on {combined_data.shape[0]} samples with {combined_data.shape[1]} features...")
    reducer = UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(combined_data)
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Source': labels_source,
        'Dataset': labels_dataset,
        'Patient_ID': patient_ids
    })
    
    # ========================================================================
    # Plot 1: Colored by Data Source (Old vs New)
    # ========================================================================
    print("\nCreating UMAP plot colored by data source...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'New': '#e74c3c', 'Old': '#3498db'}
    
    for source in ['Old', 'New']:
        mask = df_plot['Source'] == source
        ax.scatter(
            df_plot.loc[mask, 'UMAP1'],
            df_plot.loc[mask, 'UMAP2'],
            c=colors[source],
            label=source,
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP: Old vs New CAT12 Data (All Atlases, Vgm only)', fontsize=14, fontweight='bold')
    ax.legend(title='Data Source', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_source = Path(output_dir) / 'umap_comparison_by_source.png'
    plt.savefig(output_path_source, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path_source}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Colored by Dataset
    # ========================================================================
    print("\nCreating UMAP plot colored by dataset...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get unique datasets
    unique_datasets = sorted([d for d in df_plot['Dataset'].unique() if d != 'Unknown'])
    
    # Create color palette
    if len(unique_datasets) > 0:
        palette = sns.color_palette('husl', len(unique_datasets))
        color_map = dict(zip(unique_datasets, palette))
        color_map['Unknown'] = '#cccccc'
    else:
        color_map = {'Unknown': '#cccccc'}
    
    # Left plot: All data colored by dataset
    for dataset in unique_datasets + ['Unknown']:
        mask = df_plot['Dataset'] == dataset
        if mask.sum() > 0:
            axes[0].scatter(
                df_plot.loc[mask, 'UMAP1'],
                df_plot.loc[mask, 'UMAP2'],
                c=[color_map[dataset]],
                label=dataset,
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )
    
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title('All Data Colored by Dataset', fontsize=14, fontweight='bold')
    axes[0].legend(title='Dataset', fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Separate by Old/New
    for i, source in enumerate(['Old', 'New']):
        mask_source = df_plot['Source'] == source
        
        for dataset in unique_datasets:
            mask = mask_source & (df_plot['Dataset'] == dataset)
            if mask.sum() > 0:
                marker = 'o' if source == 'Old' else '^'
                axes[1].scatter(
                    df_plot.loc[mask, 'UMAP1'],
                    df_plot.loc[mask, 'UMAP2'],
                    c=[color_map[dataset]],
                    label=f'{dataset} ({source})',
                    alpha=0.6,
                    s=50,
                    marker=marker,
                    edgecolors='white',
                    linewidths=0.5
                )
    
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title('Data Colored by Dataset (○=Old, △=New)', fontsize=14, fontweight='bold')
    axes[1].legend(title='Dataset (Source)', fontsize=8, loc='best', ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_dataset = Path(output_dir) / 'umap_comparison_by_dataset.png'
    plt.savefig(output_path_dataset, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path_dataset}")
    plt.close()
    
    # ========================================================================
    # Plot 3: Separate UMAPs for Old and New (side by side)
    # ========================================================================
    print("\nCreating separate UMAPs for old vs new data...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Old data only
    mask_old = df_plot['Source'] == 'Old'
    for dataset in unique_datasets:
        mask = mask_old & (df_plot['Dataset'] == dataset)
        if mask.sum() > 0:
            axes[0].scatter(
                df_plot.loc[mask, 'UMAP1'],
                df_plot.loc[mask, 'UMAP2'],
                c=[color_map[dataset]],
                label=dataset,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )
    
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title('OLD Data Only (Colored by Dataset)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Dataset', fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Right: New data only
    mask_new = df_plot['Source'] == 'New'
    for dataset in unique_datasets:
        mask = mask_new & (df_plot['Dataset'] == dataset)
        if mask.sum() > 0:
            axes[1].scatter(
                df_plot.loc[mask, 'UMAP1'],
                df_plot.loc[mask, 'UMAP2'],
                c=[color_map[dataset]],
                label=dataset,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )
    
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title('NEW Data Only (Colored by Dataset)', fontsize=14, fontweight='bold')
    axes[1].legend(title='Dataset', fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_separate = Path(output_dir) / 'umap_old_vs_new_separate.png'
    plt.savefig(output_path_separate, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path_separate}")
    plt.close()
    
    # ========================================================================
    # Plot 4: Separate UMAP embeddings (independent clustering)
    # ========================================================================
    print("\nCreating independent UMAPs for old and new data...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Fit separate UMAP for OLD data (only matched patients)
    old_data = df_plot[df_plot['Source'] == 'Old'].copy()
    old_scaled = scaler_old.transform(df_old.loc[old_data['Patient_ID']].values)
    
    reducer_old = UMAP(**UMAP_PARAMS)
    embedding_old = reducer_old.fit_transform(old_scaled)
    
    old_data['UMAP1_independent'] = embedding_old[:, 0]
    old_data['UMAP2_independent'] = embedding_old[:, 1]
    
    # Fit separate UMAP for NEW data (ALL new data, not just matched)
    print("  Using ALL new data for independent UMAP (including non-matched patients)...")
    new_data_all = df_new.copy()  # Use original df_new with ALL patients
    new_data_all = new_data_all.fillna(new_data_all.mean())
    
    # Add metadata for all new patients
    new_patient_ids = new_data_all.index.tolist()
    new_datasets = []
    for pid in new_patient_ids:
        if metadata_dict and pid in metadata_dict:
            new_datasets.append(metadata_dict[pid])
        else:
            new_datasets.append('Unknown')
    
    # Scale and fit UMAP for all new data
    scaler_new_all = StandardScaler()
    new_scaled_all = scaler_new_all.fit_transform(new_data_all.values)
    
    reducer_new = UMAP(**UMAP_PARAMS)
    embedding_new = reducer_new.fit_transform(new_scaled_all)
    
    new_data_all_df = pd.DataFrame({
        'UMAP1_independent': embedding_new[:, 0],
        'UMAP2_independent': embedding_new[:, 1],
        'Patient_ID': new_patient_ids,
        'Dataset': new_datasets
    })
    
    # Update unique datasets to include all datasets in new data
    all_unique_datasets = sorted([d for d in set(new_datasets) if d != 'Unknown'])
    if len(all_unique_datasets) > len(unique_datasets):
        # Regenerate color palette for all datasets
        palette_all = sns.color_palette('husl', len(all_unique_datasets))
        color_map_all = dict(zip(all_unique_datasets, palette_all))
        color_map_all['Unknown'] = '#cccccc'
    else:
        color_map_all = color_map
    
    print(f"  Old data: {len(old_data)} patients")
    print(f"  New data: {len(new_data_all_df)} patients (ALL)")
    print(f"  Datasets in new data: {all_unique_datasets}")
    
    # Plot OLD data with independent UMAP
    for dataset in unique_datasets:
        mask = old_data['Dataset'] == dataset
        if mask.sum() > 0:
            axes[0].scatter(
                old_data.loc[mask, 'UMAP1_independent'],
                old_data.loc[mask, 'UMAP2_independent'],
                c=[color_map[dataset]],
                label=dataset,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )
    
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title('OLD Data - Independent UMAP (Colored by Dataset)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Dataset', fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot NEW data with independent UMAP (ALL data)
    for dataset in all_unique_datasets:
        mask = new_data_all_df['Dataset'] == dataset
        if mask.sum() > 0:
            axes[1].scatter(
                new_data_all_df.loc[mask, 'UMAP1_independent'],
                new_data_all_df.loc[mask, 'UMAP2_independent'],
                c=[color_map_all[dataset]],
                label=dataset,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )
    
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title(f'NEW Data - Independent UMAP (Colored by Dataset, n={len(new_data_all_df)})', fontsize=14, fontweight='bold')
    axes[1].legend(title='Dataset', fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_independent = Path(output_dir) / 'umap_old_vs_new_independent.png'
    plt.savefig(output_path_independent, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path_independent}")
    plt.close()
    
    # ========================================================================
    # Plot 5 & 6: Age and Sex colored UMAPs
    # ========================================================================
    print("\nCreating UMAPs colored by Age and Sex...")
    
    # Load full metadata for age and sex
    try:
        df_meta_full = pd.read_csv(metadata_path)
        
        # Find age and sex columns
        age_col = None
        sex_col = None
        
        for col in df_meta_full.columns:
            col_lower = col.lower()
            if 'age' in col_lower:
                age_col = col
            if col_lower in ['sex', 'gender', 'sexe']:
                sex_col = col
        
        if age_col is None or sex_col is None:
            print(f"  Warning: Could not find Age or Sex columns in metadata")
            print(f"  Available columns: {df_meta_full.columns.tolist()}")
        else:
            print(f"  Using '{age_col}' for age and '{sex_col}' for sex")
            
            # Find filename column
            filename_col = None
            for col in df_meta_full.columns:
                if col.lower() in ['filename', 'file_name', 'id', 'subject']:
                    filename_col = col
                    break
            
            if filename_col:
                # Create mappings
                df_meta_full['patient_id_normalized'] = df_meta_full[filename_col].apply(normalize_patient_id)
                age_dict = df_meta_full.set_index('patient_id_normalized')[age_col].to_dict()
                sex_dict = df_meta_full.set_index('patient_id_normalized')[sex_col].to_dict()
                
                # Add age and sex to new_data_all_df
                new_data_all_df['Age'] = new_data_all_df['Patient_ID'].map(age_dict)
                new_data_all_df['Sex'] = new_data_all_df['Patient_ID'].map(sex_dict)
                
                # Add age and sex to old_data
                old_data['Age'] = old_data['Patient_ID'].map(age_dict)
                old_data['Sex'] = old_data['Patient_ID'].map(sex_dict)
                
                # ====== AGE PLOT ======
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                # Old data - Age
                mask_age_old = old_data['Age'].notna()
                if mask_age_old.sum() > 0:
                    scatter_old = axes[0].scatter(
                        old_data.loc[mask_age_old, 'UMAP1_independent'],
                        old_data.loc[mask_age_old, 'UMAP2_independent'],
                        c=old_data.loc[mask_age_old, 'Age'],
                        cmap='viridis',
                        alpha=0.7,
                        s=60,
                        edgecolors='white',
                        linewidths=0.5
                    )
                    plt.colorbar(scatter_old, ax=axes[0], label='Age')
                
                axes[0].set_xlabel('UMAP 1', fontsize=12)
                axes[0].set_ylabel('UMAP 2', fontsize=12)
                axes[0].set_title(f'OLD Data - Colored by Age (n={mask_age_old.sum()})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # New data - Age
                mask_age_new = new_data_all_df['Age'].notna()
                if mask_age_new.sum() > 0:
                    scatter_new = axes[1].scatter(
                        new_data_all_df.loc[mask_age_new, 'UMAP1_independent'],
                        new_data_all_df.loc[mask_age_new, 'UMAP2_independent'],
                        c=new_data_all_df.loc[mask_age_new, 'Age'],
                        cmap='viridis',
                        alpha=0.7,
                        s=60,
                        edgecolors='white',
                        linewidths=0.5
                    )
                    plt.colorbar(scatter_new, ax=axes[1], label='Age')
                
                axes[1].set_xlabel('UMAP 1', fontsize=12)
                axes[1].set_ylabel('UMAP 2', fontsize=12)
                axes[1].set_title(f'NEW Data - Colored by Age (n={mask_age_new.sum()})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                output_path_age = Path(output_dir) / 'umap_old_vs_new_by_age.png'
                plt.savefig(output_path_age, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path_age}")
                plt.close()
                
                # ====== SEX PLOT ======
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                sex_colors = {'M': '#3498db', 'F': '#e74c3c', 'm': '#3498db', 'f': '#e74c3c', 
                              'Male': '#3498db', 'Female': '#e74c3c', 'male': '#3498db', 'female': '#e74c3c',
                              1: '#3498db', 2: '#e74c3c', '1': '#3498db', '2': '#e74c3c'}
                
                # Old data - Sex
                mask_sex_old = old_data['Sex'].notna()
                if mask_sex_old.sum() > 0:
                    for sex_val in old_data.loc[mask_sex_old, 'Sex'].unique():
                        mask = (old_data['Sex'] == sex_val) & mask_sex_old
                        if mask.sum() > 0:
                            color = sex_colors.get(sex_val, '#cccccc')
                            axes[0].scatter(
                                old_data.loc[mask, 'UMAP1_independent'],
                                old_data.loc[mask, 'UMAP2_independent'],
                                c=color,
                                label=str(sex_val),
                                alpha=0.7,
                                s=60,
                                edgecolors='white',
                                linewidths=0.5
                            )
                
                axes[0].set_xlabel('UMAP 1', fontsize=12)
                axes[0].set_ylabel('UMAP 2', fontsize=12)
                axes[0].set_title(f'OLD Data - Colored by Sex (n={mask_sex_old.sum()})', fontsize=14, fontweight='bold')
                axes[0].legend(title='Sex', fontsize=9)
                axes[0].grid(True, alpha=0.3)
                
                # New data - Sex
                mask_sex_new = new_data_all_df['Sex'].notna()
                if mask_sex_new.sum() > 0:
                    for sex_val in new_data_all_df.loc[mask_sex_new, 'Sex'].unique():
                        mask = (new_data_all_df['Sex'] == sex_val) & mask_sex_new
                        if mask.sum() > 0:
                            color = sex_colors.get(sex_val, '#cccccc')
                            axes[1].scatter(
                                new_data_all_df.loc[mask, 'UMAP1_independent'],
                                new_data_all_df.loc[mask, 'UMAP2_independent'],
                                c=color,
                                label=str(sex_val),
                                alpha=0.7,
                                s=60,
                                edgecolors='white',
                                linewidths=0.5
                            )
                
                axes[1].set_xlabel('UMAP 1', fontsize=12)
                axes[1].set_ylabel('UMAP 2', fontsize=12)
                axes[1].set_title(f'NEW Data - Colored by Sex (n={mask_sex_new.sum()})', fontsize=14, fontweight='bold')
                axes[1].legend(title='Sex', fontsize=9)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                output_path_sex = Path(output_dir) / 'umap_old_vs_new_by_sex.png'
                plt.savefig(output_path_sex, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path_sex}")
                plt.close()
                
    except Exception as e:
        print(f"  Warning: Could not create Age/Sex plots: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Plot 7: NEW data with ALL features (761 features, not just matched 260)
    # ========================================================================
    print("\nCreating UMAP for NEW data using ALL features (all atlases)...")
    
    try:
        # Load the ORIGINAL new data with ALL features
        from pathlib import Path as PathLib
        df_new_original = load_new_data(NEW_DATA_PATH)
        df_new_original = df_new_original.fillna(df_new_original.mean())
        
        # Get metadata for all patients
        all_patient_ids = df_new_original.index.tolist()
        all_datasets = []
        all_ages = []
        all_sexes = []
        
        for pid in all_patient_ids:
            if metadata_dict and pid in metadata_dict:
                all_datasets.append(metadata_dict[pid])
            else:
                all_datasets.append('Unknown')
            
            # Try to get age and sex if we have the mappings
            try:
                all_ages.append(age_dict.get(pid, np.nan))
                all_sexes.append(sex_dict.get(pid, None))
            except:
                all_ages.append(np.nan)
                all_sexes.append(None)
        
        # Scale and fit UMAP with ALL 761 features
        scaler_all_features = StandardScaler()
        new_all_features_scaled = scaler_all_features.fit_transform(df_new_original.values)
        
        print(f"  Fitting UMAP with ALL {df_new_original.shape[1]} features for {df_new_original.shape[0]} patients...")
        reducer_all_features = UMAP(**UMAP_PARAMS)
        embedding_all_features = reducer_all_features.fit_transform(new_all_features_scaled)
        
        # Create DataFrame
        df_all_features = pd.DataFrame({
            'UMAP1': embedding_all_features[:, 0],
            'UMAP2': embedding_all_features[:, 1],
            'Patient_ID': all_patient_ids,
            'Dataset': all_datasets,
            'Age': all_ages,
            'Sex': all_sexes
        })
        
        # Get unique datasets
        datasets_all_feat = sorted([d for d in set(all_datasets) if d != 'Unknown'])
        palette_all_feat = sns.color_palette('husl', len(datasets_all_feat))
        color_map_all_feat = dict(zip(datasets_all_feat, palette_all_feat))
        color_map_all_feat['Unknown'] = '#cccccc'
        
        print(f"  Datasets found: {datasets_all_feat}")
        
        # Create 2x2 plot grid
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        
        # Plot 1: By Dataset
        for dataset in datasets_all_feat + ['Unknown']:
            mask = df_all_features['Dataset'] == dataset
            if mask.sum() > 0:
                axes[0, 0].scatter(
                    df_all_features.loc[mask, 'UMAP1'],
                    df_all_features.loc[mask, 'UMAP2'],
                    c=[color_map_all_feat[dataset]],
                    label=dataset,
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidths=0.5
                )
        
        axes[0, 0].set_xlabel('UMAP 1', fontsize=12)
        axes[0, 0].set_ylabel('UMAP 2', fontsize=12)
        axes[0, 0].set_title(f'Colored by Dataset (n={len(df_all_features)})', fontsize=14, fontweight='bold')
        axes[0, 0].legend(title='Dataset', fontsize=9, loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: By Age
        mask_age = df_all_features['Age'].notna()
        if mask_age.sum() > 0:
            scatter = axes[0, 1].scatter(
                df_all_features.loc[mask_age, 'UMAP1'],
                df_all_features.loc[mask_age, 'UMAP2'],
                c=df_all_features.loc[mask_age, 'Age'],
                cmap='viridis',
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )
            plt.colorbar(scatter, ax=axes[0, 1], label='Age')
        
        axes[0, 1].set_xlabel('UMAP 1', fontsize=12)
        axes[0, 1].set_ylabel('UMAP 2', fontsize=12)
        axes[0, 1].set_title(f'Colored by Age (n={mask_age.sum()})', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: By Sex
        mask_sex = df_all_features['Sex'].notna()
        if mask_sex.sum() > 0:
            for sex_val in df_all_features.loc[mask_sex, 'Sex'].unique():
                mask = (df_all_features['Sex'] == sex_val) & mask_sex
                if mask.sum() > 0:
                    color = sex_colors.get(sex_val, '#cccccc')
                    axes[1, 0].scatter(
                        df_all_features.loc[mask, 'UMAP1'],
                        df_all_features.loc[mask, 'UMAP2'],
                        c=color,
                        label=str(sex_val),
                        alpha=0.7,
                        s=50,
                        edgecolors='white',
                        linewidths=0.5
                    )
        
        axes[1, 0].set_xlabel('UMAP 1', fontsize=12)
        axes[1, 0].set_ylabel('UMAP 2', fontsize=12)
        axes[1, 0].set_title(f'Colored by Sex (n={mask_sex.sum()})', fontsize=14, fontweight='bold')
        axes[1, 0].legend(title='Sex', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: By Diagnosis
        # Find diagnosis column in metadata
        diag_col = None
        for col in df_meta_full.columns:
            col_lower = col.lower()
            if 'diagnosis' in col_lower or 'diag' in col_lower or 'group' in col_lower:
                diag_col = col
                break
        
        if diag_col:
            print(f"  Using '{diag_col}' for diagnosis")
            diag_dict = df_meta_full.set_index('patient_id_normalized')[diag_col].to_dict()
            df_all_features['Diagnosis'] = df_all_features['Patient_ID'].map(diag_dict)
            
            # Get unique diagnoses
            mask_diag = df_all_features['Diagnosis'].notna()
            if mask_diag.sum() > 0:
                diagnoses = sorted([d for d in df_all_features.loc[mask_diag, 'Diagnosis'].unique()])
                palette_diag = sns.color_palette('Set2', len(diagnoses))
                color_map_diag = dict(zip(diagnoses, palette_diag))
                
                for diag in diagnoses:
                    mask = (df_all_features['Diagnosis'] == diag) & mask_diag
                    if mask.sum() > 0:
                        axes[1, 1].scatter(
                            df_all_features.loc[mask, 'UMAP1'],
                            df_all_features.loc[mask, 'UMAP2'],
                            c=[color_map_diag[diag]],
                            label=str(diag),
                            alpha=0.7,
                            s=50,
                            edgecolors='white',
                            linewidths=0.5
                        )
                
                axes[1, 1].set_xlabel('UMAP 1', fontsize=12)
                axes[1, 1].set_ylabel('UMAP 2', fontsize=12)
                axes[1, 1].set_title(f'Colored by Diagnosis (n={mask_diag.sum()})', fontsize=14, fontweight='bold')
                axes[1, 1].legend(title='Diagnosis', fontsize=9, loc='best', ncol=2 if len(diagnoses) > 6 else 1)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].axis('off')
                axes[1, 1].text(0.5, 0.5, 'No diagnosis data available', 
                               ha='center', va='center', fontsize=14)
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, f'Diagnosis column not found\nAvailable: {df_meta_full.columns.tolist()}', 
                           ha='center', va='center', fontsize=10, wrap=True)
        
        plt.suptitle('NEW DATA: UMAP with ALL Features (761 regions from all atlases)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path_all_feat = Path(output_dir) / 'umap_new_data_all_features.png'
        plt.savefig(output_path_all_feat, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path_all_feat}")
        plt.close()
        
    except Exception as e:
        print(f"  Warning: Could not create all-features UMAP: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nData Source Distribution:")
    print(df_plot['Source'].value_counts())
    
    print("\nDataset Distribution:")
    print(df_plot['Dataset'].value_counts())
    
    print("\nDataset Distribution by Source:")
    print(pd.crosstab(df_plot['Dataset'], df_plot['Source']))
    
    return df_plot

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*70)
    print("CAT12 UMAP COMPARISON: Old vs New Data")
    print("="*70)
    
    # Use the directory where the script is located, not the script path itself
    import os
    import sys
    
    if OUTPUT_DIR == ".":
        output_dir = os.getcwd()
    else:
        output_dir = OUTPUT_DIR
    
    # If output_dir is a file (the script itself), use its parent directory
    if os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir)
    
    print(f"\nOutput directory: {output_dir}")
    
    try:
        # Load data
        df_new = load_new_data(NEW_DATA_PATH)
        df_old = load_old_data(OLD_DATA_DIR)
        metadata_dict = load_metadata(METADATA_PATH)
        
        # Match patients
        df_new_matched, df_old_matched, common_patients = match_and_align_data(df_new, df_old)
        
        # Create UMAP visualizations
        df_plot = create_combined_umap(
            df_new_matched, 
            df_old_matched, 
            common_patients,
            metadata_dict,
            output_dir,
            METADATA_PATH
        )
        
        # Save combined data for reference
        output_csv = Path(output_dir) / 'umap_embeddings.csv'
        df_plot.to_csv(output_csv, index=False)
        print(f"\nSaved UMAP embeddings to: {output_csv}")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nOutput files saved to: {output_dir}")
        print("  - umap_comparison_by_source.png (Combined UMAP: Old vs New)")
        print("  - umap_comparison_by_dataset.png (Combined UMAP: By Dataset)")
        print("  - umap_old_vs_new_separate.png (Same embedding, split view)")
        print("  - umap_old_vs_new_independent.png (Independent embeddings, ALL new data)")
        print("  - umap_old_vs_new_by_age.png (Colored by Age)")
        print("  - umap_old_vs_new_by_sex.png (Colored by Sex)")
        print("  - umap_new_data_all_features.png (NEW data with ALL 761 features!)")
        print("  - umap_embeddings.csv")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())