"""
preprocessing_normalize_multimodal_SEPARATE.py

CORRECTED VERSION:
- Training data normalized with Training-HC stats
- Testing data normalized with Testing-HC stats
- NO information leakage between train/test!
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import os

def load_and_prepare_data(
    mri_csv_path: str,
    train_meta_path: str,
    test_meta_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MRI data and metadata."""
    
    print("="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    # Load MRI data
    mri_data = pd.read_csv(mri_csv_path)
    print(f"✓ Loaded MRI data: {mri_data.shape}")
    
    # Load metadata
    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)
    print(f"✓ Loaded train metadata: {len(train_meta)} subjects")
    print(f"✓ Loaded test metadata: {len(test_meta)} subjects")
    
    return mri_data, train_meta, test_meta


def identify_roi_columns(
    mri_data: pd.DataFrame,
    atlases: List[str] = None
) -> Dict[str, List[str]]:
    """
    Identify ROI columns for each volume type.
    
    Returns:
        {'Vgm': [col1, col2, ...], 'G': [...], 'T': [...]}
    """
    
    print("\n" + "="*80)
    print("STEP 2: Identifying ROI Columns")
    print("="*80)
    
    volume_types = {
        'Vgm': [],
        'G': [],
        'T': []
    }
    
    for col in mri_data.columns:
        # Skip non-ROI columns
        if col in ['Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV',
                   'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol',
                   'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
                   'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global']:
            continue
        
        # Check volume type
        for vtype in ['Vgm', 'G', 'T']:
            if col.startswith(f'{vtype}_'):
                volume_types[vtype].append(col)
                break
    
    print(f"✓ Found {len(volume_types['Vgm'])} Vgm columns")
    print(f"✓ Found {len(volume_types['G'])} G columns")
    print(f"✓ Found {len(volume_types['T'])} T columns")
    print(f"✓ Total ROI columns: {sum(len(v) for v in volume_types.values())}")
    
    return volume_types


def calculate_hc_stats(
    mri_data: pd.DataFrame,
    hc_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    split_name: str = "Split"
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Calculate mean and std from HC subjects ONLY.
    
    Args:
        mri_data: Full MRI data
        hc_filenames: List of HC subject filenames
        volume_type_columns: Dict of volume type columns
        split_name: "Training" or "Testing" (for logging)
    
    Returns:
        {
            'Vgm': {'mean': Series, 'std': Series},
            'G': {'mean': Series, 'std': Series},
            'T': {'mean': Series, 'std': Series}
        }
    """
    
    print(f"\n{'='*80}")
    print(f"Calculating {split_name}-HC Statistics")
    print(f"{'='*80}")
    
    print(f"✓ {split_name}-HC subjects: {len(hc_filenames)}")
    
    # Filter MRI data to HC subjects
    hc_mask = mri_data['Filename'].isin(hc_filenames)
    hc_data = mri_data[hc_mask]
    print(f"✓ Matched {hc_mask.sum()}/{len(hc_filenames)} in MRI data")
    
    if hc_mask.sum() < len(hc_filenames):
        missing = len(hc_filenames) - hc_mask.sum()
        print(f"⚠ WARNING: {missing} {split_name}-HC subjects not found in MRI data")
    
    # Calculate stats per volume type
    hc_stats = {}
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        print(f"\nProcessing {vtype}:")
        
        # Get HC data for this volume type
        vtype_hc_data = hc_data[columns].copy()
        
        # TIV normalization (only for Vgm)
        if vtype == 'Vgm' and 'TIV' in hc_data.columns:
            print(f"  - Applying TIV normalization")
            tiv = hc_data['TIV'].values.reshape(-1, 1)
            vtype_hc_data = vtype_hc_data.div(tiv, axis=0)
        
        # Calculate mean and std
        means = vtype_hc_data.mean(axis=0)
        stds = vtype_hc_data.std(axis=0)
        
        # Handle zero std (constant values)
        zero_std_cols = stds[stds == 0].index.tolist()
        if zero_std_cols:
            print(f"  ⚠ WARNING: {len(zero_std_cols)} columns with std=0")
            stds = stds.replace(0, 1)  # Avoid division by zero
        
        hc_stats[vtype] = {
            'mean': means,
            'std': stds
        }
        
        print(f"  ✓ Stats calculated for {len(columns)} features")
        print(f"  - Mean range: [{means.min():.6f}, {means.max():.6f}]")
        print(f"  - Std range: [{stds.min():.6f}, {stds.max():.6f}]")
    
    return hc_stats


def normalize_split_data(
    mri_data: pd.DataFrame,
    split_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    hc_stats: Dict[str, Dict[str, pd.Series]],
    split_name: str = "Split"
) -> pd.DataFrame:
    """
    Normalize subjects from one split using its own HC stats.
    
    Args:
        mri_data: Full MRI data
        split_filenames: List of filenames in this split
        volume_type_columns: Dict of volume type columns
        hc_stats: HC statistics for this split
        split_name: "Training" or "Testing" (for logging)
    
    Returns:
        Normalized data for this split only
    """
    
    print(f"\n{'='*80}")
    print(f"Normalizing {split_name} Data")
    print(f"{'='*80}")
    
    # Filter to split subjects
    split_mask = mri_data['Filename'].isin(split_filenames)
    split_data = mri_data[split_mask].copy()
    print(f"✓ Found {split_mask.sum()}/{len(split_filenames)} subjects in MRI data")
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        print(f"\nNormalizing {vtype}:")
        
        # Get data for this volume type
        vtype_data = split_data[columns].copy()
        
        # TIV normalization (only for Vgm)
        if vtype == 'Vgm' and 'TIV' in split_data.columns:
            print(f"  - Applying TIV normalization")
            tiv = split_data['TIV'].values.reshape(-1, 1)
            vtype_data = vtype_data.div(tiv, axis=0)
        
        # Z-score normalization with HC stats
        means = hc_stats[vtype]['mean']
        stds = hc_stats[vtype]['std']
        
        vtype_normalized = (vtype_data - means) / stds
        
        # Replace in split_data
        split_data[columns] = vtype_normalized
        
        print(f"  ✓ Normalized {len(columns)} features")
        print(f"  - Value range: [{vtype_normalized.min().min():.3f}, {vtype_normalized.max().max():.3f}]")
    
    return split_data


def validate_normalization(
    normalized_data: pd.DataFrame,
    hc_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    split_name: str = "Split"
):
    """
    Validate that HC has mean≈0, std≈1.
    """
    
    print(f"\n{'='*80}")
    print(f"Validation: {split_name}-HC")
    print(f"{'='*80}")
    
    # Get HC from normalized data
    hc_mask = normalized_data['Filename'].isin(hc_filenames)
    hc_normalized = normalized_data[hc_mask]
    
    print(f"\nValidating {split_name}-HC (n={hc_mask.sum()}):")
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        vtype_data = hc_normalized[columns]
        
        mean_of_means = vtype_data.mean().mean()
        mean_of_stds = vtype_data.std().mean()
        
        print(f"\n{vtype}:")
        print(f"  Mean of means: {mean_of_means:.6f} (should be ≈0)")
        print(f"  Mean of stds:  {mean_of_stds:.6f} (should be ≈1)")
        
        if abs(mean_of_means) > 0.01:
            print(f"  ⚠ WARNING: Mean is not close to 0!")
        if abs(mean_of_stds - 1) > 0.1:
            print(f"  ⚠ WARNING: Std is not close to 1!")


def save_results(
    train_normalized: pd.DataFrame,
    test_normalized: pd.DataFrame,
    train_hc_stats: Dict,
    test_hc_stats: Dict,
    output_dir: str,
    prefix: str = "columnwise_HC_separate"
):
    """
    Save normalized data and stats for both splits.
    """
    
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TRAINING normalized CSV
    train_csv_path = os.path.join(output_dir, f"CAT12_results_NORMALIZED_{prefix}_TRAIN.csv")
    train_normalized.to_csv(train_csv_path, index=False)
    print(f"✓ Saved TRAINING normalized CSV: {train_csv_path}")
    
    # Save TESTING normalized CSV
    test_csv_path = os.path.join(output_dir, f"CAT12_results_NORMALIZED_{prefix}_TEST.csv")
    test_normalized.to_csv(test_csv_path, index=False)
    print(f"✓ Saved TESTING normalized CSV: {test_csv_path}")
    
    # Save training stats
    train_stats_path = os.path.join(output_dir, f"normalization_stats_{prefix}_TRAIN.pkl")
    with open(train_stats_path, 'wb') as f:
        pickle.dump(train_hc_stats, f)
    print(f"✓ Saved TRAINING normalization stats: {train_stats_path}")
    
    # Save testing stats
    test_stats_path = os.path.join(output_dir, f"normalization_stats_{prefix}_TEST.pkl")
    with open(test_stats_path, 'wb') as f:
        pickle.dump(test_hc_stats, f)
    print(f"✓ Saved TESTING normalization stats: {test_stats_path}")
    
    # Save human-readable comparison
    comparison_path = os.path.join(output_dir, f"normalization_stats_{prefix}_COMPARISON.txt")
    with open(comparison_path, 'w') as f:
        f.write("SEPARATE HC-ONLY NORMALIZATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write("Training and Testing are normalized SEPARATELY with their own HC stats.\n")
        f.write("This prevents information leakage between train/test splits.\n\n")
        
        for vtype in ['Vgm', 'G', 'T']:
            if vtype not in train_hc_stats:
                continue
            
            f.write(f"\n{vtype}:\n")
            f.write(f"-"*80 + "\n")
            
            # Training stats
            f.write(f"TRAINING-HC:\n")
            f.write(f"  Number of features: {len(train_hc_stats[vtype]['mean'])}\n")
            f.write(f"  Mean range: [{train_hc_stats[vtype]['mean'].min():.6f}, {train_hc_stats[vtype]['mean'].max():.6f}]\n")
            f.write(f"  Std range: [{train_hc_stats[vtype]['std'].min():.6f}, {train_hc_stats[vtype]['std'].max():.6f}]\n")
            
            # Testing stats
            f.write(f"\nTESTING-HC:\n")
            f.write(f"  Number of features: {len(test_hc_stats[vtype]['mean'])}\n")
            f.write(f"  Mean range: [{test_hc_stats[vtype]['mean'].min():.6f}, {test_hc_stats[vtype]['mean'].max():.6f}]\n")
            f.write(f"  Std range: [{test_hc_stats[vtype]['std'].min():.6f}, {test_hc_stats[vtype]['std'].max():.6f}]\n")
            
            # Difference
            mean_diff = np.abs(train_hc_stats[vtype]['mean'] - test_hc_stats[vtype]['mean']).mean()
            std_diff = np.abs(train_hc_stats[vtype]['std'] - test_hc_stats[vtype]['std']).mean()
            f.write(f"\nDIFFERENCE (Train vs Test):\n")
            f.write(f"  Mean difference (avg): {mean_diff:.6f}\n")
            f.write(f"  Std difference (avg): {std_diff:.6f}\n")
    
    print(f"✓ Saved comparison stats: {comparison_path}")
    
    print("\n" + "="*80)
    print("✅ PRE-PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nTraining CSV: {train_csv_path}")
    print(f"Testing CSV: {test_csv_path}")
    print(f"\n⚠️ IMPORTANT: Use TRAINING CSV for training, TESTING CSV for testing!")
    print("="*80)


def main():
    """Main preprocessing pipeline with SEPARATE normalization."""
    
    # Configuration
    MRI_CSV_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    TRAIN_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/train_metadataHC_0.720251103_1438.csv"
    TEST_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/test_metadataHC_0.720251103_1438.csv"
    OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    print("\n" + "="*80)
    print("MULTIMODAL HC-ONLY NORMALIZATION (SEPARATE TRAIN/TEST)")
    print("="*80)
    print(f"\nMRI Data: {MRI_CSV_PATH}")
    print(f"Train Metadata: {TRAIN_META_PATH}")
    print(f"Test Metadata: {TEST_META_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    # Step 1: Load data
    mri_data, train_meta, test_meta = load_and_prepare_data(
        MRI_CSV_PATH, TRAIN_META_PATH, TEST_META_PATH
    )
    
    # Step 2: Identify ROI columns
    volume_type_columns = identify_roi_columns(mri_data)
    
    # ========== TRAINING SPLIT ==========
    print("\n" + "="*80)
    print("PROCESSING TRAINING SPLIT")
    print("="*80)
    
    # Get Training-HC filenames
    train_hc = train_meta[train_meta['Diagnosis'] == 'HC']
    train_hc_filenames = train_hc['Filename'].values
    
    # Get all Training filenames
    train_all_filenames = train_meta['Filename'].values
    
    # Calculate Training-HC stats
    train_hc_stats = calculate_hc_stats(
        mri_data, train_hc_filenames, volume_type_columns, split_name="Training"
    )
    
    # Normalize Training data with Training-HC stats
    train_normalized = normalize_split_data(
        mri_data, train_all_filenames, volume_type_columns, 
        train_hc_stats, split_name="Training"
    )
    
    # Validate Training
    validate_normalization(
        train_normalized, train_hc_filenames, volume_type_columns, split_name="Training"
    )
    
    # ========== TESTING SPLIT ==========
    print("\n" + "="*80)
    print("PROCESSING TESTING SPLIT")
    print("="*80)
    
    # Get Testing-HC filenames
    test_hc = test_meta[test_meta['Diagnosis'] == 'HC']
    test_hc_filenames = test_hc['Filename'].values
    
    # Get all Testing filenames
    test_all_filenames = test_meta['Filename'].values
    
    # Calculate Testing-HC stats
    test_hc_stats = calculate_hc_stats(
        mri_data, test_hc_filenames, volume_type_columns, split_name="Testing"
    )
    
    # Normalize Testing data with Testing-HC stats
    test_normalized = normalize_split_data(
        mri_data, test_all_filenames, volume_type_columns, 
        test_hc_stats, split_name="Testing"
    )
    
    # Validate Testing
    validate_normalization(
        test_normalized, test_hc_filenames, volume_type_columns, split_name="Testing"
    )
    
    # ========== SAVE RESULTS ==========
    save_results(
        train_normalized, test_normalized,
        train_hc_stats, test_hc_stats,
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()