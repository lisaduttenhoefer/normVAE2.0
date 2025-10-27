"""
preprocessing_normalize_multimodal.py

This script:
1. Loads your FIXED train/test split metadata
2. Identifies Training-HC subjects
3. Calculates normalization stats ONLY from Training-HC
4. Normalizes ALL subjects with these HC-only stats
5. Saves normalized CSV for future use

Run this ONCE, then use the normalized CSV for all training/testing!
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
    
    if atlases is None:
        atlases = ['neuromorphometrics', 'lpba40', 'cobra', 'suit', 
                   'ibsr', 'aal3', 'schaefer100', 'schaefer200', 
                   'aparc_dk40', 'aparc_destrieux']
    
    # Mapping of atlas names to prefixes in column names
    atlas_prefixes = {
        'neuromorphometrics': 'Neurom',
        'lpba40': 'lpba40',
        'cobra': 'cobra',
        'suit': 'SUIT',
        'ibsr': 'IBSR',
        'aal3': 'AAL3',
        'schaefer100': 'Sch100',
        'schaefer200': 'Sch200',
        'aparc_dk40': 'DK40',
        'aparc_destrieux': 'Destrieux'
    }
    
    volume_types = {
        'Vgm': [],
        'G': [],
        'T': []
    }
    
    for col in mri_data.columns:
        # Skip non-ROI columns
        if col in ['Filename', 'Dataset', 'IQR', 'NCR', 'TIV']:
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
    train_meta: pd.DataFrame,
    volume_type_columns: Dict[str, List[str]]
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Calculate mean and std from Training-HC ONLY.
    
    Returns:
        {
            'Vgm': {'mean': Series, 'std': Series},
            'G': {'mean': Series, 'std': Series},
            'T': {'mean': Series, 'std': Series}
        }
    """
    
    print("\n" + "="*80)
    print("STEP 3: Calculating HC-Only Statistics")
    print("="*80)
    
    # Identify Training-HC subjects
    train_hc = train_meta[train_meta['Diagnosis'] == 'HC']
    train_hc_filenames = train_hc['Filename'].values
    print(f"✓ Found {len(train_hc_filenames)} Training-HC subjects")
    
    # Filter MRI data to Training-HC
    train_hc_mask = mri_data['Filename'].isin(train_hc_filenames)
    train_hc_data = mri_data[train_hc_mask]
    print(f"✓ Matched {train_hc_mask.sum()}/{len(train_hc_filenames)} in MRI data")
    
    if train_hc_mask.sum() < len(train_hc_filenames):
        missing = len(train_hc_filenames) - train_hc_mask.sum()
        print(f"⚠ WARNING: {missing} Training-HC subjects not found in MRI data")
    
    # Calculate stats per volume type
    hc_stats = {}
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        print(f"\nProcessing {vtype}:")
        
        # Get HC data for this volume type
        vtype_hc_data = train_hc_data[columns]
        
        # TIV normalization (only for Vgm)
        if vtype == 'Vgm' and 'TIV' in train_hc_data.columns:
            print(f"  - Applying TIV normalization")
            tiv = train_hc_data['TIV'].values.reshape(-1, 1)
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


def normalize_all_data(
    mri_data: pd.DataFrame,
    volume_type_columns: Dict[str, List[str]],
    hc_stats: Dict[str, Dict[str, pd.Series]]
) -> pd.DataFrame:
    """
    Normalize ALL subjects using HC-only stats.
    """
    
    print("\n" + "="*80)
    print("STEP 4: Normalizing ALL Subjects with HC Stats")
    print("="*80)
    
    # Create a copy
    normalized_data = mri_data.copy()
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        print(f"\nNormalizing {vtype}:")
        
        # Get data for this volume type
        vtype_data = normalized_data[columns].copy()
        
        # TIV normalization (only for Vgm)
        if vtype == 'Vgm' and 'TIV' in normalized_data.columns:
            print(f"  - Applying TIV normalization")
            tiv = normalized_data['TIV'].values.reshape(-1, 1)
            vtype_data = vtype_data.div(tiv, axis=0)
        
        # Z-score normalization with HC stats
        means = hc_stats[vtype]['mean']
        stds = hc_stats[vtype]['std']
        
        vtype_normalized = (vtype_data - means) / stds
        
        # Replace in normalized_data
        normalized_data[columns] = vtype_normalized
        
        print(f"  ✓ Normalized {len(columns)} features")
        print(f"  - Value range: [{vtype_normalized.min().min():.3f}, {vtype_normalized.max().max():.3f}]")
    
    return normalized_data


def validate_normalization(
    normalized_data: pd.DataFrame,
    train_meta: pd.DataFrame,
    volume_type_columns: Dict[str, List[str]]
):
    """
    Validate that Training-HC has mean≈0, std≈1.
    """
    
    print("\n" + "="*80)
    print("STEP 5: Validation")
    print("="*80)
    
    # Get Training-HC from normalized data
    train_hc_filenames = train_meta[train_meta['Diagnosis'] == 'HC']['Filename'].values
    train_hc_mask = normalized_data['Filename'].isin(train_hc_filenames)
    train_hc_normalized = normalized_data[train_hc_mask]
    
    print(f"\nValidating Training-HC (n={train_hc_mask.sum()}):")
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        vtype_data = train_hc_normalized[columns]
        
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
    normalized_data: pd.DataFrame,
    hc_stats: Dict,
    output_dir: str,
    prefix: str = "columnwise_HC"
):
    """
    Save normalized data and stats.
    """
    
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save normalized CSV
    csv_path = os.path.join(output_dir, f"CAT12_results_NORMALIZED_{prefix}.csv")
    normalized_data.to_csv(csv_path, index=False)
    print(f"✓ Saved normalized CSV: {csv_path}")
    
    # Save stats
    stats_path = os.path.join(output_dir, f"normalization_stats_{prefix}.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(hc_stats, f)
    print(f"✓ Saved normalization stats: {stats_path}")
    
    # Save human-readable stats
    stats_txt_path = os.path.join(output_dir, f"normalization_stats_{prefix}.txt")
    with open(stats_txt_path, 'w') as f:
        f.write("HC-ONLY NORMALIZATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for vtype in ['Vgm', 'G', 'T']:
            if vtype not in hc_stats:
                continue
            
            f.write(f"\n{vtype}:\n")
            f.write(f"  Number of features: {len(hc_stats[vtype]['mean'])}\n")
            f.write(f"  Mean range: [{hc_stats[vtype]['mean'].min():.6f}, {hc_stats[vtype]['mean'].max():.6f}]\n")
            f.write(f"  Std range: [{hc_stats[vtype]['std'].min():.6f}, {hc_stats[vtype]['std'].max():.6f}]\n")
    
    print(f"✓ Saved human-readable stats: {stats_txt_path}")
    
    print("\n" + "="*80)
    print("✅ PRE-PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nNormalized CSV: {csv_path}")
    print(f"Use this file for ALL future training and testing!")
    print("="*80)


def main():
    """Main preprocessing pipeline."""
    
    # Configuration
    MRI_CSV_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    TRAIN_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/train_metadataHC_0.720251024_1927.csv"
    TEST_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/test_metadataHC_0.720251024_1927.csv"
    OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    print("\n" + "="*80)
    print("MULTIMODAL HC-ONLY NORMALIZATION PREPROCESSING")
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
    
    # Step 3: Calculate HC-only stats
    hc_stats = calculate_hc_stats(mri_data, train_meta, volume_type_columns)
    
    # Step 4: Normalize all data
    normalized_data = normalize_all_data(mri_data, volume_type_columns, hc_stats)
    
    # Step 5: Validate
    validate_normalization(normalized_data, train_meta, volume_type_columns)
    
    # Step 6: Save
    save_results(normalized_data, hc_stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()