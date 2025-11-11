"""
preprocessing_normalize_IQR_SEPARATE.py

IQR-BASED ROBUST NORMALIZATION (Paper Method)
- Uses MEDIAN and IQR instead of MEAN and STD
- Robust to outliers (important for neuroimaging!)
- Training data normalized with Training-HC stats
- Testing data normalized with Testing-HC stats
- NO information leakage between train/test

Based on: Pinaya et al. (2022) normative modeling approach
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import os
from pathlib import Path


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
        {'Vgm': [col1, col2, ...], 'Vwm': [...], 'Vcsf': [...], 'G': [...], 'T': [...]}
    """
    
    print("\n" + "="*80)
    print("STEP 2: Identifying ROI Columns")
    print("="*80)
    
    volume_types = {
        'Vgm': [],
        'Vwm': [],
        'Vcsf': [],
        'G': [],
        'T': []
    }
    
    # Non-ROI columns to skip
    skip_cols = [
        'Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV',
        'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol',
        'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
        'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global'
    ]
    
    for col in mri_data.columns:
        # Skip non-ROI columns
        if col in skip_cols:
            continue
        
        # Check volume type
        for vtype in ['Vgm', 'Vwm', 'Vcsf', 'G', 'T']:
            if col.startswith(f'{vtype}_'):
                volume_types[vtype].append(col)
                break
    
    # Print summary
    total_rois = 0
    for vtype, cols in volume_types.items():
        if len(cols) > 0:
            print(f"✓ Found {len(cols)} {vtype} columns")
            total_rois += len(cols)
    
    print(f"✓ Total ROI columns: {total_rois}")
    
    return volume_types


def calculate_hc_stats_iqr(
    mri_data: pd.DataFrame,
    hc_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    split_name: str = "Split"
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Calculate MEDIAN and IQR from HC subjects ONLY.
    
    IQR-based robust normalization:
    - Uses MEDIAN instead of MEAN (robust to outliers)
    - Uses IQR instead of STD (robust to outliers)
    - Better for neuroimaging data with outliers
    
    Args:
        mri_data: Full MRI data
        hc_filenames: List of HC subject filenames
        volume_type_columns: Dict of volume type columns
        split_name: "Training" or "Testing" (for logging)
    
    Returns:
        {
            'Vgm': {'median': Series, 'iqr': Series, 'q1': Series, 'q3': Series},
            'G': {...},
            'T': {...}
        }
    """
    
    print(f"\n{'='*80}")
    print(f"Calculating {split_name}-HC Statistics (IQR METHOD)")
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
        
        # TIV normalization (only for volume types, not thickness/gyrification)
        if vtype in ['Vgm', 'Vwm', 'Vcsf'] and 'TIV' in hc_data.columns:
            print(f"  - Applying TIV normalization (divide by TIV)")
            tiv = hc_data['TIV'].values.reshape(-1, 1)
            vtype_hc_data = vtype_hc_data.div(tiv, axis=0)
        
        # Calculate MEDIAN and IQR (robust statistics!)
        medians = vtype_hc_data.median(axis=0)
        q1 = vtype_hc_data.quantile(0.25, axis=0)
        q3 = vtype_hc_data.quantile(0.75, axis=0)
        iqr = q3 - q1
        
        # Handle zero IQR (constant values across subjects)
        zero_iqr_cols = iqr[iqr == 0].index.tolist()
        if zero_iqr_cols:
            print(f"  ⚠ WARNING: {len(zero_iqr_cols)} columns with IQR=0 (constant)")
            print(f"    Setting IQR=1 to avoid division by zero")
            iqr = iqr.replace(0, 1)
        
        # Also check for very small IQR (might cause instability)
        small_iqr_threshold = 1e-6
        small_iqr_cols = iqr[iqr < small_iqr_threshold].index.tolist()
        if small_iqr_cols:
            print(f"  ⚠ WARNING: {len(small_iqr_cols)} columns with very small IQR (<{small_iqr_threshold})")
            iqr = iqr.clip(lower=small_iqr_threshold)
        
        hc_stats[vtype] = {
            'median': medians,
            'iqr': iqr,
            'q1': q1,
            'q3': q3
        }
        
        print(f"  ✓ Stats calculated for {len(columns)} features")
        print(f"  - Median range: [{medians.min():.6f}, {medians.max():.6f}]")
        print(f"  - IQR range: [{iqr.min():.6f}, {iqr.max():.6f}]")
        print(f"  - Q1 range: [{q1.min():.6f}, {q1.max():.6f}]")
        print(f"  - Q3 range: [{q3.min():.6f}, {q3.max():.6f}]")
    
    return hc_stats


def normalize_split_data_iqr(
    mri_data: pd.DataFrame,
    split_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    hc_stats: Dict[str, Dict[str, pd.Series]],
    split_name: str = "Split"
) -> pd.DataFrame:
    """
    Normalize subjects using IQR-based robust scaling.
    
    Formula: normalized = (x - median_HC) / IQR_HC
    
    Where:
    - median_HC: Median of HC subjects for this feature
    - IQR_HC: Interquartile range (Q3 - Q1) of HC subjects
    
    Args:
        mri_data: Full MRI data
        split_filenames: List of filenames in this split
        volume_type_columns: Dict of volume type columns
        hc_stats: HC statistics for this split (from calculate_hc_stats_iqr)
        split_name: "Training" or "Testing" (for logging)
    
    Returns:
        Normalized data for this split only
    """
    
    print(f"\n{'='*80}")
    print(f"Normalizing {split_name} Data (IQR METHOD)")
    print(f"{'='*80}")
    
    # Filter to split subjects
    split_mask = mri_data['Filename'].isin(split_filenames)
    split_data = mri_data[split_mask].copy()
    print(f"✓ Found {split_mask.sum()}/{len(split_filenames)} subjects in MRI data")
    
    if split_mask.sum() < len(split_filenames):
        missing = len(split_filenames) - split_mask.sum()
        print(f"⚠ WARNING: {missing} subjects not found in MRI data")
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        print(f"\nNormalizing {vtype}:")
        
        # Get data for this volume type
        vtype_data = split_data[columns].copy()
        
        # TIV normalization (only for volumes)
        if vtype in ['Vgm', 'Vwm', 'Vcsf'] and 'TIV' in split_data.columns:
            print(f"  - Applying TIV normalization")
            tiv = split_data['TIV'].values.reshape(-1, 1)
            vtype_data = vtype_data.div(tiv, axis=0)
        
        # IQR-based robust scaling
        medians = hc_stats[vtype]['median']
        iqr = hc_stats[vtype]['iqr']
        
        # Formula: (x - median) / IQR
        vtype_normalized = (vtype_data - medians) / iqr
        
        # Replace in split_data
        split_data[columns] = vtype_normalized
        
        print(f"  ✓ Normalized {len(columns)} features using IQR")
        print(f"  - Value range: [{vtype_normalized.min().min():.3f}, {vtype_normalized.max().max():.3f}]")
        print(f"  - Mean: {vtype_normalized.mean().mean():.3f}")
        print(f"  - Std: {vtype_normalized.std().mean():.3f}")
    
    return split_data


def validate_normalization_iqr(
    normalized_data: pd.DataFrame,
    hc_filenames: List[str],
    volume_type_columns: Dict[str, List[str]],
    split_name: str = "Split"
):
    """
    Validate IQR-based normalization.
    
    For HC subjects:
    - Median should be ≈ 0
    - IQR should be ≈ 1
    
    This is the expected result of IQR normalization.
    """
    
    print(f"\n{'='*80}")
    print(f"Validation: {split_name}-HC (IQR NORMALIZATION)")
    print(f"{'='*80}")
    
    # Get HC from normalized data
    hc_mask = normalized_data['Filename'].isin(hc_filenames)
    hc_normalized = normalized_data[hc_mask]
    
    print(f"\nValidating {split_name}-HC (n={hc_mask.sum()}):")
    print("Expected: Median ≈ 0, IQR ≈ 1\n")
    
    all_good = True
    
    for vtype, columns in volume_type_columns.items():
        if not columns:
            continue
        
        vtype_data = hc_normalized[columns]
        
        # Calculate statistics
        median_of_medians = vtype_data.median().median()
        q1 = vtype_data.quantile(0.25)
        q3 = vtype_data.quantile(0.75)
        iqr_of_iqrs = (q3 - q1).median()
        
        # Also check mean and std for reference
        mean_of_means = vtype_data.mean().mean()
        mean_of_stds = vtype_data.std().mean()
        
        print(f"{vtype}:")
        print(f"  Median (should be ≈0): {median_of_medians:.6f}")
        print(f"  IQR (should be ≈1):    {iqr_of_iqrs:.6f}")
        print(f"  Mean (reference):      {mean_of_means:.6f}")
        print(f"  Std (reference):       {mean_of_stds:.6f}")
        
        # Validation checks
        if abs(median_of_medians) > 0.1:
            print(f"  ⚠ WARNING: Median is not close to 0!")
            all_good = False
        else:
            print(f"  ✓ Median check passed")
        
        if abs(iqr_of_iqrs - 1) > 0.2:
            print(f"  ⚠ WARNING: IQR is not close to 1!")
            all_good = False
        else:
            print(f"  ✓ IQR check passed")
        
        print()
    
    if all_good:
        print(f"✅ ALL VALIDATION CHECKS PASSED for {split_name}-HC!")
    else:
        print(f"⚠️  SOME VALIDATION CHECKS FAILED for {split_name}-HC!")


def save_results(
    train_normalized: pd.DataFrame,
    test_normalized: pd.DataFrame,
    train_hc_stats: Dict,
    test_hc_stats: Dict,
    output_dir: str,
    prefix: str = "IQR_HC_separate"
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
    print(f"  Shape: {train_normalized.shape}")
    
    # Save TESTING normalized CSV
    test_csv_path = os.path.join(output_dir, f"CAT12_results_NORMALIZED_{prefix}_TEST.csv")
    test_normalized.to_csv(test_csv_path, index=False)
    print(f"✓ Saved TESTING normalized CSV: {test_csv_path}")
    print(f"  Shape: {test_normalized.shape}")
    
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
    
    # Save human-readable summary
    summary_path = os.path.join(output_dir, f"normalization_summary_{prefix}.txt")
    with open(summary_path, 'w') as f:
        f.write("IQR-BASED ROBUST NORMALIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("METHOD: IQR-based robust scaling (Pinaya et al. 2022)\n")
        f.write("Formula: normalized = (x - median_HC) / IQR_HC\n\n")
        f.write("ADVANTAGES:\n")
        f.write("  - Robust to outliers (uses median instead of mean)\n")
        f.write("  - Preserves relative differences (uses IQR instead of std)\n")
        f.write("  - Better for neuroimaging data\n\n")
        f.write("Training and Testing are normalized SEPARATELY with their own HC stats.\n")
        f.write("This prevents information leakage between train/test splits.\n\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINING SET\n")
        f.write("="*80 + "\n")
        f.write(f"Total subjects: {len(train_normalized)}\n\n")
        
        for vtype in ['Vgm', 'Vwm', 'Vcsf', 'G', 'T']:
            if vtype not in train_hc_stats or len(train_hc_stats[vtype]['median']) == 0:
                continue
            
            f.write(f"\n{vtype}:\n")
            f.write(f"-"*80 + "\n")
            f.write(f"  Number of features: {len(train_hc_stats[vtype]['median'])}\n")
            f.write(f"  Median range: [{train_hc_stats[vtype]['median'].min():.6f}, {train_hc_stats[vtype]['median'].max():.6f}]\n")
            f.write(f"  IQR range: [{train_hc_stats[vtype]['iqr'].min():.6f}, {train_hc_stats[vtype]['iqr'].max():.6f}]\n")
            f.write(f"  Q1 range: [{train_hc_stats[vtype]['q1'].min():.6f}, {train_hc_stats[vtype]['q1'].max():.6f}]\n")
            f.write(f"  Q3 range: [{train_hc_stats[vtype]['q3'].min():.6f}, {train_hc_stats[vtype]['q3'].max():.6f}]\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TESTING SET\n")
        f.write("="*80 + "\n")
        f.write(f"Total subjects: {len(test_normalized)}\n\n")
        
        for vtype in ['Vgm', 'Vwm', 'Vcsf', 'G', 'T']:
            if vtype not in test_hc_stats or len(test_hc_stats[vtype]['median']) == 0:
                continue
            
            f.write(f"\n{vtype}:\n")
            f.write(f"-"*80 + "\n")
            f.write(f"  Number of features: {len(test_hc_stats[vtype]['median'])}\n")
            f.write(f"  Median range: [{test_hc_stats[vtype]['median'].min():.6f}, {test_hc_stats[vtype]['median'].max():.6f}]\n")
            f.write(f"  IQR range: [{test_hc_stats[vtype]['iqr'].min():.6f}, {test_hc_stats[vtype]['iqr'].max():.6f}]\n")
            f.write(f"  Q1 range: [{test_hc_stats[vtype]['q1'].min():.6f}, {test_hc_stats[vtype]['q1'].max():.6f}]\n")
            f.write(f"  Q3 range: [{test_hc_stats[vtype]['q3'].min():.6f}, {test_hc_stats[vtype]['q3'].max():.6f}]\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON (Train vs Test)\n")
        f.write("="*80 + "\n\n")
        
        for vtype in ['Vgm', 'Vwm', 'Vcsf', 'G', 'T']:
            if vtype not in train_hc_stats or vtype not in test_hc_stats:
                continue
            if len(train_hc_stats[vtype]['median']) == 0:
                continue
            
            f.write(f"\n{vtype}:\n")
            f.write(f"-"*80 + "\n")
            
            # Calculate differences
            median_diff = np.abs(train_hc_stats[vtype]['median'] - test_hc_stats[vtype]['median']).mean()
            iqr_diff = np.abs(train_hc_stats[vtype]['iqr'] - test_hc_stats[vtype]['iqr']).mean()
            
            f.write(f"  Mean absolute difference in medians: {median_diff:.6f}\n")
            f.write(f"  Mean absolute difference in IQR: {iqr_diff:.6f}\n")
            
            # Correlation
            median_corr = np.corrcoef(train_hc_stats[vtype]['median'], test_hc_stats[vtype]['median'])[0,1]
            iqr_corr = np.corrcoef(train_hc_stats[vtype]['iqr'], test_hc_stats[vtype]['iqr'])[0,1]
            
            f.write(f"  Correlation of medians: {median_corr:.4f}\n")
            f.write(f"  Correlation of IQR: {iqr_corr:.4f}\n")
    
    print(f"✓ Saved normalization summary: {summary_path}")
    
    # Save comparison CSV for easy inspection
    comparison_csv_path = os.path.join(output_dir, f"normalization_comparison_{prefix}.csv")
    comparison_rows = []
    
    for vtype in ['Vgm', 'Vwm', 'Vcsf', 'G', 'T']:
        if vtype not in train_hc_stats or vtype not in test_hc_stats:
            continue
        if len(train_hc_stats[vtype]['median']) == 0:
            continue
        
        for col in train_hc_stats[vtype]['median'].index:
            comparison_rows.append({
                'Volume_Type': vtype,
                'Feature': col,
                'Train_Median': train_hc_stats[vtype]['median'][col],
                'Test_Median': test_hc_stats[vtype]['median'][col],
                'Train_IQR': train_hc_stats[vtype]['iqr'][col],
                'Test_IQR': test_hc_stats[vtype]['iqr'][col],
                'Median_Diff': abs(train_hc_stats[vtype]['median'][col] - test_hc_stats[vtype]['median'][col]),
                'IQR_Diff': abs(train_hc_stats[vtype]['iqr'][col] - test_hc_stats[vtype]['iqr'][col])
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"✓ Saved comparison CSV: {comparison_csv_path}")
    
    print("\n" + "="*80)
    print("✅ IQR-BASED NORMALIZATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Key Files:")
    print(f"  Training CSV: {train_csv_path}")
    print(f"  Testing CSV:  {test_csv_path}")
    print(f"  Summary:      {summary_path}")
    print(f"\n⚠️ IMPORTANT:")
    print(f"  - Use TRAINING CSV for model training")
    print(f"  - Use TESTING CSV for model testing")
    print(f"  - NO information leakage between splits!")
    print("="*80)


def main():
    """Main preprocessing pipeline with IQR-based robust normalization."""
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    MRI_CSV_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
    TRAIN_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/train_metadataHC_0.720251103_1438.csv"
    TEST_META_PATH = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/test_metadataHC_0.720251103_1438.csv"
    OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training"
    
    print("\n" + "="*80)
    print("IQR-BASED ROBUST NORMALIZATION (SEPARATE TRAIN/TEST)")
    print("="*80)
    print("\nMethod: Pinaya et al. (2022) normative modeling approach")
    print("Formula: normalized = (x - median_HC) / IQR_HC")
    print("\nAdvantages:")
    print("  - Robust to outliers")
    print("  - Preserves natural variance")
    print("  - Better for neuroimaging data")
    print("\n" + "="*80)
    print(f"\nConfiguration:")
    print(f"  MRI Data: {MRI_CSV_PATH}")
    print(f"  Train Metadata: {TRAIN_META_PATH}")
    print(f"  Test Metadata: {TEST_META_PATH}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    # ============================================================================
    # STEP 1: Load data
    # ============================================================================
    
    mri_data, train_meta, test_meta = load_and_prepare_data(
        MRI_CSV_PATH, TRAIN_META_PATH, TEST_META_PATH
    )
    
    # ============================================================================
    # STEP 2: Identify ROI columns
    # ============================================================================
    
    volume_type_columns = identify_roi_columns(mri_data)
    
    # ============================================================================
    # STEP 3: Process TRAINING split
    # ============================================================================
    
    print("\n" + "="*80)
    print("PROCESSING TRAINING SPLIT")
    print("="*80)
    
    # Get Training-HC filenames
    train_hc = train_meta[train_meta['Diagnosis'] == 'HC']
    train_hc_filenames = train_hc['Filename'].values
    print(f"\nTraining-HC subjects: {len(train_hc_filenames)}")
    
    # Get all Training filenames
    train_all_filenames = train_meta['Filename'].values
    print(f"Training-ALL subjects: {len(train_all_filenames)}")
    
    # Calculate Training-HC stats (IQR method)
    train_hc_stats = calculate_hc_stats_iqr(
        mri_data, train_hc_filenames, volume_type_columns, split_name="Training"
    )
    
    # Normalize Training data with Training-HC stats
    train_normalized = normalize_split_data_iqr(
        mri_data, train_all_filenames, volume_type_columns, 
        train_hc_stats, split_name="Training"
    )
    
    # Validate Training normalization
    validate_normalization_iqr(
        train_normalized, train_hc_filenames, volume_type_columns, split_name="Training"
    )
    
    # ============================================================================
    # STEP 4: Process TESTING split
    # ============================================================================
    
    print("\n" + "="*80)
    print("PROCESSING TESTING SPLIT")
    print("="*80)
    
    # Get Testing-HC filenames
    test_hc = test_meta[test_meta['Diagnosis'] == 'HC']
    test_hc_filenames = test_hc['Filename'].values
    print(f"\nTesting-HC subjects: {len(test_hc_filenames)}")
    
    # Get all Testing filenames
    test_all_filenames = test_meta['Filename'].values
    print(f"Testing-ALL subjects: {len(test_all_filenames)}")
    
    # Calculate Testing-HC stats (IQR method)
    test_hc_stats = calculate_hc_stats_iqr(
        mri_data, test_hc_filenames, volume_type_columns, split_name="Testing"
    )
    
    # Normalize Testing data with Testing-HC stats
    test_normalized = normalize_split_data_iqr(
        mri_data, test_all_filenames, volume_type_columns, 
        test_hc_stats, split_name="Testing"
    )
    
    # Validate Testing normalization
    validate_normalization_iqr(
        test_normalized, test_hc_filenames, volume_type_columns, split_name="Testing"
    )
    
    # ============================================================================
    # STEP 5: Save results
    # ============================================================================
    
    save_results(
        train_normalized, test_normalized,
        train_hc_stats, test_hc_stats,
        OUTPUT_DIR, prefix="IQR_HC_separate"
    )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\n✅ IQR-based normalization completed successfully!")
    print("\n📊 Next Steps:")
    print("  1. Update your training script to use:")
    print(f"     CAT12_results_NORMALIZED_IQR_HC_separate_TRAIN.csv")
    print("  2. Update your testing script to use:")
    print(f"     CAT12_results_NORMALIZED_IQR_HC_separate_TEST.csv")
    print("  3. Re-train your CVAE model")
    print("  4. Check UMAP plots - should show better clustering!")
    print("\n💡 Expected improvements:")
    print("  - Better preservation of variance")
    print("  - Robust to outliers")
    print("  - VAE can learn meaningful patterns")
    print("  - UMAP should show clear HC vs Patient separation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()