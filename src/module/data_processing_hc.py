import os
from typing import List, Tuple
import pandas as pd
import regex as re
import anndata as ad
import pandas as pd
import scanpy as sc
from typing import List, Tuple
import numpy as np
import pathlib
import h5py
import torch
import torchio as tio
from torch.utils.data import DataLoader

from utils.logging_utils import log_and_print, log_checkpoint

def flatten_df(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array
    array = df.to_numpy()
    flat_array = array.flatten()
    return flat_array

def flatten_array(arr):
    """Flatten an array or series to 1D."""
    if isinstance(arr, pd.Series):
        return arr.values
    return arr.flatten() if hasattr(arr, 'flatten') else arr

def normalize_and_scale_og(df: pd.DataFrame) -> pd.DataFrame:
    # erst ne log transformation, dann ne reihenweise Z-Standarisierung (aber immer nur die columns mit gleichem volume-type)
    #results in scaled input df
    #currently in use

    df_copy = df.copy()
    column_sums = df_copy.sum()
    # Apply the formula: ln((10000*value)/sum_values + 1) "Log transformation"
    # Alternatively for Min-Max Scaling applied to the columns: df_copy/df_copy.max() - Problem: Some rows have std = 0
    transformed_df = np.log((10000 * df_copy) / column_sums + 1)

    norm_copy = transformed_df.copy()

    cols = norm_copy.columns.get_level_values(-1).tolist() # Select lowest level of Multiindex (volume_types)
    unique_cols = list(set(cols))

    for col_type in unique_cols:  # Z-scale each of the measurement types separately
        cols_to_scale = [col for col in norm_copy.columns if col[-1] == col_type]

        # Scale the selected columns per row
        scaled = norm_copy[cols_to_scale].apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
            axis=1
        )
        
        norm_copy.loc[:, cols_to_scale] = scaled

    return norm_copy

def get_all_data(directory: str, ext: str = "h5") -> list:
    #to get all files in a directory that are h5 files
    data_paths = list(pathlib.Path(directory).rglob(f"*.{ext}"))
    return data_paths


def get_atlas(path: pathlib.PosixPath) -> str:
    stem = path.stem
    match = re.search(r"_(.*)", stem)
    if match:
        atlas = match.group(1)
    return atlas


def combine_dfs(paths: list):
    # Combines any number of csv files to a single pandas DataFrame, keeping only shared column indices (-> "inner")
    
    if len(paths) > 1: 
        for i in range(1,len(paths)):
            if i == 1: 
                joined_df = pd.read_csv(paths[i-1], header=[0])
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")  
            else:
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")
    else: 
        joined_df = pd.read_csv(paths[0], header=[0])
    return joined_df

def read_hdf5_to_df(filepath: str) -> pd.DataFrame:
    #hdf5 to df transformation (tries hdf5 and hdf method)
    try:
        with h5py.File(filepath, 'r') as f:
           
            # Assuming HDF5 file contains 'data' and 'index' datasets
            if 'data' in f and 'index' in f and 'columns' in f:
                data = f['data'][:]
                index = [idx.decode('utf-8') if isinstance(idx, bytes) else idx for idx in f['index'][:]]
                columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in f['columns'][:]]
                
                df = pd.DataFrame(data, index=index, columns=columns)
            else:
                # Try to load with pandas directly
                df = pd.read_hdf(filepath)
                
            return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load HDF5 file {filepath}: {e}")
        try:
            df = pd.read_hdf(filepath)
            return df
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}")
            raise

#splitting the NORM patients in training and validation subsets
def train_val_split_annotations(
    # The annotations dataframe (metadata)that you want to split into a train and validation part
    annotations: pd.DataFrame,
    # The proportion of the data that should be in the training set (the rest is in the test set)
    #test set needs NORM too to calculate dev_scores
    train_proportion: float = 0.8,
    # The diagnoses you want to include in the split, defaults to all
    diagnoses: List[str] = None,
    # The datasets you want to include in the split, defaults to all
    #datasets: List[str] = None,
    # The random seed for reproducibility
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train = pd.DataFrame()
    valid = pd.DataFrame()

    # If diagnoses is a string, convert to list
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]
    # If no diagnoses are specified, use all diagnoses in the annotations
    elif diagnoses is None:
        diagnoses = annotations["Diagnosis"].unique().tolist()
    
    datasets = annotations["Dataset"].unique()

    # For each diagnosis, dataset and sex, take a random sample of the data and split it into train and test
    for diagnosis in diagnoses:
        for dataset in datasets:
            for sex in ["Female", "Male"]:
                
                dataset_annotations = annotations[
                    (annotations["Diagnosis"] == diagnosis)
                    & (annotations["Dataset"] == dataset)
                    & (annotations["Sex"] == sex)
                ]
                # shuffle the data randomly 
                dataset_annotations = dataset_annotations.sample(
                    frac=1, random_state=seed
                )
                # split the data
                split = round(len(dataset_annotations) * train_proportion)

                train = pd.concat(
                    [train, dataset_annotations[:split]], ignore_index=True
                )
                valid = pd.concat([valid, dataset_annotations[split:]], ignore_index=True)

    return train, valid

def train_val_split_subjects(
    subjects: List[dict], #list of dictionaries 
    train_ann: pd.DataFrame,
    val_ann: pd.DataFrame
) -> Tuple[List, List]:
    #Split subject data based on previously split annotations

    # Get filenames from annotations (metadata files)
    train_files = set(train_ann["Filename"].str.replace('.nii.gz', '').str.replace('.nii', ''))
    valid_files = set(val_ann["Filename"].str.replace('.nii.gz', '').str.replace('.nii', ''))
    
    train_subjects = []
    valid_subjects = []
    unmatched_subjects = []
    
    # Assign subjects to train or validation set
    for subject in subjects:
        subject_name = subject["name"]
        # Remove file extensions if present
        subject_name_clean = subject_name.replace('.nii.gz', '').replace('.nii', '')
        
        if subject_name_clean in train_files:
            train_subjects.append(subject)
        elif subject_name_clean in valid_files:
            valid_subjects.append(subject)
        else:
            unmatched_subjects.append(subject_name)
    
    print(f"[INFO] {len(train_subjects)} subjects in training set")
    print(f"[INFO] {len(valid_subjects)} subjects in validation set")
    
    if unmatched_subjects:
        print(f"[WARNING] {len(unmatched_subjects)} subjects not found in annotations:")
        for i, name in enumerate(unmatched_subjects[:5]):
            print(f"  - {name}")
        if len(unmatched_subjects) > 5:
            print(f"  ... and {len(unmatched_subjects) - 5} more")
    
    return train_subjects, valid_subjects

class CustomDataset_2D():  
    # Create Datasets that can then be converted into DataLoader objects
    def __init__(self, subjects):
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        measurements = self.subjects[idx]["measurements"]
        labels = self.subjects[idx]["labels"]

        labels_df = pd.DataFrame(labels)
        labels_arr = labels_df.values
        names = self.subjects[idx]["name"]

        measurements = torch.as_tensor(measurements, dtype=torch.float32)  # Ensure this is float32, as the weights of the model are initialized.
        labels = torch.as_tensor(labels_arr, dtype=torch.float32)  # float32 required for linear operations!

        return measurements, labels, names

"""
New load_mri_data_2D for CSV-based data with:
- TIV normalization for volume data
- Original normalize_and_scale_og() logic (Log + Row-wise Z-score)
- Compatible with CAT12_results_final.csv format
"""

import pandas as pd
import numpy as np
import os
import re
from typing import List, Tuple


def normalize_and_scale_og_csv(df: pd.DataFrame, volume_columns: List[str]) -> pd.DataFrame:
    """
    Apply the original normalization from the HDF5 version:
    1. Log transformation: ln((10000*value)/sum_values + 1)
    2. Row-wise Z-standardization per volume type
    
    Args:
        df: DataFrame with Rows=ROIs, Columns=Patients (transposed from CSV)
        volume_columns: List of column names grouped by volume type
    
    Returns:
        Normalized DataFrame
    """
    df_copy = df.copy()
    
    # Step 1: Log transformation
    column_sums = df_copy.sum()
    transformed_df = np.log((10000 * df_copy) / column_sums + 1)
    
    norm_copy = transformed_df.copy()
    
    # Step 2: Extract volume types from column names
    # For CSV: columns are like "Vgm_Neurom_Region", extract "Vgm"
    volume_types = {}
    for col in norm_copy.columns:
        # Extract volume type prefix (Vgm, Vwm, Vcsf, G, T)
        vtype = col.split('_')[0]
        if vtype not in volume_types:
            volume_types[vtype] = []
        volume_types[vtype].append(col)
    
    # Step 3: Z-scale each volume type separately (row-wise, i.e. per ROI)
    for vtype, cols in volume_types.items():
        if not cols:
            continue
            
        # Scale the selected columns per row (per ROI across patients)
        scaled = norm_copy[cols].apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
            axis=1  # Row-wise (across patients for each ROI)
        )
        
        norm_copy.loc[:, cols] = scaled
    
    return norm_copy

"""
Add these TWO normalization functions to module/data_processing_hc.py

Replace the current normalize_volume_types_separately with both versions.
"""

import pandas as pd
import numpy as np
from typing import List


def normalize_volume_types_separately_rowwise(
    mri_data: pd.DataFrame, 
    selected_columns: List[str], 
    use_tiv: bool = True
) -> pd.DataFrame:
    """
    ROW-WISE normalization (Pinaya approach).
    
    Normalizes each volume type separately:
    1. TIV normalization (only for Vgm, Vwm, Vcsf)
    2. Log transformation (per volume type)
    3. Row-wise Z-score (per ROI across patients)
    
    This preserves RELATIVE patterns within each brain.
    """
    print("[INFO] Starting ROW-WISE normalization per volume type...")
    
    normalized_data = pd.DataFrame(index=mri_data.index)
    
    volume_types = {
        'Vgm': [col for col in selected_columns if col.startswith('Vgm_')],
        'Vwm': [col for col in selected_columns if col.startswith('Vwm_')],
        'Vcsf': [col for col in selected_columns if col.startswith('Vcsf_')],
        'G': [col for col in selected_columns if col.startswith('G_')],
        'T': [col for col in selected_columns if col.startswith('T_')]
    }
    
    for vtype, cols in volume_types.items():
        if not cols:
            continue
            
        print(f"[INFO] Normalizing {vtype}: {len(cols)} features (ROW-WISE)")
        
        # 1. TIV NORMALIZATION (only for Vgm, Vwm, Vcsf)
        if vtype in ['Vgm', 'Vwm', 'Vcsf'] and use_tiv and 'TIV' in mri_data.columns:
            print(f"[INFO]   - Applying TIV normalization to {vtype}")
            vtype_data = mri_data[cols].div(mri_data['TIV'], axis=0)
        else:
            vtype_data = mri_data[cols].copy()
        
        # 2. TRANSPOSE (ROIs x Patients)
        vtype_transposed = vtype_data.T
        
        # 3. LOG TRANSFORMATION (per volume type)
        column_sums = vtype_transposed.sum(axis=0)
        log_transformed = np.log((10000 * vtype_transposed) / column_sums + 1)
        print(f"[INFO]   - Applied log transformation")
        
        # 4. ROW-WISE Z-SCORE (per ROI across patients)
        z_scored = log_transformed.apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
            axis=1
        )
        print(f"[INFO]   - Applied ROW-WISE Z-score normalization")
        
        # 5. TRANSPOSE BACK (Patients x ROIs)
        normalized_vtype = z_scored.T
        
        normalized_data = pd.concat([normalized_data, normalized_vtype], axis=1)
    
    print(f"[INFO] ROW-WISE normalization complete. Total features: {normalized_data.shape[1]}")
    
    return normalized_data


def normalize_volume_types_separately_columnwise(
    mri_data: pd.DataFrame, 
    selected_columns: List[str], 
    use_tiv: bool = True
) -> pd.DataFrame:
    """
    COLUMN-WISE normalization (Classical neuroimaging approach).
    
    Normalizes each volume type separately:
    1. TIV normalization (only for Vgm, Vwm, Vcsf)
    2. Column-wise Z-score (per feature across patients)
    
    This preserves ABSOLUTE differences between patients.
    No log transformation needed as each feature has its own scale.
    """
    print("[INFO] Starting COLUMN-WISE normalization per volume type...")
    
    normalized_data = pd.DataFrame(index=mri_data.index)
    
    volume_types = {
        'Vgm': [col for col in selected_columns if col.startswith('Vgm_')],
        'Vwm': [col for col in selected_columns if col.startswith('Vwm_')],
        'Vcsf': [col for col in selected_columns if col.startswith('Vcsf_')],
        'G': [col for col in selected_columns if col.startswith('G_')],
        'T': [col for col in selected_columns if col.startswith('T_')]
    }
    
    for vtype, cols in volume_types.items():
        if not cols:
            continue
            
        print(f"[INFO] Normalizing {vtype}: {len(cols)} features (COLUMN-WISE)")
        
        # 1. TIV NORMALIZATION (only for Vgm, Vwm, Vcsf)
        if vtype in ['Vgm', 'Vwm', 'Vcsf'] and use_tiv and 'TIV' in mri_data.columns:
            print(f"[INFO]   - Applying TIV normalization to {vtype}")
            vtype_data = mri_data[cols].div(mri_data['TIV'], axis=0)
        else:
            vtype_data = mri_data[cols].copy()
        
        # 2. COLUMN-WISE Z-SCORE (per feature across patients)
        # Calculate mean and std for each feature across all patients
        means = vtype_data.mean(axis=0)  # Mean per column (feature)
        stds = vtype_data.std(axis=0)    # Std per column (feature)
        
        # Handle features with zero std (constant values)
        stds = stds.replace(0, 1)
        
        normalized_vtype = (vtype_data - means) / stds
        print(f"[INFO]   - Applied COLUMN-WISE Z-score normalization")
        
        normalized_data = pd.concat([normalized_data, normalized_vtype], axis=1)
    
    print(f"[INFO] COLUMN-WISE normalization complete. Total features: {normalized_data.shape[1]}")
    
    return normalized_data


def normalize_volume_types_separately(
    mri_data: pd.DataFrame, 
    selected_columns: List[str], 
    use_tiv: bool = True,
    method: str = "rowwise"
) -> pd.DataFrame:
    """
    Wrapper function that calls either row-wise or column-wise normalization.
    
    Args:
        mri_data: DataFrame with all MRI measurements
        selected_columns: List of column names to normalize
        use_tiv: Whether to apply TIV normalization to volume data
        method: "rowwise" (Pinaya approach) or "columnwise" (classical approach)
        
    Returns:
        DataFrame with normalized data
    """
    if method == "rowwise":
        return normalize_volume_types_separately_rowwise(mri_data, selected_columns, use_tiv)
    elif method == "columnwise":
        return normalize_volume_types_separately_columnwise(mri_data, selected_columns, use_tiv)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'rowwise' or 'columnwise'")

"""
Simplified load_mri_data_2D for PRE-NORMALIZED data.

This version:
1. Loads the pre-normalized CSV (already has HC-only stats applied)
2. Filters to requested subjects (from metadata CSV)
3. Returns subjects in the expected format

NO normalization is done here - it was already done in preprocessing!
"""

import pandas as pd
import numpy as np
import os
import re
from typing import List, Tuple


def load_mri_data_2D_prenormalized(
    normalized_csv_path: str,  # NEW! Path to pre-normalized CSV
    csv_paths: List[str] = None,
    diagnoses: List[str] = None,
    covars: List[str] = [],
) -> Tuple:
    """
    Load pre-normalized MRI data and filter to requested subjects.
    
    Args:
        normalized_csv_path: Path to the pre-normalized CSV
        csv_paths: List of metadata CSV paths (determines train/test split)
        diagnoses: List of diagnoses to include (None = all)
        covars: List of covariates for one-hot encoding
        
    Returns:
        subjects: List of subject dictionaries
        data_overview: DataFrame with metadata
        roi_names: List of ROI column names
    """
    
    print("="*80)
    print("[INFO] Loading PRE-NORMALIZED MRI data")
    print("="*80)
    
    # ==================== STEP 1: Load Metadata ====================
    print(f"[INFO] Loading metadata from: {csv_paths}")
    
    csv_paths = [path.strip("[]'\"") for path in csv_paths]
    
    data_overview_list = []
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file '{csv_path}' not found")
        df = pd.read_csv(csv_path)
        data_overview_list.append(df)
    
    data_overview = pd.concat(data_overview_list, ignore_index=True)
    print(f"[INFO] Loaded metadata with {len(data_overview)} subjects")
    
    # ==================== STEP 2: Filter by Diagnosis ====================
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    data_overview = data_overview.reset_index(drop=True)
    
    print(f"[INFO] Filtered to {len(data_overview)} subjects with diagnoses: {diagnoses}")
    
    # ==================== STEP 3: Load Pre-Normalized MRI Data ====================
    print(f"[INFO] Loading pre-normalized MRI data from: {normalized_csv_path}")
    
    if not os.path.isfile(normalized_csv_path):
        raise FileNotFoundError(f"[ERROR] Normalized CSV not found: {normalized_csv_path}")
    
    mri_data = pd.read_csv(normalized_csv_path)
    print(f"[INFO] Loaded MRI data with shape: {mri_data.shape}")
    
    # ==================== STEP 4: Identify ROI Columns ====================
    # ROI columns are everything except metadata columns
    metadata_columns = [
        'Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV',
        'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol',
        'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
        'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global'
    ]
    roi_columns = [col for col in mri_data.columns if col not in metadata_columns]
    
    print(f"[INFO] Found {len(roi_columns)} ROI columns")
    
    # Count by volume type
    vgm_cols = [col for col in roi_columns if col.startswith('Vgm_')]
    g_cols = [col for col in roi_columns if col.startswith('G_')]
    t_cols = [col for col in roi_columns if col.startswith('T_')]
    
    print(f"[INFO]   - Vgm: {len(vgm_cols)} features")
    print(f"[INFO]   - G: {len(g_cols)} features")
    print(f"[INFO]   - T: {len(t_cols)} features")
    print(f"[INFO]   - Total: {len(roi_columns)} features")
    
    # ==================== STEP 5: Prepare One-Hot Labels ====================
    covars = [covars] if not isinstance(covars, list) else covars
    
    one_hot_labels = {}
    variables = ["Diagnosis"] + covars
    
    for var in variables:
        if var not in data_overview.columns:
            raise ValueError(f"[ERROR] Column '{var}' not found in metadata")
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)
    
    # ==================== STEP 6: Match Subjects ====================
    print("[INFO] Matching metadata subjects with MRI data...")
    
    subjects_dict = {}
    matched_count = 0
    unmatched = []
    
    for index, row in data_overview.iterrows():
        filename = row["Filename"]
        
        # Try to find in MRI data
        # Handle potential differences in filename format
        filename_clean = re.sub(r"\.[^.]+$", "", filename)
        
        mri_row = mri_data[mri_data['Filename'] == filename]
        if mri_row.empty:
            mri_row = mri_data[mri_data['Filename'] == filename_clean]
        
        if mri_row.empty:
            unmatched.append(filename)
            continue
        
        # Extract ROI measurements (already normalized!)
        measurements = mri_row[roi_columns].values.flatten().tolist()
        
        # Create subject entry
        subjects_dict[filename] = {
            "name": filename,
            "measurements": measurements,
            "labels": {var: one_hot_labels[var].iloc[index].to_numpy().tolist() 
                      for var in variables}
        }
        matched_count += 1
    
    print(f"[INFO] Successfully matched {matched_count}/{len(data_overview)} subjects")
    
    if unmatched:
        print(f"[WARNING] {len(unmatched)} subjects not found in MRI data")
        if len(unmatched) <= 10:
            print(f"[WARNING] Unmatched: {unmatched}")
    
    # ==================== STEP 7: Convert to List ====================
    subjects = list(subjects_dict.values())
    
    print(f"[INFO] Total subjects processed: {len(subjects)}")
    print(f"[INFO] Total ROI features per subject: {len(roi_columns)}")
    print("[INFO] Data loading complete!")
    print("="*80)
    
    return subjects, data_overview, roi_columns


# ==================== HELPER FUNCTION ====================
def verify_normalization(subjects: List, subject_count: int = 5):
    """
    Quick check to verify the data looks normalized.
    """
    print("\n[INFO] Verifying normalization (checking first few subjects)...")
    
    for i in range(min(subject_count, len(subjects))):
        measurements = np.array(subjects[i]['measurements'])
        print(f"  Subject {i+1}: mean={measurements.mean():.3f}, std={measurements.std():.3f}, "
              f"range=[{measurements.min():.3f}, {measurements.max():.3f}]")
    
    print("[INFO] Note: Individual subjects may not have mean=0, std=1")
    print("[INFO] Only the Training-HC group as a whole should have mean≈0, std≈1")

def load_mri_data_2D(
    data_path: str,
    atlas_name: List[str] = None,
    csv_paths: list[str] = None,
    annotations: pd.DataFrame = None,
    diagnoses: List[str] = None,
    covars: List[str] = [],
    hdf5: bool = True,
    train_or_test: str = "train",
    save: bool = None,
    volume_type = None,
    valid_volume_types: List[str] = ["Vgm", "Vwm", "Vcsf", "G", "T"],
    use_tiv_normalization: bool = True,
    normalization_method: str = "rowwise",  # ← NEW PARAMETER!
) -> Tuple:
    """
    Load MRI data from CSV with proper separate normalization per volume type.
    
    Key difference from old version:
    - Each volume type (Vgm, G, T) is normalized separately from the start
    - This prevents mixing of different measurement scales
    
    Args:
        data_path: Path to CAT12_results_final.csv
        atlas_name: List of atlas names or ["all"]
        csv_paths: Paths to metadata CSV files
        volume_type: Volume types to use
        valid_volume_types: All valid volume type prefixes
        use_tiv_normalization: Whether to apply TIV normalization to volume data
        
    Returns:
        subjects: List of subject dictionaries
        data_overview: DataFrame with metadata
        all_roi_names: List of ROI feature names
    """
    
    print(f"[INFO] Loading MRI data from: {data_path}")
    
    # ============================================================
    # 1. HANDLE ATLAS NAMES
    # ============================================================
    atlas_mapping = {
        "neuromorphometrics": "Neurom",
        "lpba40": "lpba40",
        "cobra": "cobra",
        "suit": "SUIT",
        "ibsr": "IBSR",
        "aal3": "AAL3",
        "schaefer100": "Sch100",
        "schaefer200": "Sch200",
        "aparc_dk40": "DK40",
        "aparc_destrieux": "Destrieux"
    }
    
    all_available_atlases = list(atlas_mapping.keys())
    
    if not isinstance(atlas_name, list):
        atlas_name = [atlas_name]
    
    if len(atlas_name) == 1 and atlas_name[0] == "all":
        atlas_name = all_available_atlases
    
    print(f"[INFO] Processing atlases: {atlas_name}")
    
    # ============================================================
    # 2. HANDLE VOLUME TYPES
    # ============================================================
    if volume_type == "all":
        target_volume_types = valid_volume_types
    elif isinstance(volume_type, str):
        target_volume_types = [volume_type]
    elif isinstance(volume_type, list):
        target_volume_types = volume_type
    else:
        target_volume_types = valid_volume_types
    
    print(f"[INFO] Target volume types: {target_volume_types}")
    
    # ============================================================
    # 3. LOAD METADATA
    # ============================================================
    csv_paths = [path.strip("[]'\"") for path in csv_paths]
    
    if csv_paths is not None:
        for csv_path in csv_paths:
            assert os.path.isfile(csv_path), f"[ERROR] CSV file '{csv_path}' not found"
        assert annotations is None, "[ERROR] Both CSV and annotations provided"
        
        dfs = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        data_overview = pd.concat(dfs, ignore_index=True)
        
    elif annotations is not None:
        assert isinstance(annotations, pd.DataFrame), "[ERROR] Annotations must be DataFrame"
        assert csv_paths is None, "[ERROR] Both CSV and annotations provided"
        data_overview = annotations
    else:
        raise ValueError("[ERROR] No CSV path or annotations provided!")
    
    # ============================================================
    # 4. FILTER BY DIAGNOSES
    # ============================================================
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    data_overview = data_overview.reset_index(drop=True)
    
    covars = [covars] if not isinstance(covars, list) else covars
    diagnoses = [diagnoses] if not isinstance(diagnoses, list) else diagnoses
    
    # ============================================================
    # 5. PREPARE ONE-HOT ENCODED LABELS
    # ============================================================
    one_hot_labels = {}
    variables = ["Diagnosis"] + covars
    for var in variables:
        if var not in data_overview.columns:
            raise ValueError(f"[ERROR] Column '{var}' not found in metadata")
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)
    
    # ============================================================
    # 6. LOAD MRI DATA CSV
    # ============================================================
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] MRI data file not found: {data_path}")
    
    mri_data = pd.read_csv(data_path)
    print(f"[INFO] Loaded MRI data with shape: {mri_data.shape}")
    
    # ============================================================
    # 7. SELECT RELEVANT COLUMNS
    # ============================================================
    qc_columns = ['Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV', 
                  'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol', 
                  'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
                  'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global']
    
    all_columns = mri_data.columns.tolist()
    roi_columns = [col for col in all_columns if col not in qc_columns]
    
    print(f"[INFO] Found {len(roi_columns)} ROI columns in MRI data")
    
    # Select columns by atlas and volume type
    selected_columns = []
    all_roi_names = []
    
    for atlas in atlas_name:
        atlas_prefix = atlas_mapping.get(atlas)
        if atlas_prefix is None:
            print(f"[WARNING] Unknown atlas: {atlas}, skipping")
            continue
        
        print(f"[INFO] Processing atlas: {atlas} (prefix: {atlas_prefix})")
        
        atlas_columns = []
        
        for col in roi_columns:
            # Check volume types
            for vt in target_volume_types:
                if col.startswith(f"{vt}_") and f"_{atlas_prefix}_" in col:
                    atlas_columns.append(col)
                    break
            
            # Check thickness and gyrification
            if col.startswith("G_") or col.startswith("T_"):
                if f"_{atlas_prefix}_" in col:
                    atlas_columns.append(col)
        
        if not atlas_columns:
            print(f"[WARNING] No columns found for atlas {atlas}")
            continue
        
        print(f"[INFO] Found {len(atlas_columns)} columns for atlas {atlas}")
        selected_columns.extend(atlas_columns)
        all_roi_names.extend(atlas_columns)
    
    # Remove duplicates
    selected_columns = list(dict.fromkeys(selected_columns))
    all_roi_names = list(dict.fromkeys(all_roi_names))
    
    print(f"[INFO] Selected {len(selected_columns)} feature columns total")
    print(f"[INFO] Total ROI names: {len(all_roi_names)}")
    
    # ============================================================
    # 8. NORMALIZE EACH VOLUME TYPE SEPARATELY
    # ============================================================
    mri_data_normalized = normalize_volume_types_separately(
        mri_data=mri_data,
        selected_columns=selected_columns,
        use_tiv=use_tiv_normalization,
        method=normalization_method  # ← NEW PARAMETER!
    )
    
    # Add Filename column back
    mri_data_normalized.insert(0, 'Filename', mri_data['Filename'])
    
    # ============================================================
    # 9. MATCH METADATA WITH MRI DATA AND CREATE SUBJECTS
    # ============================================================
    print(f"[INFO] Matching {len(data_overview)} metadata entries with MRI data...")
    
    # Create filename lookup (handle sub- prefix)
    mri_filenames = set(mri_data_normalized['Filename'])
    
    subjects_dict = {}
    matched_count = 0
    unmatched_files = []
    
    for index, row in data_overview.iterrows():
        # Clean filename
        file_name = re.sub(r"\.[^.]+$", "", row["Filename"])
        
        # Try to find match (with and without 'sub-' prefix)
        matched_filename = None
        
        if file_name in mri_filenames:
            matched_filename = file_name
        elif f"sub-{file_name}" in mri_filenames:
            matched_filename = f"sub-{file_name}"
        elif file_name.startswith('sub-') and file_name[4:] in mri_filenames:
            matched_filename = file_name[4:]
        
        if matched_filename is None:
            unmatched_files.append(file_name)
            continue
        
        # Extract measurements for this subject
        measurements = mri_data_normalized[mri_data_normalized['Filename'] == matched_filename][selected_columns].values.flatten().astype(np.float32).tolist()
        
        # Create subject dictionary
        subjects_dict[file_name] = {
            "name": file_name,
            "measurements": measurements,
            "labels": {var: one_hot_labels[var].iloc[index].to_numpy().tolist() 
                      for var in variables}
        }
        
        matched_count += 1
    
    print(f"[INFO] Successfully matched {matched_count}/{len(data_overview)} subjects")
    
    if unmatched_files:
        print(f"[WARNING] {len(unmatched_files)} subjects not found in MRI data")
        if len(unmatched_files) <= 10:
            print(f"[WARNING] Unmatched: {unmatched_files}")
        else:
            print(f"[WARNING] First 10 unmatched: {unmatched_files[:10]}")
    
    # ============================================================
    # 10. FINAL OUTPUT
    # ============================================================
    if not subjects_dict:
        raise ValueError("[ERROR] No subjects matched!")
    
    subjects = list(subjects_dict.values())
    
    print(f"[INFO] Total subjects processed: {len(subjects)}")
    print(f"[INFO] Total ROI features per subject: {len(all_roi_names)}")
    print("[INFO] Data loading complete!")
    
    return subjects, data_overview, all_roi_names


def process_subjects(subjects: List[tio.Subject], batch_size: int, shuffle_data: bool) -> DataLoader:
    #produces DataLoader object which is needed as input for the model
    
    dataset = CustomDataset_2D(subjects=subjects)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=4,
        pin_memory=True)

    return data_loader



def process_latent_space_2D(
    adata: ad.AnnData, # AnnData object that contains the latent space representation of the MRI data
    annotations: pd.DataFrame, #metadata df
    umap_neighbors: int, # The number of neighbors for the UMAP calculation
    seed: int,
    atlas_name,
    save_data: bool = False,
    save_path: str = None,
    timestamp: str = None,
    # The current epoch, used for the filename.
    epoch: int = None,
    data_type: str = None,
) -> ad.AnnData:
    #process latent space into visualization

    # align annotation metadata with anndata.obs
    aligned_ann = annotations.set_index("Filename").reindex(adata.obs_names)

    # add annotation metadata to anndata.obs
    for col in aligned_ann.columns:
        adata.obs[col] = aligned_ann[col]

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, umap_neighbors, use_rep="X")
    sc.tl.umap(adata, random_state=seed)

    # save the data if specified
    if save_data:
        adata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}.h5ad")
        )
    return adata


def load_checkpoint_model(model, model_filename: str):

    if os.path.isfile(model_filename):
        log_and_print(f"Loading checkpoint from '{model_filename}'")

        model_state = torch.load(model_filename)
        model.load_state_dict(model_state)

        log_and_print(f"Checkpoint loaded successfully")

    else:
        log_and_print(
            f"No checkpoint found at '{model_filename}', starting training from scratch"
        )

        raise FileNotFoundError(f"No checkpoint found at '{model_filename}'")

    return model

def save_model(model, save_path: str, timestamp: str, descriptor: str, epoch: int):
    model_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_e{epoch}_model.pth",
    )

    torch.save(model.state_dict(), model_save_path)

    log_checkpoint(model_path=model_save_path)


def save_model_metrics(model_metrics, save_path: str, timestamp: str, descriptor: str):
    metrics_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_model_performance.csv",
    )

    pd.DataFrame(model_metrics).to_csv(metrics_save_path, index=False)

    log_checkpoint(
        metrics_path=metrics_save_path,
    )

#=======================================================================================================================================================================

def normalize_and_scale_df(df: pd.DataFrame, df_meta: pd.DataFrame, scaling_method: str = "z", scale_per_dataset: bool = False, vol_to_use: str = None) -> pd.DataFrame:
    # alternative function with z option and iqr option for normalizing rows
    # Normalizes the columns (patient volumes) by Log Scaling and scales the rows (ROIs) with Z-transformation.

    df_copy = df.copy()
    column_sums = df_copy.sum()
    
    # Apply the formula: ln((10000*value)/sum_values + 1) "Log transformation"
    transformed_df = np.log((10000 * df_copy) / column_sums + 1)

    norm_copy = transformed_df.copy()
  
    cols = norm_copy.columns.get_level_values(-1).tolist() # Select lowest level of Multiindex (Measurements: Vgm, Vwm, Vcsf)
    unique_cols = list(set(cols))

    filename_to_dataset_map = df_meta.set_index("Filename")["Dataset"]
    filenames_in_data = norm_copy.columns.get_level_values(0).unique()
    datasets = df_meta["Dataset"].unique().tolist()

    for col_type in unique_cols:  # scale each of the measurement types separately
        vols_to_scale = [col for col in norm_copy.columns if col[-1] == col_type]
            
        # Scale the selected columns per row
        z_scaled = norm_copy[vols_to_scale]
        iqr_scaled = df_copy[vols_to_scale]
        
        if scaling_method == "z":  # For z-scaling
            z_scaled = z_scaled.apply(
                lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
                axis=1
            )

        if scaling_method == "iqr":  # For inter-quantile-range scaling
            iqr_scaled = iqr_scaled.apply(
                lambda row: (row - row.median()) / (row.quantile(0.75)-row.quantile(0.25)) if (row.quantile(0.75)-row.quantile(0.25)) > 0 else pd.Series(0.0, index=row.index),
                axis=1
            )

        # Output for z-scaled data
        if scaling_method == "z":
            norm_copy.loc[:, vols_to_scale] = z_scaled
        
        # Output for IQR scaled data
        if scaling_method == "iqr":
            df_copy.loc[:, vols_to_scale] = iqr_scaled
    
    if scaling_method == "z":
        if vol_to_use is not None:  # Subset only a certain type of measurement (Vgm, Vwm, Vcsf)
            subset = [col for col in norm_copy.columns if col[-1] == vol_to_use]
            return norm_copy[subset]
        else: 
            return norm_copy
    
    if scaling_method == "iqr":
        if vol_to_use is not None and scaling_method == "iqr":  # Subset only a certain type of measurement (Vgm, Vwm, Vcsf)
            subset = [col for col in norm_copy.columns if col[-1] == vol_to_use]
            return df_copy[subset]
        else: 
            return df_copy
        
#Normalizes brain region volumes using a similar approach to the Pinaya paper
def normalize_and_scale_Pinaya(df: pd.DataFrame, ticv_column=None) -> pd.DataFrame:
    # 1. Calculate relative volumes by dividing by total intracranial volume (if provided) -> we currently don't have that
    # 2. Perform robust normalization: (x - median) / IQR for each brain region
    
    df_copy = df.copy()
    
    # Use a dictionary to store normalized columns
    normalized_data = {}

    for column in df_copy.columns:
        median_value = df_copy[column].median()
        q1 = df_copy[column].quantile(0.25)
        q3 = df_copy[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            normalized_data[column] = pd.Series(0, index=df_copy.index)
        else:
            normalized_data[column] = (df_copy[column] - median_value) / iqr

    # Create the final normalized DataFrame at once
    normalized_df = pd.DataFrame(normalized_data, index=df_copy.index)

    return normalized_df