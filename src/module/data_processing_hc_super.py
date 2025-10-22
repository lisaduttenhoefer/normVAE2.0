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

def load_mri_data_2D(
    data_path: str,  
    atlas_name: List[str] = None, #adaptable
    csv_paths: list[str] = None,
    annotations: pd.DataFrame = None,
    diagnoses: List[str] = None, #adaptable
    covars: List[str] = [],
    hdf5: bool = True,
    train_or_test: str = "train",
    save: bool = None,
    volume_type = None, #adaptable
    valid_volume_types: List[str] = ["Vgm", "Vwm", "Vcsf"],
) -> Tuple:
    
    # Define all available atlases
    all_available_atlases = ["cobra", "lpba40", "neuromorphometrics", "suit", "thalamic_nuclei", "thalamus"]
    
    # Ensure atlas_name is a list
    if not isinstance(atlas_name, list):
        atlas_name = [atlas_name]
    
    # Check for the special case of ["all"]
    if len(atlas_name) == 1 and atlas_name[0] == "all":
        atlas_name = all_available_atlases

    print(f"[INFO] Processing atlases: {atlas_name}")
   
   #handle volume_type 
    if volume_type == "all":
        target_volume_types = valid_volume_types
    elif isinstance(volume_type, str):
        target_volume_types = [volume_type]
    elif isinstance(volume_type, list):
        target_volume_types = volume_type
    else:
        target_volume_types = valid_volume_types  # default fallback

    print(f"[INFO] Processing atlases: {atlas_name}")
    print(f"[INFO] Target volume types: {target_volume_types}")

    csv_paths = [path.strip("[]'\"") for path in csv_paths]
    
    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"[ERROR] CSV file '{csv_path}' not found"
            assert annotations is None, "[ERROR] Both CSV and annotations provided"
        data_overview = combine_dfs(csv_paths)
        
    # Handle annotations
    elif annotations is not None:
        assert isinstance(annotations, pd.DataFrame), "[ERROR] Annotations must be a pandas DataFrame"
        assert csv_paths is None, "[ERROR] Both CSV and annotations provided"
        data_overview = annotations
        print("[INFO] Annotations loaded successfully.")
        
    else:
        raise ValueError("[ERROR] No CSV path or annotations provided!")
    
    # Handling diagnoses filtering 
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    
    # Filter data_overview by diagnoses
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    
    # Reset index after filtering to ensure continuous indices
    data_overview = data_overview.reset_index(drop=True)
    
    # Handling covariates and diagnoses lists
    covars = [covars] if not isinstance(covars, list) else covars
    diagnoses = [diagnoses] if not isinstance(diagnoses, list) else diagnoses
    
    # Prepare one-hot encoded labels 
    one_hot_labels = {}
    variables = ["Diagnosis"] + covars
    
    for var in variables:
        if var not in data_overview.columns:
            raise ValueError(f"[ERROR] Column '{var}' not found in CSV file or annotations")
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)
    
    # initializing storing as combined information across atlases
    subjects_dict = {}
    all_roi_names = []
    
    # Process each atlas in the list
    for atlas in atlas_name:
        print(f"[INFO] Processing atlas: {atlas}")
        atlas_data_path = f"{data_path}/Aggregated_{atlas}.h5"
        
        if not os.path.exists(atlas_data_path):
            print(f"[ERROR] Atlas file not found: {atlas_data_path} - skipping this atlas")
            continue
        try:
            if hdf5:
                data = read_hdf5_to_df(filepath=atlas_data_path)
            else:
                data = pd.read_csv(atlas_data_path, header=[0, 1], index_col=0)
         
        except Exception as e:
            print(f"[ERROR] Loading failed {atlas}; reason: {str(e)} -> skipping")
            continue
        
        # Extract unique patient IDs before flattening
        if isinstance(data.columns, pd.MultiIndex):
            all_file_names = data.columns.get_level_values(0).unique()
            available_volume_types = data.columns.get_level_values(1).unique().tolist()
        else:
            all_file_names = data.columns
            available_volume_types = valid_volume_types  

        # Extract ROI names for this atlas
        base_roi_names = data.index.tolist()
        
        atlas_volume_types = [vt for vt in target_volume_types if vt in available_volume_types]
        
        if not atlas_volume_types:
            print(f"[WARNING] No matching volume types found for atlas {atlas}. Available: {available_volume_types}, Requested: {target_volume_types}")
            continue
        
        print(f"[INFO] Using volume types {atlas_volume_types} for atlas {atlas}")
        
        atlas_roi_names = [f"{atlas}_{roi}_{vt}" for roi in base_roi_names for vt in atlas_volume_types]
        
        # Add this atlas's ROI names to the overall list
        all_roi_names.extend(atlas_roi_names)
        print(f"[INFO] Added {len(atlas_roi_names)} ROI names from atlas {atlas}")
        
        # Normalize and scale data 
        #choose function depending on the method you want to use
        data = normalize_and_scale_og(data)
        
        #filtering the data depending on volume_type
        if isinstance(data.columns, pd.MultiIndex):
            if atlas_volume_types and atlas_volume_types != available_volume_types:
                # Get only columns matching the specified volume_types
                filtered_columns = [(patient, vol) for patient, vol in data.columns if vol in atlas_volume_types]
                if filtered_columns:
                    data = data[filtered_columns]
                    print(f"[INFO] Filtered to {len(filtered_columns)} columns for volume types {atlas_volume_types}")
                else:
                    print(f"[ERROR] No columns found for volume_types {atlas_volume_types}")
                    print(f"[DEBUG] Available volume types in data: {available_volume_types}")
                    continue  # Skip this atlas if no matching volume types found -> important for atlases like thalamus that only contain certain vts
            
            #flatten multiindex (Patient_ID, volume_type)xROI -> Patient_ID x ROI_volume_type
            flattened_columns = [f"{patient}_{volume}" for patient, volume in data.columns]
            data.columns = flattened_columns
        
        # Save processed data if needed
        if save:
            volume_suffix = "_".join(atlas_volume_types)
            save_path = f"/workspace/project/catatonia_VAE-main_bq/data/proc_extracted_xml_data/Proc_{atlas}_{volume_suffix}_uiui.csv"
            data.to_csv(save_path)
        
        # Process each subject for this atlas
        for index, row in data_overview.iterrows():
            file_name = re.sub(r"\.[^.]+$", "", row["Filename"])
            
            # Check if file_name is in all_file_names
            if isinstance(all_file_names, pd.Index):
                file_found = any(file_name in fn for fn in all_file_names)
            else:
                file_found = file_name in all_file_names
                
            if not file_found:
                print(f"[ERROR] Filename {file_name} not found in MRI data for atlas {atlas}.")
                continue
            
            # Select patient data 
            patient_columns = []
            for vt in atlas_volume_types:
                cols = [col for col in data.columns if col.startswith(f"{file_name}_{vt}")]
                patient_columns.extend(cols)
            
            if not patient_columns:
                print(f"[ERROR] No columns for patient {file_name} with volume types {atlas_volume_types} in atlas {atlas}.")
                continue
                
            # Extract the actual numerical values from the DataFrame
            patient_data = data[patient_columns]

            # Convert DataFrame to numpy array, then flatten
            if isinstance(patient_data, pd.DataFrame):
                # Get the values as numpy array and flatten
                patient_values = patient_data.values.flatten()
            else:
                # If it's already a numpy array or similar
                patient_values = patient_data.flatten()

            # Convert to list for consistent handling
            flat_patient_data = patient_values.tolist()

            # Create/update subject entry in the dictionary
            if file_name not in subjects_dict:
                subjects_dict[file_name] = {
                    "name": file_name,
                    "measurements": flat_patient_data,  # Now contains actual numbers, not DataFrame
                    "labels": {var: one_hot_labels[var].iloc[index].to_numpy().tolist() for var in variables}
                }
            else:
                subjects_dict[file_name]["measurements"] += flat_patient_data
        
    # Make sure we found at least one valid atlas
    if not subjects_dict:
        raise ValueError("[ERROR] No valid data was processed from any atlas!")
    
    # Convert dictionary to list of subjects
    subjects = list(subjects_dict.values())
    print(f"[INFO] Total subjects processed across all atlases: {len(subjects)}")
    print(f"[INFO] Total ROI features: {len(all_roi_names)}")
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