import pathlib
import pandas as pd
import regex as re
import os
import numpy as np
from datetime import datetime
import torch


def flatten_array(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array. 
    array = df.to_numpy()
    flat_array = array.flatten()
    return flat_array

def get_all_data(directory: str, ext: str = "h5") -> list:
    data_paths = list(pathlib.Path(directory).rglob(f"*.{ext}"))
    return data_paths


def get_atlas(path: pathlib.PosixPath) -> str:
    stem = path.stem
    match = re.search(r"_(.*)", stem)
    if match:
        atlas = match.group(1)
    return atlas

def extract_measurements(subjects):
    #Extract measurements from subjects as torch tensors
    all_measurements = []
    for subject in subjects:
        all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
    return torch.stack(all_measurements)

def combine_dfs(paths: list):
    # Combines any number of csv files to a single pandas DataFrame, keeping only shared column indices. 
    
    if len(paths) > 1: 
        for i in range(1,len(paths)):
            if i == 1: 
                joined_df = pd.read_csv(paths[i-1], header=[0])
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")  # Parameter "inner" keeps only the shared column indices.
            else:
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")
    else: 
        joined_df = pd.read_csv(paths[0], header=[0])
    return joined_df


def read_hdf5_to_df(filepath: str):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return None
    try:
        df = pd.read_hdf(filepath, key='atlas_data')
        return df 
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def read_hdf5_to_df_t(filepath: str):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return None
    try:
        return pd.read_hdf(filepath, key='atlas_data_t')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def split_df(path_original: str, path_to_dir: str):
    # old function -> just splits depending on HC or not
    df = pd.read_csv(path_original, header=[0])
    df = df.drop(columns=["Unnamed: 0"])  
    path_hc = f"{path_to_dir}/hc_metadata.csv"
    subset_hc = df[df["Diagnosis"] == "HC"]
    subset_hc.to_csv(path_hc, index=False)  
    path_non_hc = f"{path_to_dir}/non_hc_metadata.csv"
    subset_non_hc = df[df["Diagnosis"] != "HC"]
    subset_non_hc.to_csv(path_non_hc, index=False)  
    
    return path_hc, path_non_hc


def split_df_adapt(path_original: str, path_to_dir: str, norm_diagnosis: str = "HC", train_ratio: float = 0.7, random_seed: int = 42):
    
    # function to split the dataset depending on adaptable variables norm_diagnosis and train_ratio
    
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    
    df = pd.read_csv(path_original)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    np.random.seed(random_seed)
    
    norm_data = df[df["Diagnosis"] == norm_diagnosis].copy() #-> training (NORM)
    other_data = df[df["Diagnosis"] != norm_diagnosis].copy() #-> testing (not NORM)
    
    num_train = int(len(norm_data) * train_ratio)
    
    all_indices = np.array(norm_data.index)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:num_train]
    test_indices = all_indices[num_train:]
    
    train_norm = norm_data.loc[train_indices]
    test_norm = norm_data.loc[test_indices]
    
    test_data = pd.concat([test_norm, other_data])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path_train = f"{path_to_dir}/train_metadata{norm_diagnosis}_{train_ratio}{timestamp}.csv"
    path_test = f"{path_to_dir}/test_metadata{norm_diagnosis}_{train_ratio}{timestamp}.csv"
    
    train_norm.to_csv(path_train, index=False)
    test_data.to_csv(path_test, index=False)
    
    print(f"Training set erstellt: {path_train} ({len(train_norm)} Patienten)")
    print(f"Test set erstellt: {path_test} ({len(test_data)} Patienten, davon {len(test_norm)} {norm_diagnosis} und {len(other_data)} andere)")
    
    return path_train, path_test

    