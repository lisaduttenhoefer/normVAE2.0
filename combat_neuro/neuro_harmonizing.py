from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Adjust paths to your files
HC_METADATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_metadata/metadata_for_harmonizing_hc.csv"
PAT_METADATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_metadata/metadata_for_harmonizing_patients.csv"
MRI_DATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results"

# Datasets to exclude (no HCs available)
EXCLUDE_DATASETS = ['NSS', 'EPSY']

# Load the data
print("Loading data...")
hc_covars = pd.read_csv(HC_METADATA)
pat_covars = pd.read_csv(PAT_METADATA)
all_mri_data = pd.read_csv(MRI_DATA)

print(f"HC Metadata (initial): {hc_covars.shape}")
print(f"Patient Metadata (initial): {pat_covars.shape}")
print(f"MRI Data: {all_mri_data.shape}")

# ============================================
# EXCLUDE DATASETS WITHOUT HCs
# ============================================

print("\n" + "="*60)
print("EXCLUDING DATASETS WITHOUT HEALTHY CONTROLS")
print("="*60)

# Check which datasets exist
print(f"\nDatasets in HCs: {sorted(hc_covars['Dataset'].unique())}")
print(f"Datasets in Patients: {sorted(pat_covars['Dataset'].unique())}")

# Count patients to be excluded
n_excluded = len(pat_covars[pat_covars['Dataset'].isin(EXCLUDE_DATASETS)])
print(f"\nExcluding {n_excluded} patients from datasets: {EXCLUDE_DATASETS}")
print(f"Dataset distribution of excluded patients:")
print(pat_covars[pat_covars['Dataset'].isin(EXCLUDE_DATASETS)]['Dataset'].value_counts())

# Filter out patients from excluded datasets
pat_covars = pat_covars[~pat_covars['Dataset'].isin(EXCLUDE_DATASETS)].copy()

print(f"\nPatient Metadata (after exclusion): {pat_covars.shape}")

# Define columns that are NOT ROIs and should be ignored
non_roi_columns = ['Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV', 
                   'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol', 
                   'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
                   'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global']

# Separate HC and Patient MRI data based on Filename
hc_filenames = hc_covars['Filename'].values
pat_filenames = pat_covars['Filename'].values

hc_mri = all_mri_data[all_mri_data['Filename'].isin(hc_filenames)].copy()
pat_mri = all_mri_data[all_mri_data['Filename'].isin(pat_filenames)].copy()

print(f"\nHC MRI Data: {hc_mri.shape}")
print(f"Patient MRI Data (after exclusion): {pat_mri.shape}")

# Check if all Filenames were found
missing_hc = len(hc_filenames) - len(hc_mri)
missing_pat = len(pat_filenames) - len(pat_mri)
print(f"\nHC: {len(hc_filenames)} metadata entries, {len(hc_mri)} MRI scans found ({missing_hc} missing)")
print(f"Patients: {len(pat_filenames)} metadata entries, {len(pat_mri)} MRI scans found ({missing_pat} missing)")

if missing_hc > 0 or missing_pat > 0:
    print("⚠️ WARNING: Some subjects in metadata don't have MRI data!")

# Sort both DataFrames by Filename
hc_mri = hc_mri.sort_values('Filename').reset_index(drop=True)
hc_covars = hc_covars.sort_values('Filename').reset_index(drop=True)

pat_mri = pat_mri.sort_values('Filename').reset_index(drop=True)
pat_covars = pat_covars.sort_values('Filename').reset_index(drop=True)

# Check if the order matches
hc_order_match = all(hc_mri['Filename'].values == hc_covars['Filename'].values)
print(f"\nHC: Order matches: {hc_order_match}")
pat_order_match = all(pat_mri['Filename'].values == pat_covars['Filename'].values)
print(f"Patients: Order matches: {pat_order_match}")

if not hc_order_match or not pat_order_match:
    raise ValueError("Filename order does not match!")

# Extract only ROI columns (all except non_roi_columns)
roi_columns = [col for col in hc_mri.columns if col not in non_roi_columns]
print(f"\nNumber of ROI features: {len(roi_columns)}")
print(f"First 10 ROIs: {roi_columns[:10]}")

hc_roi = hc_mri[roi_columns].copy()
pat_roi = pat_mri[roi_columns].copy()

# Check for features with no variance
hc_roi_var = hc_roi.var()
pat_roi_var = pat_roi.var()

hc_zero_var = hc_roi_var[hc_roi_var == 0].index.tolist()
pat_zero_var = pat_roi_var[pat_roi_var == 0].index.tolist()

print(f"\nHC: {len(hc_zero_var)} features with no variance")
print(f"Patients: {len(pat_zero_var)} features with no variance")

# Remove features with no variance
if hc_zero_var:
    print(f"Removing HC features with no variance (showing first 5): {hc_zero_var[:5]}")
    hc_roi = hc_roi.drop(columns=hc_zero_var)
    
if pat_zero_var:
    print(f"Removing patient features with no variance (showing first 5): {pat_zero_var[:5]}")
    pat_roi = pat_roi.drop(columns=pat_zero_var)

# Ensure both datasets have the same ROI columns
common_rois = list(set(hc_roi.columns) & set(pat_roi.columns))
print(f"\nCommon ROIs after variance filtering: {len(common_rois)}")

hc_roi = hc_roi[common_rois]
pat_roi = pat_roi[common_rois]

# ============================================
# PREPARE COVARIATES FOR HARMONIZATION
# ============================================

print("\n" + "="*60)
print("PREPARING COVARIATES")
print("="*60)

# Covariates for harmonization:
# - SITE: will be harmonized (batch effect REMOVED)
# - Age: biological covariate (effect PRESERVED, smoothed because non-linear)
# - TIV: biological covariate (effect PRESERVED - head size matters!)
# - IQR: technical covariate (effect PRESERVED - quality matters!)
# - Sex_Male: biological covariate (effect PRESERVED)

required_covars = ['Age', 'TIV', 'IQR', 'SITE', 'Sex_Male']

# Check if all required covariates exist
for cov in required_covars:
    if cov not in hc_covars.columns:
        raise ValueError(f"Required covariate '{cov}' not found in HC metadata!")
    if cov not in pat_covars.columns:
        raise ValueError(f"Required covariate '{cov}' not found in Patient metadata!")

hc_covars_harm = hc_covars[required_covars].copy()
pat_covars_harm = pat_covars[required_covars].copy()

print("\nCovariate columns for harmonization:")
print(f"Columns: {hc_covars_harm.columns.tolist()}")
print("\nCovariate interpretation:")
print("  - SITE: Harmonized (scanner effects REMOVED)")
print("  - Age: Preserved (biological effect kept, smoothed)")
print("  - TIV: Preserved (head size effect kept)")
print("  - IQR: Preserved (quality effect kept)")
print("  - Sex_Male: Preserved (sex effect kept)")

# Check site distribution
print("\n" + "="*60)
print("SITE ANALYSIS")
print("="*60)

sites_in_hc = set(hc_covars['SITE'].unique())
sites_in_pat = set(pat_covars['SITE'].unique())

print(f"\nSites in HCs: {sorted(sites_in_hc)}")
print(f"Sites in Patients: {sorted(sites_in_pat)}")

sites_only_in_patients = sites_in_pat - sites_in_hc
if sites_only_in_patients:
    print(f"\n⚠️ ERROR: These sites have patients but NO HCs: {sites_only_in_patients}")
    print("Cannot harmonize these sites! Please check your data.")
    raise ValueError("Some sites have no healthy controls for training!")

# Site distribution in HCs
print("\nSite distribution in HCs:")
site_dist_hc = hc_covars['SITE'].value_counts().sort_index()
print(site_dist_hc)
print(f"\nTotal HCs: {len(hc_covars)}")
print(f"Number of sites: {len(site_dist_hc)}")
print(f"Min HCs per site: {site_dist_hc.min()}")
print(f"Max HCs per site: {site_dist_hc.max()}")
print(f"Mean HCs per site: {site_dist_hc.mean():.1f}")

if site_dist_hc.min() < 5:
    print("\n⚠️ WARNING: Some sites have <5 HCs!")
    print("Sites with <5 HCs:")
    print(site_dist_hc[site_dist_hc < 5])
    print("Consider combining small sites or using more data!")

# Site distribution in Patients
print("\nSite distribution in Patients:")
site_dist_pat = pat_covars['SITE'].value_counts().sort_index()
print(site_dist_pat)

# Check for missing values
print("\n" + "="*60)
print("CHECKING FOR MISSING VALUES")
print("="*60)

print(f"\nHC ROI: {hc_roi.isnull().sum().sum()} NaNs")
print(f"HC Covars: {hc_covars_harm.isnull().sum().sum()} NaNs")
if hc_covars_harm.isnull().sum().sum() > 0:
    print("Missing values per covariate:")
    print(hc_covars_harm.isnull().sum())

print(f"\nPatient ROI: {pat_roi.isnull().sum().sum()} NaNs")
print(f"Patient Covars: {pat_covars_harm.isnull().sum().sum()} NaNs")
if pat_covars_harm.isnull().sum().sum() > 0:
    print("Missing values per covariate:")
    print(pat_covars_harm.isnull().sum())

if hc_roi.isnull().sum().sum() > 0 or pat_roi.isnull().sum().sum() > 0:
    raise ValueError("ROI data contains NaNs! Please clean your data first.")

if hc_covars_harm.isnull().sum().sum() > 0 or pat_covars_harm.isnull().sum().sum() > 0:
    raise ValueError("Covariate data contains NaNs! Please clean your data first.")

# Check data types
print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(f"\nHC ROI dtypes: {hc_roi.dtypes.value_counts().to_dict()}")
print(f"Patient ROI dtypes: {pat_roi.dtypes.value_counts().to_dict()}")

# ============================================
# TRAIN/TEST SPLIT (BEFORE HARMONIZATION!)
# ============================================

print("\n" + "="*60)
print("SPLITTING HC DATA (Train/Test) - NO DATA LEAKAGE")
print("="*60)

# Determine test_size based on total HC count
total_hc = len(hc_roi)
if total_hc < 100:
    test_size = 0.15  # 15% test for small datasets
    print(f"Small dataset detected ({total_hc} HCs). Using test_size=0.15")
elif total_hc < 200:
    test_size = 0.20  # 20% test for medium datasets
    print(f"Medium dataset detected ({total_hc} HCs). Using test_size=0.20")
else:
    test_size = 0.20  # 20% test for large datasets
    print(f"Large dataset detected ({total_hc} HCs). Using test_size=0.20")

# Split HCs into train and test BEFORE any harmonization
# Stratify by SITE to ensure all sites are represented in train set
train_idx, test_idx = train_test_split(
    range(len(hc_roi)),
    test_size=test_size,
    random_state=42,  # for reproducibility
    stratify=hc_covars['SITE']  # Keep site distribution balanced
)

print(f"\nHC Split: {len(train_idx)} train ({100*(1-test_size):.0f}%), {len(test_idx)} test ({100*test_size:.0f}%)")

# Create train and test subsets
hc_roi_train = hc_roi.iloc[train_idx].copy()
hc_covars_train = hc_covars_harm.iloc[train_idx].copy()
hc_filenames_train = hc_mri["Filename"].iloc[train_idx].values

hc_roi_test = hc_roi.iloc[test_idx].copy()
hc_covars_test = hc_covars_harm.iloc[test_idx].copy()
hc_filenames_test = hc_mri["Filename"].iloc[test_idx].values

# Check site distribution after split
print("\nSite distribution in HC TRAIN:")
site_dist_train = hc_covars.iloc[train_idx]['SITE'].value_counts().sort_index()
print(site_dist_train)

print("\nSite distribution in HC TEST:")
site_dist_test = hc_covars.iloc[test_idx]['SITE'].value_counts().sort_index()
print(site_dist_test)

# Check minimum HCs per site in training set
min_per_site_train = site_dist_train.min()
print(f"\nMinimum HCs per site in TRAIN set: {min_per_site_train}")
if min_per_site_train < 5:
    print("⚠️ WARNING: Some sites have <5 HCs in training set!")
    print("Sites with <5 HCs in training:")
    print(site_dist_train[site_dist_train < 5])

# Convert to NumPy arrays
hc_roi_train_np = hc_roi_train.values.astype(np.float64)
hc_roi_test_np = hc_roi_test.values.astype(np.float64)
pat_roi_np = pat_roi.values.astype(np.float64)

# Final variance check
hc_roi_var_train = np.var(hc_roi_train_np, axis=0)
hc_roi_var_test = np.var(hc_roi_test_np, axis=0)
pat_roi_var_final = np.var(pat_roi_np, axis=0)

print(f"\nFinal variance check:")
print(f"  HC TRAIN: {np.sum(hc_roi_var_train == 0)} features with no variance")
print(f"  HC TEST: {np.sum(hc_roi_var_test == 0)} features with no variance")
print(f"  Patients: {np.sum(pat_roi_var_final == 0)} features with no variance")

if np.sum(hc_roi_var_train == 0) > 0:
    print("⚠️ WARNING: Some features have no variance in training set!")

# ============================================
# FILTER PATIENTS: ONLY SITES PRESENT IN HCs
# ============================================

print("\n" + "="*60)
print("FILTERING PATIENTS TO VALID SITES ONLY")
print("="*60)

# Sites die im HC TRAIN set waren (das Model kennt nur diese!)
valid_sites = set(hc_covars_train['SITE'].unique())
print(f"\nValid sites (from HC TRAIN): {sorted(valid_sites)}")

# Check welche Patient-Sites NICHT im Training waren
patient_sites = set(pat_covars_harm['SITE'].unique())
invalid_sites = patient_sites - valid_sites

if invalid_sites:
    print(f"\n⚠️  WARNING: Found {len(invalid_sites)} sites in PATIENTS that were NOT in HC TRAIN:")
    for site in sorted(invalid_sites):
        n = (pat_covars_harm['SITE'] == site).sum()
        print(f"  - {site}: {n} patients")
    
    # Speichere Patienten mit invaliden Sites
    invalid_patients = pat_covars[pat_covars['SITE'].isin(invalid_sites)].copy()
    invalid_pat_path = f"{OUTPUT_DIR}/patients_invalid_sites.csv"
    invalid_patients.to_csv(invalid_pat_path, index=False)
    print(f"\n✓ Saved patients with invalid sites to: {invalid_pat_path}")
    
    # Filter Patienten zu nur validen Sites
    valid_mask = pat_covars_harm['SITE'].isin(valid_sites)
    
    print(f"\nFiltering patients:")
    print(f"  Before: {len(pat_covars_harm)} patients")
    print(f"  Removing: {(~valid_mask).sum()} patients from invalid sites")
    
    pat_covars_harm = pat_covars_harm[valid_mask].copy()
    pat_roi_np = pat_roi_np[valid_mask]
    pat_mri = pat_mri[valid_mask].reset_index(drop=True)
    pat_covars = pat_covars[valid_mask].reset_index(drop=True)
    
    print(f"  After: {len(pat_covars_harm)} patients")
else:
    print("✓ All patient sites are valid!")

print(f"\nFinal patient sites: {sorted(pat_covars_harm['SITE'].unique())}")
print(f"Final patient count: {len(pat_covars_harm)}")

# ============================================
# LEARN HARMONIZATION MODEL (TRAIN HCs ONLY)
# ============================================

print("\n" + "="*60)
print("LEARNING HARMONIZATION MODEL ON TRAIN HCs ONLY")
print("="*60)

print("\nModel configuration:")
print("  - Training on: HC TRAIN set only (no data leakage!)")
print("  - Smooth terms: ['Age'] (non-linear age effects)")
print("  - Linear covariates: TIV, IQR, Sex_Male (effects preserved)")
print("  - Harmonized variable: SITE (effects removed)")

model_smoothage, data_adj_train = harmonizationLearn(
    data=hc_roi_train_np,
    covars=hc_covars_train,
    smooth_terms=["Age"]  # Age has non-linear effects on brain structure
)

print("\n✓ Learned neuroHarmonize model with smooth age term on TRAIN HCs only.")

# Save the learned model
model_path = f"{OUTPUT_DIR}/neuroharmonize_model_smoothage.joblib"
joblib.dump(model_smoothage, model_path)
print(f"✓ Saved model to: {model_path}")

# ============================================
# APPLY HARMONIZATION (TEST HCs + PATIENTS)
# ============================================

print("\n" + "="*60)
print("APPLYING HARMONIZATION TO TEST HCs AND PATIENTS")
print("="*60)

# Apply to test HCs (these did NOT contribute to learning the model!)
print("\nApplying to HC TEST set...")
data_adj_test = harmonizationApply(
    data=hc_roi_test_np,
    covars=hc_covars_test,
    model=model_smoothage
)
print("✓ Applied neuroHarmonize model to TEST HCs.")

# Apply to all patients
print("\nApplying to PATIENTS...")
data_adj_pat = harmonizationApply(
    data=pat_roi_np,
    covars=pat_covars_harm,
    model=model_smoothage
)
print("✓ Applied neuroHarmonize model to PATIENTS.")

# ============================================================
# PFADE ANPASSEN
# ============================================================

OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results"

# RAW DATA (vor Harmonization)
roi_raw_hc_train_path = f"{OUTPUT_DIR}/hc_train_roi_raw.csv"
roi_raw_hc_test_path = f"{OUTPUT_DIR}/hc_test_roi_raw.csv"
roi_raw_pat_path = f"{OUTPUT_DIR}/pat_roi_raw.csv"

# HARMONIZED DATA (nach Harmonization)
roi_harm_hc_train_path = f"{OUTPUT_DIR}/hc_train_roi_harmonized.csv"
roi_harm_hc_test_path = f"{OUTPUT_DIR}/hc_test_roi_harmonized.csv"
roi_harm_pat_path = f"{OUTPUT_DIR}/pat_roi_harmonized.csv"

# METADATA
metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/metadata_HARMONIZE_READY.csv"

# OUTPUT
output_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/evaluation_plots"

# ============================================
# SAVE ALL HARMONIZED DATA
# ============================================

print("\n" + "="*60)
print("SAVING HARMONIZED DATA")
print("="*60)

# Save HC TRAIN (harmonized)
hc_train_df = pd.DataFrame(
    data_adj_train,
    index=hc_filenames_train,
    columns=hc_roi.columns
)
hc_train_path = f"{OUTPUT_DIR}/hc_train_harmonized.csv"
hc_train_df.to_csv(hc_train_path, index_label="Filename")
print(f"✓ Saved HC TRAIN (harmonized): {hc_train_path}")

# Save HC TEST (harmonized)
hc_test_df = pd.DataFrame(
    data_adj_test,
    index=hc_filenames_test,
    columns=hc_roi.columns
)
hc_test_path = f"{OUTPUT_DIR}/hc_test_harmonized.csv"
hc_test_df.to_csv(hc_test_path, index_label="Filename")
print(f"✓ Saved HC TEST (harmonized): {hc_test_path}")

# Save PATIENTS (harmonized)
pat_df = pd.DataFrame(
    data_adj_pat,
    index=pat_mri["Filename"],
    columns=pat_roi.columns
)
pat_path = f"{OUTPUT_DIR}/patients_harmonized.csv"
pat_df.to_csv(pat_path, index_label="Filename")
print(f"✓ Saved PATIENTS (harmonized): {pat_path}")

# Save split information for later reference
split_info = pd.DataFrame({
    'Filename': list(hc_filenames_train) + list(hc_filenames_test),
    'Split': ['train'] * len(hc_filenames_train) + ['test'] * len(hc_filenames_test)
})
split_info_path = f"{OUTPUT_DIR}/hc_split_info.csv"
split_info.to_csv(split_info_path, index=False)
print(f"✓ Saved split info: {split_info_path}")

# Save excluded patients info
excluded_patients = all_mri_data[all_mri_data['Filename'].isin(
    pd.read_csv(PAT_METADATA)['Filename']
)][all_mri_data['Filename'].isin(
    pd.read_csv(PAT_METADATA)[pd.read_csv(PAT_METADATA)['Dataset'].isin(EXCLUDE_DATASETS)]['Filename']
)]
if len(excluded_patients) > 0:
    excluded_path = f"{OUTPUT_DIR}/excluded_patients.csv"
    excluded_patients[['Filename']].to_csv(excluded_path, index=False)
    print(f"✓ Saved excluded patients list: {excluded_path}")

# Also save as NumPy arrays if needed
np.save(f"{OUTPUT_DIR}/hc_train_harmonized.npy", data_adj_train)
np.save(f"{OUTPUT_DIR}/hc_test_harmonized.npy", data_adj_test)
np.save(f"{OUTPUT_DIR}/patients_harmonized.npy", data_adj_pat)
print(f"✓ Saved NumPy arrays.")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("HARMONIZATION COMPLETE (NO DATA LEAKAGE!)")
print("="*60)

print(f"\nExcluded datasets (no HCs): {EXCLUDE_DATASETS}")
print(f"Excluded patients: {n_excluded}")

print(f"\nData shapes:")
print(f"  HC TRAIN (for your model training): {hc_train_df.shape}")
print(f"  HC TEST (for your model testing): {hc_test_df.shape}")
print(f"  PATIENTS (for your model): {pat_df.shape}")

print(f"\nSites included:")
print(f"  {len(sites_in_hc)} sites with HCs: {sorted(sites_in_hc)}")
print(f"  Min HCs per site (train): {min_per_site_train}")

print(f"\nCovariates used:")
print(f"  Harmonized: SITE (scanner effects removed)")
print(f"  Preserved: Age (smooth), TIV, IQR, Sex_Male (linear)")

print(f"\nNext steps:")
print(f"  1. Use 'hc_train_harmonized.csv' to train your classification model")
print(f"  2. Use 'hc_test_harmonized.csv' ONLY for final evaluation")
print(f"  3. Use 'patients_harmonized.csv' for patient predictions")
print(f"  4. All harmonized data has SITE effects removed but biological effects preserved")

print("="*60)