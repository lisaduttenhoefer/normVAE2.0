from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os

# ============================================================
# PFADE DEFINIEREN
# ============================================================

HC_METADATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_metadata/metadata_for_harmonizing_hc.csv"
PAT_METADATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_metadata/metadata_for_harmonizing_patients.csv"
MRI_DATA = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"
OUTPUT_DIR = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⭐ NEU: Datasets to exclude (NU wird für den Test hinzugefügt)
EXCLUDE_DATASETS = ['NSS', 'EPSY', 'NU'] 

# ============================================================
# SCHRITT 1: DATEN LADEN
# ============================================================

print("="*70)
print("STEP 1: LOADING DATA")
print("="*70)

hc_covars = pd.read_csv(HC_METADATA)
pat_covars = pd.read_csv(PAT_METADATA)
all_mri_data = pd.read_csv(MRI_DATA)

# ============================================================
# SCHRITT 2: DATASETS OHNE HCs AUSSCHLIESSEN (Nun inklusive NU)
# ============================================================

print("\n" + "="*70)
print("STEP 2: EXCLUDING DATASETS (NSS, EPSY, NU)")
print("="*70)

# Filter HC und Patient Metadaten basierend auf den auszuschließenden Datasets
hc_covars = hc_covars[~hc_covars['Dataset'].isin(EXCLUDE_DATASETS)].copy()
pat_covars = pat_covars[~pat_covars['Dataset'].isin(EXCLUDE_DATASETS)].copy()
print(f"HC Metadata (nach Ausschluss): {hc_covars.shape}")
print(f"Patient Metadata (nach Ausschluss): {pat_covars.shape}")


# ============================================================
# SCHRITT 3: MRI DATEN VORBEREITEN
# ============================================================

print("\n" + "="*70)
print("STEP 3: PREPARING MRI DATA")
print("="*70)

non_roi_columns = ['Filename', 'Dataset', 'IQR', 'NCR', 'ICR', 'res_RMS', 'TIV', 
                   'GM_vol', 'WM_vol', 'CSF_vol', 'WMH_vol', 
                   'mean_thickness_lh', 'mean_thickness_rh', 'mean_thickness_global',
                   'mean_gyri_lh', 'mean_gyri_rh', 'mean_gyri_global']

hc_filenames = hc_covars['Filename'].values
pat_filenames = pat_covars['Filename'].values

hc_mri = all_mri_data[all_mri_data['Filename'].isin(hc_filenames)].copy()
pat_mri = all_mri_data[all_mri_data['Filename'].isin(pat_filenames)].copy()

# Sortierung und Abgleich (wichtig für die Trennung in Schritt 7)
hc_mri = hc_mri.sort_values('Filename').reset_index(drop=True)
hc_covars = hc_covars.sort_values('Filename').reset_index(drop=True)
pat_mri = pat_mri.sort_values('Filename').reset_index(drop=True)
pat_covars = pat_covars.sort_values('Filename').reset_index(drop=True)

# ⭐ Filename muss im Index für die Sortierung sein, aber NICHT in den ROI-Spalten
roi_columns = [col for col in hc_mri.columns if col not in non_roi_columns]

hc_roi = hc_mri[roi_columns].copy()
pat_roi = pat_mri[roi_columns].copy()

# ... (Ihre Logik für Zero-Variance Features) ...
# Ensure both datasets have same ROI columns
common_rois = list(set(hc_roi.columns) & set(pat_roi.columns)) 
hc_roi = hc_roi[common_rois]
pat_roi = pat_roi[common_rois]


# ============================================================
# SCHRITT 4: COVARIATES VORBEREITEN
# ============================================================
print("\n" + "="*70)
print("STEP 4: PREPARING COVARIATES (IQR EXCLUDED)")
print("="*70)

# Required covariates
# ⭐ ÄNDERUNG: 'IQR' wird aus den Kovariaten entfernt!
required_covars = ['Age', 'TIV', 'SITE'] 

# Dynamischer Check der existierenden Sex-Spalte
sex_col = None
for col in ['Sex_Male', 'Sex_M', 'Sex_F', 'Sex']:
    if col in hc_covars.columns:
        sex_col = col
        break

if sex_col is None:
    raise ValueError("No Sex column found!")

required_covars.append(sex_col)

print(f"\nUsing covariates: {required_covars}") # Wird nun ['Age', 'TIV', 'SITE', 'Sex_Male'] sein

# HIER WERDEN DIE DATAFRAMES ERSTELLT:
hc_covars_harm = hc_covars[required_covars].copy()
pat_covars_harm = pat_covars[required_covars].copy()

# ============================================================
# SCHRITT 7: STRATIFIED SPLIT FÜR HARMONIZATION
# ============================================================

print("\n" + "="*70)
print("STEP 7: STRATIFIED SPLIT (20% for harmonization learning)")
print("="*70)

# 1. Split: 20% für Harmonization Learning, 80% für alles andere
from sklearn.model_selection import train_test_split

harm_learn_idx, harm_apply_idx = train_test_split(
    range(len(hc_roi)),
    test_size=0.8,  # 80% werden NICHT für Harmonization-Learning genutzt
    random_state=42,
    stratify=hc_covars['SITE']
)

print(f"HCs for learning harmonization: {len(harm_learn_idx)}")
print(f"HCs for VAE (after harmonization): {len(harm_apply_idx)}")

# ============================================================
# HARMONIZATION LEARNING SUBSET (20% HCs)
# ============================================================

hc_roi_harm_learn = hc_roi.iloc[harm_learn_idx].copy()
hc_covars_harm_learn = hc_covars_harm.iloc[harm_learn_idx].copy()
hc_filenames_harm_learn = hc_covars['Filename'].iloc[harm_learn_idx].values

print("\nSite distribution in harmonization learning set:")
print(hc_covars.iloc[harm_learn_idx]['SITE'].value_counts())

# ============================================================
# HARMONIZATION APPLICATION DATA (80% HCs + ALL patients)
# ============================================================

# Remaining 80% HCs
hc_roi_for_vae = hc_roi.iloc[harm_apply_idx].copy()
hc_covars_for_vae = hc_covars_harm.iloc[harm_apply_idx].copy()
hc_filenames_for_vae = hc_covars['Filename'].iloc[harm_apply_idx].values

# Combine with ALL patients
app_roi = pd.concat([hc_roi_for_vae, pat_roi], ignore_index=False)
app_covars = pd.concat([hc_covars_for_vae, pat_covars_harm], ignore_index=False)
app_filenames = pd.Series(
    np.concatenate([hc_filenames_for_vae, pat_filenames]), 
    index=app_roi.index
)

print(f"\nTotal application set size: {len(app_roi)}")
print(f"  - HCs for VAE: {len(hc_roi_for_vae)}")
print(f"  - Patients: {len(pat_roi)}")

# ============================================================
# SCHRITT 8/9: AGE CLAMPING (basiert auf LEARNING set)
# ============================================================

print("\n" + "="*70)
print("STEP 8/9: AGE CLAMPING")
print("="*70)

min_age_learn = hc_covars_harm_learn['Age'].min()
max_age_learn = hc_covars_harm_learn['Age'].max()

print(f"Age range in harmonization learning set: [{min_age_learn:.1f}, {max_age_learn:.1f}]")

# Clamp application data
app_covars.loc[app_covars['Age'] < min_age_learn, 'Age'] = min_age_learn
app_covars.loc[app_covars['Age'] > max_age_learn, 'Age'] = max_age_learn

# Reorder covariates
expected_order = ['SITE', 'Age', 'TIV', sex_col]
hc_covars_harm_learn = hc_covars_harm_learn[expected_order]
app_covars = app_covars[expected_order]

# ⚠️ CRITICAL: Check for zero-variance BEFORE converting to NumPy
print("\nChecking final ROI variance in HARMONIZATION LEARNING SET...")

train_variances = hc_roi_harm_learn.var(axis=0)
zero_var_rois = train_variances[train_variances < 1e-6].index.tolist() 

if zero_var_rois:
    print(f"⚠️ Removing {len(zero_var_rois)} ROIs due to near-zero variance.")
    
    hc_roi_harm_learn = hc_roi_harm_learn.drop(columns=zero_var_rois)
    app_roi = app_roi.drop(columns=zero_var_rois)
    
    print(f"New ROI count: {len(hc_roi_harm_learn.columns)}")
else:
    print("✅ No zero-variance ROIs found.")

# NOW convert to NumPy (after cleaning)
hc_roi_harm_learn_np = hc_roi_harm_learn.values.astype(np.float64)
app_roi_np = app_roi.values.astype(np.float64)

# ============================================================
# SCHRITT 10: LEARN HARMONIZATION (nur auf 20% HCs)
# ============================================================

print("\n" + "="*70)
print("STEP 10: LEARNING HARMONIZATION (on 20% HC subset)")
print("="*70)

model_smoothage, data_adj_learn = harmonizationLearn(
    data=hc_roi_harm_learn_np,
    covars=hc_covars_harm_learn,
    smooth_terms=[]  # Linear model
)

print("✓ Learned neuroHarmonize model on SEPARATE HC subset!")

model_path = f"{OUTPUT_DIR}/neuroharmonize_model_separate_subset.joblib"
joblib.dump(model_smoothage, model_path)

# ============================================================
# SCHRITT 11: APPLY TO APPLICATION SET (80% HCs + patients)
# ============================================================

print("\n" + "="*70)
print("STEP 11: APPLYING HARMONIZATION")
print("="*70)

data_adj_app = harmonizationApply(
    data=app_roi_np,
    covars=app_covars,
    model=model_smoothage
)

print("✓ Applied harmonization to application set")
print("  → This data will be used for VAE training AND testing")
# ============================================================
# PREPARE METADATA FOR SAVING
# ============================================================

# WICHTIG: Vollständige Metadata speichern, nicht nur Covariates!
# Get original metadata for subjects used in VAE (80%)
vae_hc_metadata = hc_covars.iloc[harm_apply_idx].copy()  # ← ORIGINAL hc_covars, nicht hc_covars_harm!

# Get original metadata for harmonization learning subjects (20%)
harm_learn_metadata = hc_covars.iloc[harm_learn_idx].copy()

# Ensure required columns exist
required_cols = ['Filename', 'Age', 'Sex', 'Dataset', 'Diagnosis', 'SITE']
for col in required_cols:
    if col not in vae_hc_metadata.columns:
        print(f"⚠️  WARNING: Column '{col}' missing in metadata!")

print(f"\n[DEBUG] vae_hc_metadata columns: {vae_hc_metadata.columns.tolist()}")
print(f"[DEBUG] vae_hc_metadata shape: {vae_hc_metadata.shape}")

# ============================================================
# SCHRITT 12: SAVE HARMONIZED DATA
# ============================================================

print("\n" + "="*70)
print("STEP 12: SAVING HARMONIZED DATA")
print("="*70)

# ⚠️ WICHTIG: data_adj_app ist NumPy array, muss korrekt zu DataFrame konvertiert werden!

# Get the ROI column names (from the harmonization learning set)
roi_column_names = hc_roi_harm_learn.columns.tolist()

print(f"[DEBUG] data_adj_app shape: {data_adj_app.shape}")
print(f"[DEBUG] Number of HC filenames: {len(hc_filenames_for_vae)}")
print(f"[DEBUG] Number of patient filenames: {len(pat_filenames)}")
print(f"[DEBUG] Expected ROI columns: {len(roi_column_names)}")

# Check dimensions match
expected_total_rows = len(hc_filenames_for_vae) + len(pat_filenames)
if data_adj_app.shape[0] != expected_total_rows:
    raise ValueError(f"Size mismatch! data_adj_app has {data_adj_app.shape[0]} rows but expected {expected_total_rows}")

if data_adj_app.shape[1] != len(roi_column_names):
    raise ValueError(f"Column mismatch! data_adj_app has {data_adj_app.shape[1]} columns but expected {len(roi_column_names)}")

# Split data_adj_app into HC and patient portions
hc_data_harmonized = data_adj_app[:len(hc_filenames_for_vae), :]
pat_data_harmonized = data_adj_app[len(hc_filenames_for_vae):, :]

print(f"[DEBUG] HC data shape: {hc_data_harmonized.shape}")
print(f"[DEBUG] Patient data shape: {pat_data_harmonized.shape}")

# Create DataFrames
hc_for_vae_df = pd.DataFrame(
    hc_data_harmonized,
    index=hc_filenames_for_vae,
    columns=roi_column_names
)

patients_harm_df = pd.DataFrame(
    pat_data_harmonized,
    index=pat_filenames,
    columns=roi_column_names
)

# Verify no empty data
print(f"\n[DEBUG] HC DataFrame shape: {hc_for_vae_df.shape}")
print(f"[DEBUG] HC DataFrame has data: {not hc_for_vae_df.empty}")
print(f"[DEBUG] HC DataFrame first value: {hc_for_vae_df.iloc[0, 0]}")
print(f"[DEBUG] Patient DataFrame shape: {patients_harm_df.shape}")
print(f"[DEBUG] Patient DataFrame has data: {not patients_harm_df.empty}")
print(f"[DEBUG] Patient DataFrame first value: {patients_harm_df.iloc[0, 0]}")

# Check for NaN
hc_nan_count = hc_for_vae_df.isna().sum().sum()
pat_nan_count = patients_harm_df.isna().sum().sum()

if hc_nan_count > 0:
    print(f"⚠️  WARNING: HC data has {hc_nan_count} NaN values")
if pat_nan_count > 0:
    print(f"⚠️  WARNING: Patient data has {pat_nan_count} NaN values")

# Save harmonized HCs
hc_harmonized_path = f"{OUTPUT_DIR}/hc_harmonized_for_vae.csv"
hc_for_vae_df.to_csv(hc_harmonized_path, index_label="Filename")
print(f"✓ Saved {len(hc_filenames_for_vae)} harmonized HCs: {hc_harmonized_path}")

# Verify file was written correctly
verify_df = pd.read_csv(hc_harmonized_path, nrows=5)
print(f"[VERIFY] Saved file has {len(verify_df)} rows (showing first 5)")
print(f"[VERIFY] Saved file has {len(verify_df.columns)} columns")

# Save harmonized patients
patients_harmonized_path = f"{OUTPUT_DIR}/patients_harmonized.csv"
patients_harm_df.to_csv(patients_harmonized_path, index_label="Filename")
print(f"✓ Saved {len(pat_filenames)} harmonized patients: {patients_harmonized_path}")

# Save metadata for HCs that can be used for VAE
vae_hc_metadata_path = f"{OUTPUT_DIR}/hc_metadata_for_vae.csv"
vae_hc_metadata.to_csv(vae_hc_metadata_path, index=False)
print(f"✓ Saved HC metadata: {vae_hc_metadata_path}")

# Verify metadata was written correctly
verify_meta = pd.read_csv(vae_hc_metadata_path, nrows=5)
print(f"[VERIFY] Metadata has {len(verify_meta.columns)} columns: {verify_meta.columns.tolist()}")

# Save metadata for patients (with Sex column added)
patient_metadata_path = f"{OUTPUT_DIR}/patient_metadata.csv"

# Load original patient metadata
pat_metadata_full = pd.read_csv(PAT_METADATA)

# Apply exclusions if needed
if EXCLUDE_DATASETS:
    pat_metadata_full = pat_metadata_full[~pat_metadata_full['Dataset'].isin(EXCLUDE_DATASETS)].copy()

# Add 'Sex' column if missing
if 'Sex' not in pat_metadata_full.columns and 'Sex_Male' in pat_metadata_full.columns:
    pat_metadata_full['Sex'] = pat_metadata_full['Sex_Male'].map({1: 'M', 0: 'F'})

pat_metadata_full.to_csv(patient_metadata_path, index=False)
print(f"✓ Saved patient metadata: {patient_metadata_path}")

# CRITICAL: Save which HCs were used for harmonization (EXCLUDE!)
harm_learn_path = f"{OUTPUT_DIR}/hc_used_for_harmonization_EXCLUDE.csv"
harm_learn_metadata.to_csv(harm_learn_path, index=False)
print(f"⚠️  EXCLUDE from VAE: {harm_learn_path} ({len(harm_learn_idx)} HCs)")

print("\n" + "="*70)
print("✅ HARMONIZATION COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"  - Harmonization learned on: {len(harm_learn_idx)} HCs (EXCLUDED)")
print(f"  - Available for VAE: {len(hc_filenames_for_vae)} HCs")
print(f"  - Patients: {len(pat_filenames)}")
print(f"\n🎯 Training Script will split the {len(hc_filenames_for_vae)} HCs into train/test")