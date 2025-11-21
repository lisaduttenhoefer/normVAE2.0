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
# SCHRITT 7: TRAIN/TEST SPLIT
# ============================================================

print("\n" + "="*70)
print("STEP 7: TRAIN/TEST SPLIT")
print("="*70)

total_hc = len(hc_roi)
test_size = 0.5

train_idx, test_idx = train_test_split(
    range(len(hc_roi)),
    test_size=test_size,
    random_state=42,
    stratify=hc_covars['SITE']
)

# 1. HC TRAIN Subsets
hc_roi_train = hc_roi.iloc[train_idx].copy()
hc_covars_train = hc_covars_harm.iloc[train_idx].copy()
# ⭐ NEU: Filenames als Index für HC TRAIN speichern
hc_filenames_train = hc_covars['Filename'].iloc[train_idx].values 

# 2. HC TEST Subsets
hc_roi_test = hc_roi.iloc[test_idx].copy()
hc_covars_test = hc_covars_harm.iloc[test_idx].copy()
# ⭐ NEU: Filenames als Index für HC TEST speichern
hc_filenames_test = hc_covars['Filename'].iloc[test_idx].values
pat_filenames = pat_covars['Filename'].values


# 3. KONKATENATION (HC TEST + PATIENTS) für die Anwendung
app_roi = pd.concat([hc_roi_test, pat_roi], ignore_index=False)
app_covars = pd.concat([hc_covars_test, pat_covars_harm], ignore_index=False)

# ⭐ NEU: Filenames für das Application Set erstellen
app_filenames = pd.Series(
    np.concatenate([hc_filenames_test, pat_filenames]), 
    index=app_roi.index
)


# ============================================================
# SCHRITT 8/9: REORDERING & CLAMPING
# ============================================================

print("\n" + "="*70)
print("STEP 8/9: REORDERING & CLAMPING")
print("="*70)

min_age_train = hc_covars_train['Age'].min()
max_age_train = hc_covars_train['Age'].max()

# Clamping
app_covars.loc[app_covars['Age'] < min_age_train, 'Age'] = min_age_train
app_covars.loc[app_covars['Age'] > max_age_train, 'Age'] = max_age_train

# Finales Reordering
present_covars = hc_covars_train.columns.tolist()

# Ensure SITE is first, then the remaining covariates
expected_order = ['SITE']
for col in present_covars:
    if col != 'SITE':
        expected_order.append(col)

print(f"\nFinal expected order: {expected_order}") # Will now be ['SITE', 'Age', 'TIV', 'Sex_Male'] (no IQR)

hc_covars_train = hc_covars_train[expected_order] 
app_covars = app_covars[expected_order]

# Konvertierung in NumPy (jetzt ohne 'Filename'-Spalte)
hc_roi_train_np = hc_roi_train.values.astype(np.float64)
app_roi_np = app_roi.values.astype(np.float64)

# ⭐ NEU: Letzter Check auf Null-Varianz in HC TRAIN (Fix RuntimeWarning)
print("\nChecking final ROI variance in HC TRAIN...")

# Die Varianz muss über die Spalten (axis=0) berechnet werden
train_variances = hc_roi_train.var(axis=0)

# Finde alle ROIs mit Varianz nahe Null
zero_var_rois = train_variances[train_variances < 1e-6].index.tolist() 

if zero_var_rois:
    print(f"⚠️ Removing {len(zero_var_rois)} ROIs from TRAIN/APPLICATION due to near-zero variance.")
    
    # Entfernen der Spalten in allen betroffenen DataFrames
    hc_roi_train = hc_roi_train.drop(columns=zero_var_rois)
    app_roi = app_roi.drop(columns=zero_var_rois)
    
    print(f"New ROI count: {len(hc_roi_train.columns)}")
else:
    print("✅ No additional zero-variance ROIs found in HC TRAIN.")

# Konvertierung in NumPy (jetzt ohne problematische ROIs)
hc_roi_train_np = hc_roi_train.values.astype(np.float64)
app_roi_np = app_roi.values.astype(np.float64)

# ============================================================
# SCHRITT 10: LEARN HARMONIZATION MODEL (LINEAR AGE)
# ============================================================

print("\n" + "="*70)
print("STEP 10: LEARNING HARMONIZATION MODEL (LINEAR AGE)")
print("="*70)

model_smoothage, data_adj_train = harmonizationLearn(
    data=hc_roi_train_np,
    covars=hc_covars_train,
    smooth_terms=[] # LINEAR MODEL
)

print("\n✓ Learned neuroHarmonize model!")

# Save model
model_path = f"{OUTPUT_DIR}/neuroharmonize_model_linearage_noNU_50.joblib" 
joblib.dump(model_smoothage, model_path)
print(f"✓ Saved model to: {model_path}")

# ============================================================
# SCHRITT 11: APPLY HARMONIZATION
# ============================================================

print("\n" + "="*70)
print("STEP 11: APPLYING HARMONIZATION (TO COMBINED SET)")
print("="*70)

data_adj_app = harmonizationApply(
    data=app_roi_np,
    covars=app_covars,
    model=model_smoothage
)
print("✓ Applied to combined set.")

# ============================================================
# SCHRITT 12/13: SAVE DATA
# ============================================================

print("\n" + "="*70)
print("STEP 12/13: SAVING DATA (IDS GESICHERT)")
print("="*70)

# HC TRAIN harmonized
# Speichern Sie die endgültig gefilterten Spaltennamen
final_roi_columns = hc_roi_train.columns

# HC TRAIN harmonized
hc_train_harm_df = pd.DataFrame(
    data_adj_train,
    index=hc_filenames_train,
    columns=final_roi_columns # ⭐ FIX: Verwenden der Spalten des gefilterten Trainings-Sets
)
hc_train_harm_path = f"{OUTPUT_DIR}/hc_train_roi_harmonized_50.csv"
hc_train_harm_df.to_csv(hc_train_harm_path, index_label="Filename")

# APPLICATION harmonized
app_harm_df = pd.DataFrame(
    data_adj_app,
    index=app_filenames.values,
    columns=final_roi_columns # ⭐ FIX: Verwenden der Spalten des gefilterten Trainings-Sets
)
app_harm_path = f"{OUTPUT_DIR}/application_roi_harmonized_noNU_50.csv"
app_harm_df.to_csv(app_harm_path, index_label="Filename")

# Split info
split_info = pd.DataFrame({
    'Filename': app_filenames.values,
    'Split': ['test'] * len(hc_filenames_test) + ['patient'] * len(pat_filenames)
})
split_path = f"{OUTPUT_DIR}/app_split_info_noNU_50.csv"
split_info.to_csv(split_path, index=False)
print(f"✓ Harmonization complete. Final data saved without NU.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ HARMONIZATION COMPLETE (TESTING WITHOUT NU)!")
print("="*70)

print("\n🎯 Nächster Schritt: Führen Sie das Visualisierungs-Skript aus, um zu prüfen, ob die Harmonisierung erfolgreich war.")