import pandas as pd
import numpy as np

# ============================================================
# PFADE DEFINIEREN
# ============================================================

# Input files
metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/metadata_combat.csv"
tiv_iqr_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"  # ⭐ Enthält jetzt TIV UND IQR
srpbs_hc_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/hc_SRPBS_covariates.csv"
srpbs_pat_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/patient_SRPBS_covariates.csv"
tcp_covariates_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/all_tcp_covariates.csv"

# Output files
output_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro"
harmonize_ready_path = f"{output_dir}/metadata_HARMONIZE_READY.csv"
hc_harmonize_path = f"{output_dir}/metadata_HARMONIZE_READY_HC.csv"
pat_harmonize_path = f"{output_dir}/metadata_HARMONIZE_READY_PATIENTS.csv"
tcp_missing_output = f"{output_dir}/TCP_subjects_REMOVED_missing_covariates.csv"
site_mapping_path = f"{output_dir}/SRPBS_site_abbreviations.csv"

# ============================================================
# SCHRITT 1: METADATA LADEN
# ============================================================

print("="*70)
print("STEP 1: LOADING METADATA")
print("="*70)

metadata = pd.read_csv(metadata_path)

print(f"Original metadata shape: {metadata.shape}")
print("\nDataset distribution:")
print(metadata['Dataset'].value_counts())
print("\nDiagnosis distribution (before TCP update):")
print(metadata['Diagnosis'].value_counts())

# Nur benötigte Spalten
columns_to_keep = ['Filename', 'Dataset', 'Diagnosis', 'Age', 'Sex']
metadata_clean = metadata[columns_to_keep].copy()

# ============================================================
# SCHRITT 2: TCP DIAGNOSIS AKTUALISIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 2: UPDATING TCP DIAGNOSIS")
print("="*70)

tcp_covars = pd.read_csv(tcp_covariates_path)
print(f"TCP covariates loaded: {tcp_covars.shape}")

if 'record_id' not in tcp_covars.columns or 'Diagnosis' not in tcp_covars.columns:
    raise ValueError("TCP covariates missing required columns!")

tcp_diagnosis = tcp_covars[['record_id', 'Diagnosis']].copy()
tcp_diagnosis = tcp_diagnosis.rename(columns={'record_id': 'Filename', 'Diagnosis': 'Diagnosis_TCP'})

tcp_subjects = metadata_clean[metadata_clean['Dataset'] == 'TCP'].copy()
print(f"TCP subjects in metadata: {len(tcp_subjects)}")

tcp_updated = pd.merge(
    tcp_subjects,
    tcp_diagnosis,
    on='Filename',
    how='left',
    indicator=True
)

tcp_missing = tcp_updated[tcp_updated['_merge'] == 'left_only'].copy()

if len(tcp_missing) > 0:
    print(f"\n⚠️  {len(tcp_missing)} TCP subjects NOT found in covariates - will be REMOVED")
    tcp_missing[['Filename', 'Dataset', 'Diagnosis', 'Age', 'Sex']].to_csv(tcp_missing_output, index=False)
    print(f"✓ Saved removed subjects to: {tcp_missing_output}")

tcp_valid = tcp_updated[tcp_updated['_merge'] == 'both'].copy()
tcp_valid['Diagnosis'] = tcp_valid['Diagnosis_TCP']
tcp_valid = tcp_valid.drop(columns=['Diagnosis_TCP', '_merge'])

print(f"✓ TCP subjects with valid diagnosis: {len(tcp_valid)}")

non_tcp = metadata_clean[metadata_clean['Dataset'] != 'TCP'].copy()
metadata_clean = pd.concat([non_tcp, tcp_valid], ignore_index=True)

print(f"\nMetadata after TCP update: {metadata_clean.shape}")
print("Updated Diagnosis distribution:")
print(metadata_clean['Diagnosis'].value_counts())

# ============================================================
# SCHRITT 3: TIV UND IQR HINZUFÜGEN
# ============================================================

print("\n" + "="*70)
print("STEP 3: ADDING TIV AND IQR (Quality Control)")
print("="*70)

cat12_data = pd.read_csv(tiv_iqr_path)
print(f"CAT12 data shape: {cat12_data.shape}")

# ⭐ Check welche Spalten vorhanden sind
required_cols = ['Filename', 'TIV', 'IQR']
missing = [col for col in required_cols if col not in cat12_data.columns]
if missing:
    raise ValueError(f"Missing columns in CAT12 data: {missing}")

cat12_subset = cat12_data[['Filename', 'TIV', 'IQR']].copy()

metadata_with_qc = pd.merge(
    metadata_clean,
    cat12_subset,
    on='Filename',
    how='left'
)

print(f"\nMetadata after TIV/IQR merge: {metadata_with_qc.shape}")

n_missing_tiv = metadata_with_qc['TIV'].isna().sum()
n_missing_iqr = metadata_with_qc['IQR'].isna().sum()
print(f"Missing TIV: {n_missing_tiv}/{len(metadata_with_qc)}")
print(f"Missing IQR: {n_missing_iqr}/{len(metadata_with_qc)}")

if n_missing_iqr > 0:
    print("\n⭐ IQR statistics (for available subjects):")
    iqr_available = metadata_with_qc[metadata_with_qc['IQR'].notna()]
    print(f"  N: {len(iqr_available)}")
    print(f"  Min: {iqr_available['IQR'].min():.4f}")
    print(f"  Max: {iqr_available['IQR'].max():.4f}")
    print(f"  Mean: {iqr_available['IQR'].mean():.4f} ± {iqr_available['IQR'].std():.4f}")

# ============================================================
# SCHRITT 4: DATASET-NAMEN STANDARDISIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 4: STANDARDIZING DATASET NAMES")
print("="*70)

def standardize_dataset_name(dataset_name):
    ds = str(dataset_name).upper().strip()
    if any(variant in ds for variant in ['SRPBS', 'SRBPS', 'SPRBS']):
        return 'SRPBS'
    return dataset_name

metadata_with_qc['Dataset'] = metadata_with_qc['Dataset'].apply(standardize_dataset_name)

print("Standardized datasets:")
print(metadata_with_qc['Dataset'].value_counts())

# ============================================================
# SCHRITT 5: SRPBS SITES HINZUFÜGEN
# ============================================================

print("\n" + "="*70)
print("STEP 5: ADDING SRPBS SITE INFORMATION")
print("="*70)

srpbs_hc = pd.read_csv(srpbs_hc_path)
srpbs_pat = pd.read_csv(srpbs_pat_path)

srpbs_combined = pd.concat([
    srpbs_hc[['record_id', 'Site']],
    srpbs_pat[['record_id', 'Site']]
], ignore_index=True)

print(f"Combined SRPBS sites: {srpbs_combined.shape}")

# Site mapping mit Abkürzungen
site_mapping = {
    'Kyoto university': 'SRPBS_KU',
    'Kyoto University': 'SRPBS_KU',
    'Center of Innovation in Hiroshima University': 'SRPBS_COIH',
    'University of Tokyo': 'SRPBS_UT',
    'Hiroshima University Hospital': 'SRPBS_HUH',
    'Showa university': 'SRPBS_SU',
    'Hiroshima Kajikawa Hospital': 'SRPBS_HKH',
    'ATR': 'SRPBS_ATR',
    'CiNet': 'SRPBS_CiNet'
}

def standardize_site(site_name):
    if pd.isna(site_name):
        return site_name
    for original, abbrev in site_mapping.items():
        if site_name.strip().lower() == original.lower():
            return abbrev
    return 'SRPBS_Unknown'

srpbs_combined['SITE_temp'] = srpbs_combined['Site'].apply(standardize_site)
srpbs_combined = srpbs_combined.rename(columns={'record_id': 'Filename'})

metadata_with_site = pd.merge(
    metadata_with_qc,
    srpbs_combined[['Filename', 'SITE_temp']],
    on='Filename',
    how='left'
)

print("SRPBS sites distribution:")
print(srpbs_combined['SITE_temp'].value_counts())

# Finale SITE Variable erstellen
def create_final_site(row):
    if row['Dataset'] == 'SRPBS' and pd.notna(row['SITE_temp']):
        return row['SITE_temp']
    elif row['Dataset'] == 'SRPBS':
        return 'SRPBS_Unknown'
    else:
        return row['Dataset']

metadata_with_site['SITE'] = metadata_with_site.apply(create_final_site, axis=1)
metadata_with_site = metadata_with_site.drop(columns=['SITE_temp'])

print(f"\n✓ Final SITES: {metadata_with_site['SITE'].nunique()}")

# Speichere Site-Mapping
unique_mappings = {abbrev: orig for orig, abbrev in site_mapping.items() 
                   if abbrev not in ['SRPBS_KU'] or orig == 'Kyoto university'}
mapping_df = pd.DataFrame([
    {'Abbreviation': abbrev, 'Full_Name': name}
    for abbrev, name in unique_mappings.items()
])
mapping_df.to_csv(site_mapping_path, index=False)

# ============================================================
# SCHRITT 6: DATA QUALITY CHECKS
# ============================================================

print("\n" + "="*70)
print("STEP 6: DATA QUALITY SUMMARY")
print("="*70)

print(f"\n📊 Total subjects: {len(metadata_with_site)}")
print(f"\n⭐ Diagnosis distribution:")
for dx in sorted(metadata_with_site['Diagnosis'].unique()):
    n = (metadata_with_site['Diagnosis'] == dx).sum()
    pct = 100 * n / len(metadata_with_site)
    print(f"  {dx:15s}: {n:4d} ({pct:.1f}%)")

print(f"\n⭐ Age: {metadata_with_site['Age'].min():.1f} - {metadata_with_site['Age'].max():.1f} "
      f"(mean: {metadata_with_site['Age'].mean():.1f})")

print(f"\n⭐ Sex distribution:")
print(metadata_with_site['Sex'].value_counts())

print(f"\n⭐ Sites (n={metadata_with_site['SITE'].nunique()}):")
site_counts = metadata_with_site['SITE'].value_counts().sort_values(ascending=False)
for site, count in site_counts.head(15).items():
    print(f"  {site:20s}: {count:4d}")
if len(site_counts) > 15:
    print(f"  ... and {len(site_counts) - 15} more sites")

small_sites = site_counts[site_counts < 10]
if len(small_sites) > 0:
    print(f"\n⚠️  {len(small_sites)} sites with <10 subjects")

# ============================================================
# SCHRITT 7: KATEGORISCHE VARIABLEN ENCODEN
# ============================================================

print("\n" + "="*70)
print("STEP 7: ENCODING FOR NEUROHARMONIZE")
print("="*70)

# Identifiziere Patient-Diagnosen (alles außer HC)
all_diagnoses = metadata_with_site['Diagnosis'].unique()
patient_diagnoses = sorted([dx for dx in all_diagnoses if dx != 'HC'])

print(f"Patient diagnosis types: {patient_diagnoses}")

# Erstelle binäre Diagnose-Spalten
for dx in patient_diagnoses:
    col_name = f'Diagnosis_{dx}'
    metadata_with_site[col_name] = (metadata_with_site['Diagnosis'] == dx).astype(float)
    n = int(metadata_with_site[col_name].sum())
    print(f"  {col_name}: {n} subjects")

# Sex encoding
metadata_harmonize = pd.get_dummies(
    metadata_with_site,
    columns=['Sex'],
    drop_first=True,
    dtype=float
)

sex_cols = [col for col in metadata_harmonize.columns if 'Sex_' in col]
diagnosis_cols = [col for col in metadata_harmonize.columns if 'Diagnosis_' in col]

print(f"\n✓ Sex column: {sex_cols}")
print(f"✓ Diagnosis columns: {diagnosis_cols}")

# Spalten für Harmonization
harmonization_cols = ['SITE', 'Age', 'IQR', 'TIV'] + sex_cols + diagnosis_cols

print(f"\n⭐ Columns for harmonization:")
print(f"  🎯 Batch (removed): SITE")
print(f"  🛡️  Protected (preserved):")
for col in harmonization_cols:
    if col != 'SITE':
        print(f"     - {col}")

# ============================================================
# SCHRITT 8: FINALE FILES SPEICHERN
# ============================================================

print("\n" + "="*70)
print("STEP 8: SAVING FINAL FILES")
print("="*70)

# Alle Subjects
metadata_harmonize.to_csv(harmonize_ready_path, index=False)
print(f"✓ All subjects: {harmonize_ready_path}")

# HC only
hc_harmonize = metadata_harmonize[metadata_harmonize['Diagnosis'] == 'HC'].copy()
hc_harmonize.to_csv(hc_harmonize_path, index=False)
print(f"✓ HC only ({len(hc_harmonize)} subjects): {hc_harmonize_path}")

# Patients only
pat_harmonize = metadata_harmonize[metadata_harmonize['Diagnosis'] != 'HC'].copy()
pat_harmonize.to_csv(pat_harmonize_path, index=False)
print(f"✓ Patients only ({len(pat_harmonize)} subjects): {pat_harmonize_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ METADATA PREPARATION COMPLETE!")
print("="*70)

print(f"\n📁 Files created:")
print(f"1. {harmonize_ready_path}")
print(f"2. {hc_harmonize_path}")
print(f"3. {pat_harmonize_path}")
print(f"4. {site_mapping_path}")
if len(tcp_missing) > 0:
    print(f"5. {tcp_missing_output}")

print(f"\n📊 Summary:")
print(f"  Total subjects: {len(metadata_harmonize)}")
print(f"    HC: {len(hc_harmonize)}")
print(f"    Patients: {len(pat_harmonize)}")
print(f"  Unique sites: {metadata_harmonize['SITE'].nunique()}")
print(f"  Diagnosis types: {len(patient_diagnoses) + 1} (HC + {len(patient_diagnoses)} patient types)")

print(f"\n⭐ Quality Control (IQR):")
iqr_stats = metadata_harmonize['IQR'].describe()
print(f"  Available: {metadata_harmonize['IQR'].notna().sum()}/{len(metadata_harmonize)}")
print(f"  Range: {iqr_stats['min']:.4f} - {iqr_stats['max']:.4f}")
print(f"  Mean: {iqr_stats['mean']:.4f} ± {iqr_stats['std']:.4f}")

print("\n🎯 Ready for neuroHarmonize!")
print("   Use these columns as covariates:")
print(f"   covars = data[{harmonization_cols}]")

print("\n" + "="*70)