# run with conda environment: umap_env
"""
Evaluate harmonization performance for YOUR data
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import seaborn as sns

# ============================================================
# PFADE ANPASSEN
# ============================================================

# RAW DATA (vor Harmonization) - EINE große CSV
roi_raw_path = "/PFAD/ZU/IHRER/roi_data_BEFORE_harmonization.csv"  # ⭐ TODO: Anpassen!

# HARMONIZED DATA (nach Harmonization) - getrennte CSVs
roi_harm_hc_train_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/hc_train_roi_harmonized.csv"
roi_harm_hc_test_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/hc_test_roi_harmonized.csv"
roi_harm_pat_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/pat_roi_harmonized.csv"

# METADATA
metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/metadata_HARMONIZE_READY.csv"

# OUTPUT
output_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/evaluation_plots"
import os
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# SCHRITT 1: DATEN LADEN
# ============================================================

print("="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# Metadata laden
metadata = pd.read_csv(metadata_path)
print(f"Metadata shape: {metadata.shape}")
print(f"Metadata columns: {metadata.columns.tolist()}")

# Check ob Filename als Index oder Spalte
if 'Filename' in metadata.columns:
    metadata = metadata.set_index('Filename')
    print("✓ Set Filename as index")

print(f"\nMetadata index preview:")
print(metadata.index[:5])

# ROI RAW DATA laden (eine große CSV)
roi_raw_all = pd.read_csv(roi_raw_path)
print(f"\nRaw ROI data shape: {roi_raw_all.shape}")
print(f"Raw ROI columns (first 5): {roi_raw_all.columns[:5].tolist()}")

# Check ob Filename als Index oder Spalte
if 'Filename' in roi_raw_all.columns:
    roi_raw_all = roi_raw_all.set_index('Filename')
    print("✓ Set Filename as index for raw data")
elif roi_raw_all.index.name != 'Filename':
    # Wenn kein Filename, erster Column könnte es sein
    print(f"⚠️  Raw data index name: {roi_raw_all.index.name}")
    print(f"First few index values: {roi_raw_all.index[:5].tolist()}")

# HARMONIZED DATA laden
roi_harm_hc_train = pd.read_csv(roi_harm_hc_train_path, index_col=0)
roi_harm_hc_test = pd.read_csv(roi_harm_hc_test_path, index_col=0)
roi_harm_pat = pd.read_csv(roi_harm_pat_path, index_col=0)

print(f"\nHarmonized data shapes:")
print(f"  HC train: {roi_harm_hc_train.shape}")
print(f"  HC test:  {roi_harm_hc_test.shape}")
print(f"  Patients: {roi_harm_pat.shape}")

# Kombiniere harmonized data
roi_harmonized = pd.concat([roi_harm_hc_train, roi_harm_hc_test, roi_harm_pat])
print(f"  Combined harmonized: {roi_harmonized.shape}")

# ============================================================
# SCHRITT 2: SUBJECTS ALIGNIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 2: ALIGNING SUBJECTS")
print("="*70)

# Finde gemeinsame Subjects
subjects_in_harmonized = set(roi_harmonized.index)
subjects_in_metadata = set(metadata.index)
subjects_in_raw = set(roi_raw_all.index)

print(f"Subjects in harmonized data: {len(subjects_in_harmonized)}")
print(f"Subjects in metadata: {len(subjects_in_metadata)}")
print(f"Subjects in raw data: {len(subjects_in_raw)}")

# Gemeinsame Subjects über alle drei
common_subjects = subjects_in_harmonized.intersection(subjects_in_metadata).intersection(subjects_in_raw)
print(f"\n✓ Common subjects across all: {len(common_subjects)}")

if len(common_subjects) == 0:
    print("\n⚠️  ERROR: No common subjects found!")
    print("\nExample subject IDs from each dataset:")
    print(f"  Harmonized: {list(subjects_in_harmonized)[:5]}")
    print(f"  Metadata:   {list(subjects_in_metadata)[:5]}")
    print(f"  Raw:        {list(subjects_in_raw)[:5]}")
    raise ValueError("No common subjects - check if subject IDs match across files!")

# Filter zu gemeinsamen Subjects
common_subjects_sorted = sorted(common_subjects)

roi_raw = roi_raw_all.loc[common_subjects_sorted].copy()
roi_harmonized = roi_harmonized.loc[common_subjects_sorted].copy()
metadata = metadata.loc[common_subjects_sorted].copy()

print(f"\nFinal aligned shapes:")
print(f"  Raw ROIs:        {roi_raw.shape}")
print(f"  Harmonized ROIs: {roi_harmonized.shape}")
print(f"  Metadata:        {metadata.shape}")

# ============================================================
# SCHRITT 3: ROI SPALTEN ALIGNIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 3: ALIGNING ROI COLUMNS")
print("="*70)

common_rois = roi_raw.columns.intersection(roi_harmonized.columns)
print(f"Common ROIs: {len(common_rois)}/{len(roi_raw.columns)}")

if len(common_rois) < len(roi_raw.columns):
    n_dropped = len(roi_raw.columns) - len(common_rois)
    print(f"⚠️  Dropping {n_dropped} ROIs not in both datasets")
    dropped_rois = set(roi_raw.columns) - set(common_rois)
    print(f"Example dropped ROIs: {list(dropped_rois)[:5]}")

roi_raw = roi_raw[common_rois].copy()
roi_harmonized = roi_harmonized[common_rois].copy()

print(f"\nFinal ROI shapes:")
print(f"  Raw:        {roi_raw.shape}")
print(f"  Harmonized: {roi_harmonized.shape}")

# ============================================================
# SCHRITT 4: METADATA VORBEREITEN
# ============================================================

print("\n" + "="*70)
print("STEP 4: PREPARING METADATA FOR EVALUATION")
print("="*70)

print("Available metadata columns:", metadata.columns.tolist())

# Required columns check
required_cols = ['SITE', 'Age', 'Diagnosis']
missing = [col for col in required_cols if col not in metadata.columns]
if missing:
    raise ValueError(f"Missing required columns in metadata: {missing}")

print(f"\n✓ Required columns present: {required_cols}")

# Check für optionale Spalten
optional_cols = ['TIV', 'IQR', 'Sex_Male', 'Sex_M']
available_optional = [col for col in optional_cols if col in metadata.columns]
print(f"✓ Optional columns present: {available_optional}")

# Diagnosis distribution
print("\n📊 Diagnosis distribution:")
dx_dist = metadata['Diagnosis'].value_counts()
for dx, count in dx_dist.items():
    pct = 100 * count / len(metadata)
    print(f"  {dx:15s}: {count:4d} ({pct:.1f}%)")

# Site distribution
print("\n📊 Site distribution:")
site_dist = metadata['SITE'].value_counts()
print(f"Number of sites: {len(site_dist)}")
for site, count in site_dist.head(10).items():
    print(f"  {site:20s}: {count:4d}")
if len(site_dist) > 10:
    print(f"  ... and {len(site_dist) - 10} more sites")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def site_r2(data, covars, batch_col="SITE"):
    """Compute mean R² explained by site."""
    X = pd.get_dummies(covars[batch_col], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    
    r2s = []
    for i in range(data.shape[1]):
        y = data.iloc[:, i]
        try:
            res = sm.OLS(y, X).fit()
            r2s.append(res.rsquared)
        except:
            continue
    
    return np.mean(r2s) if r2s else 0.0

def site_classification_accuracy(data, covars, batch_col="SITE"):
    """Cross-validated site prediction accuracy."""
    X = StandardScaler().fit_transform(data)
    y = covars[batch_col].astype(str)
    
    # Remove sites with too few samples
    site_counts = y.value_counts()
    valid_sites = site_counts[site_counts >= 5].index
    mask = y.isin(valid_sites)
    
    if mask.sum() < 10:
        print("⚠️  Too few samples for site classification")
        return 0.0
    
    X = X[mask]
    y = y[mask]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        acc = cross_val_score(clf, X, y, cv=3).mean()
    except:
        acc = 0.0
    
    return acc

def diagnosis_classification_accuracy(data, covars, dx_col="Diagnosis"):
    """Cross-validated diagnosis prediction accuracy."""
    X = StandardScaler().fit_transform(data)
    y = covars[dx_col].astype(str)
    
    # Check if we have multiple diagnosis types
    if y.nunique() < 2:
        return 0.0
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        acc = cross_val_score(clf, X, y, cv=min(3, y.value_counts().min())).mean()
    except:
        acc = 0.0
    
    return acc

def mean_covariate_r2(data, covars, covar_name):
    """Mean ROI R² explained by a covariate."""
    if covar_name not in covars.columns:
        return None
    
    X = sm.add_constant(covars[covar_name])
    r2s = []
    for i in range(data.shape[1]):
        y = data.iloc[:, i]
        try:
            res = sm.OLS(y, X).fit()
            r2s.append(res.rsquared)
        except:
            continue
    return np.mean(r2s) if r2s else 0.0

def diagnosis_effect_cohen_d(data, covars):
    """Compute Cohen's d for each patient group vs HC."""
    results = {}
    
    dx_types = covars['Diagnosis'].unique()
    
    hc_mask = covars['Diagnosis'] == 'HC'
    if hc_mask.sum() == 0:
        return results
    
    hc_data = data[hc_mask]
    
    for dx in dx_types:
        if dx == 'HC':
            continue
        
        pat_mask = covars['Diagnosis'] == dx
        if pat_mask.sum() < 5:
            continue
        
        pat_data = data[pat_mask]
        
        m_hc = hc_data.mean(axis=0)
        s_hc = hc_data.std(axis=0)
        m_pat = pat_data.mean(axis=0)
        s_pat = pat_data.std(axis=0)
        
        d = (m_pat - m_hc) / np.sqrt((s_hc**2 + s_pat**2) / 2)
        results[f"{dx}_vs_HC"] = d.abs().mean()
    
    return results

# ============================================================
# QUANTITATIVE EVALUATION
# ============================================================

print("\n" + "="*70)
print("QUANTITATIVE EVALUATION")
print("="*70)

print("\n🎯 Computing metrics (this may take a few minutes)...")

# Site effects
print("1. Site effect reduction...")
r2_before = site_r2(roi_raw, metadata)
r2_after = site_r2(roi_harmonized, metadata)
acc_before = site_classification_accuracy(roi_raw, metadata)
acc_after = site_classification_accuracy(roi_harmonized, metadata)

# Age
print("2. Age preservation...")
age_r2_before = mean_covariate_r2(roi_raw, metadata, 'Age')
age_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'Age')

# TIV
tiv_r2_before = tiv_r2_after = None
if 'TIV' in metadata.columns:
    print("3. TIV preservation...")
    tiv_r2_before = mean_covariate_r2(roi_raw, metadata, 'TIV')
    tiv_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'TIV')

# IQR
iqr_r2_before = iqr_r2_after = None
if 'IQR' in metadata.columns:
    print("4. IQR preservation...")
    iqr_r2_before = mean_covariate_r2(roi_raw, metadata, 'IQR')
    iqr_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'IQR')

# Sex
sex_r2_before = sex_r2_after = None
sex_col = None
for col in ['Sex_Male', 'Sex_M']:
    if col in metadata.columns:
        sex_col = col
        break

if sex_col:
    print("5. Sex preservation...")
    sex_r2_before = mean_covariate_r2(roi_raw, metadata, sex_col)
    sex_r2_after = mean_covariate_r2(roi_harmonized, metadata, sex_col)

# Diagnosis
print("6. Diagnosis effects...")
diag_acc_before = diagnosis_classification_accuracy(roi_raw, metadata)
diag_acc_after = diagnosis_classification_accuracy(roi_harmonized, metadata)
dx_d_before = diagnosis_effect_cohen_d(roi_raw, metadata)
dx_d_after = diagnosis_effect_cohen_d(roi_harmonized, metadata)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "="*70)
print("📊 HARMONIZATION PERFORMANCE RESULTS")
print("="*70)

print("\n=== 🎯 SITE EFFECT REDUCTION (Goal: DECREASE) ===")
print(f"Mean Site R²:")
print(f"  Before: {r2_before:.4f}")
print(f"  After:  {r2_after:.4f}")
change_pct = 100*(r2_after - r2_before)/r2_before if r2_before > 0 else 0
print(f"  Change: {r2_after - r2_before:+.4f} ({change_pct:+.1f}%)")

print(f"\nSite Classification Accuracy:")
print(f"  Before: {acc_before:.3f}")
print(f"  After:  {acc_after:.3f}")
print(f"  Change: {acc_after - acc_before:+.3f}")

if r2_after < r2_before * 0.8:  # 20% reduction
    print("  ✅ GOOD: Site effects substantially reduced!")
elif r2_after < r2_before:
    print("  ✓ OK: Site effects reduced")
else:
    print("  ⚠️  WARNING: Site effects not reduced")

print("\n=== 🛡️  BIOLOGICAL PRESERVATION (Goal: MAINTAIN) ===")

def print_preservation(name, before, after):
    if before is None:
        return
    change = after - before
    change_pct = 100*change/before if before > 0 else 0
    print(f"\n{name} R²:")
    print(f"  Before: {before:.4f}")
    print(f"  After:  {after:.4f}")
    print(f"  Change: {change:+.4f} ({change_pct:+.1f}%)")
    
    if abs(change_pct) < 10:
        print(f"  ✅ GOOD: {name} effect well preserved")
    elif abs(change_pct) < 20:
        print(f"  ✓ OK: {name} effect reasonably preserved")
    else:
        print(f"  ⚠️  WARNING: {name} effect changed substantially")

print_preservation("Age", age_r2_before, age_r2_after)
print_preservation("TIV", tiv_r2_before, tiv_r2_after)
print_preservation("IQR", iqr_r2_before, iqr_r2_after)
print_preservation("Sex", sex_r2_before, sex_r2_after)

print(f"\nDiagnosis Classification Accuracy:")
print(f"  Before: {diag_acc_before:.3f}")
print(f"  After:  {diag_acc_after:.3f}")
print(f"  Change: {diag_acc_after - diag_acc_before:+.3f}")

if diag_acc_after >= diag_acc_before * 0.9:
    print("  ✅ GOOD: Diagnosis signal preserved")
else:
    print("  ⚠️  WARNING: Diagnosis signal reduced")

if dx_d_before:
    print(f"\nCohen's d (Effect Sizes):")
    for dx_type in sorted(dx_d_before.keys()):
        d_before = dx_d_before[dx_type]
        d_after = dx_d_after.get(dx_type, 0)
        change = d_after - d_before
        change_pct = 100*change/d_before if d_before > 0 else 0
        print(f"  {dx_type}:")
        print(f"    Before: {d_before:.3f}")
        print(f"    After:  {d_after:.3f} ({change_pct:+.1f}%)")

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

def plot_pca_umap(data_raw, data_har, covars, color_by="SITE", title_prefix=""):
    """Generate PCA and UMAP plots."""
    print(f"  Plotting {color_by}...", end=" ")
    
    if color_by not in covars.columns:
        print(f"SKIPPED (column not found)")
        return
    
    try:
        scaler = StandardScaler()
        Xb = scaler.fit_transform(data_raw)
        Xa = scaler.fit_transform(data_har)

        # PCA
        pca = PCA(n_components=2)
        coords_b = pca.fit_transform(Xb)
        coords_a = pca.fit_transform(Xa)

        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
        u_b = reducer.fit_transform(Xb)
        u_a = reducer.fit_transform(Xa)

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        var = covars[color_by]

        # Continuous vs categorical
        if pd.api.types.is_numeric_dtype(var) and var.nunique() > 10:
            # Continuous
            for ax, coords, title in zip(
                [axs[0,0], axs[0,1], axs[1,0], axs[1,1]],
                [coords_b, coords_a, u_b, u_a],
                ["Before (PCA)", "After (PCA)", "Before (UMAP)", "After (UMAP)"]
            ):
                scatter = ax.scatter(coords[:,0], coords[:,1], c=var, cmap="viridis", alpha=0.6, s=15)
                ax.set_title(f"{title_prefix}{title}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            # Categorical
            var_cat = var.astype("category")
            n_cats = len(var_cat.cat.categories)
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_cats, 20)))
            color_map = dict(zip(var_cat.cat.categories, colors))
            
            for ax, coords, title in zip(
                [axs[0,0], axs[0,1], axs[1,0], axs[1,1]],
                [coords_b, coords_a, u_b, u_a],
                ["Before (PCA)", "After (PCA)", "Before (UMAP)", "After (UMAP)"]
            ):
                for cat in var_cat.cat.categories:
                    mask = var == cat
                    if mask.sum() > 0:
                        ax.scatter(coords[mask,0], coords[mask,1], 
                                  c=[color_map[cat]], label=str(cat)[:20], alpha=0.6, s=15)
                ax.set_title(f"{title_prefix}{title}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Legend
            if n_cats <= 20:
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cat], markersize=8, label=str(cat)[:20])
                          for cat in var_cat.cat.categories]
                fig.legend(handles=handles, title=color_by, 
                          loc='lower center', ncol=min(5, n_cats),
                          bbox_to_anchor=(0.5, -0.05), fontsize=8)

        plt.tight_layout()
        save_path = f"{output_dir}/{title_prefix.strip()}_{color_by}_pca_umap.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓")
    except Exception as e:
        print(f"ERROR: {str(e)}")

# Create plots
plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by="SITE")
plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by="Age")
plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by="Diagnosis")

if 'TIV' in metadata.columns:
    plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by="TIV")

if 'IQR' in metadata.columns:
    plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by="IQR")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\n📁 Plots saved to: {output_dir}/")
print(f"\n🎯 Summary:")
print(f"   Site effects: {'✅ Reduced' if r2_after < r2_before else '⚠️  Not reduced'}")
print(f"   Age preserved: {'✅ Yes' if abs(age_r2_after - age_r2_before) < 0.1*age_r2_before else '⚠️  Changed'}")
print(f"   Diagnosis preserved: {'✅ Yes' if diag_acc_after >= diag_acc_before * 0.9 else '⚠️  Reduced'}")