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

# RAW DATA (vor Harmonization)
roi_raw_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv"  

# HARMONIZED DATA (nach Harmonization) - NEUE DATEIEN
roi_harm_hc_train_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/hc_train_roi_harmonized.csv"
# NEU: Kombinierte Application Datei (HC Test + Patienten ohne NU)
roi_harm_app_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_results/application_roi_harmonized_noNU.csv" 

# METADATA (Muss alle Probanden enthalten, wird später gefiltert)
metadata_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/combat_metadata/metadata_for_harmonizing_all.csv"

# AUSGESCHLOSSENE SITES (Muss mit dem Hauptskript übereinstimmen)
EXCLUDED_SITES = ['NSS', 'EPSY', 'NU'] 

# OUTPUT
output_dir = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/combat_neuro/evaluation_plots"
import os
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# SCHRITT 1: DATEN LADEN & VORFILTERN
# ============================================================

print("="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# Metadata laden
metadata = pd.read_csv(metadata_path)
if 'Filename' in metadata.columns:
    metadata = metadata.set_index('Filename')

# ⭐ FIX: Filtern der Metadaten, um sie mit den ROI-Daten abzugleichen
metadata = metadata[~metadata['Dataset'].isin(EXCLUDED_SITES)].copy()
print(f"Metadata shape (nach Ausschluss von {EXCLUDED_SITES}): {metadata.shape}")

# ROI RAW DATA laden
roi_raw_all = pd.read_csv(roi_raw_path)
if 'Filename' in roi_raw_all.columns:
    roi_raw_all = roi_raw_all.set_index('Filename')
elif roi_raw_all.index.name != 'Filename':
    # Versuche, die erste Spalte als Index zu setzen, falls 'Filename' fehlt
    if roi_raw_all.columns[0] == 'Filename':
        roi_raw_all = roi_raw_all.set_index('Filename')


# HARMONIZED DATA laden
roi_harm_hc_train = pd.read_csv(roi_harm_hc_train_path, index_col='Filename') # Korrekter Index-Name
# ⭐ FIX: Kombinierte Application Datei laden
roi_harm_app = pd.read_csv(roi_harm_app_path, index_col='Filename') 

# ⭐ FIX: Kombiniere harmonized data (Train + Application)
roi_harmonized = pd.concat([roi_harm_hc_train, roi_harm_app])
print(f"  Combined harmonized: {roi_harmonized.shape}")

# ============================================================
# SCHRITT 2: SUBJECTS ALIGNIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 2: ALIGNING SUBJECTS")
print("="*70)

subjects_in_harmonized = set(roi_harmonized.index)
subjects_in_metadata = set(metadata.index)
subjects_in_raw = set(roi_raw_all.index)

# Gemeinsame Subjects über alle drei
common_subjects = subjects_in_harmonized.intersection(subjects_in_metadata).intersection(subjects_in_raw)
print(f"✓ Common subjects across all: {len(common_subjects)}")

if len(common_subjects) == 0:
    raise ValueError("No common subjects - check if subject IDs match across files!")

# Filter zu gemeinsamen Subjects
common_subjects_sorted = sorted(common_subjects)

roi_raw = roi_raw_all.loc[common_subjects_sorted].copy()
roi_harmonized = roi_harmonized.loc[common_subjects_sorted].copy()
metadata = metadata.loc[common_subjects_sorted].copy()

# ============================================================
# SCHRITT 3: ROI SPALTEN ALIGNIEREN
# ============================================================

print("\n" + "="*70)
print("STEP 3: ALIGNING ROI COLUMNS")
print("="*70)

# Stellen Sie sicher, dass keine nicht-numerischen Spalten in den ROIs sind
non_roi_cols_in_raw = [col for col in roi_raw.columns if roi_raw[col].dtype == object]
if non_roi_cols_in_raw:
    roi_raw = roi_raw.drop(columns=non_roi_cols_in_raw)
    print(f"Dropped non-numeric columns in raw data: {non_roi_cols_in_raw[:5]}")

common_rois = roi_raw.columns.intersection(roi_harmonized.columns)
roi_raw = roi_raw[common_rois].copy()
roi_harmonized = roi_harmonized[common_rois].copy()

# ============================================================
# SCHRITT 4: METADATA VORBEREITEN
# ============================================================

print("\n" + "="*70)
print("STEP 4: PREPARING METADATA FOR EVALUATION")
print("="*70)

# Identifiziere Sex-Spalte
sex_col = None
for col in ['Sex_Male', 'Sex_M', 'Sex']:
    if col in metadata.columns:
        sex_col = col
        break

# Diagnosis distribution
dx_dist = metadata['Diagnosis'].value_counts()
print(f"  HCs: {dx_dist.get('HC', 0)}, Patients: {len(metadata) - dx_dist.get('HC', 0)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def site_r2(data, covars, batch_col="SITE"):
    """Compute mean R² explained by site."""
    X = pd.get_dummies(covars[batch_col], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    
    r2s = []
    # Stellen Sie sicher, dass X und data die gleiche Zeilenanzahl haben
    if X.shape[0] != data.shape[0]:
        raise ValueError("X and data must have the same number of rows for OLS.")
        
    for i in range(data.shape[1]):
        y = data.iloc[:, i]
        try:
            # Drop NaN-Werte, um Probleme mit OLS zu vermeiden
            valid_mask = ~y.isna()
            res = sm.OLS(y[valid_mask], X[valid_mask]).fit()
            r2s.append(res.rsquared)
        except Exception as e:
            # Oft durch PerfectSeparation oder fehlende Varianz
            # print(f"Warning: OLS failed for column {data.columns[i]}: {e}")
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
    
    # Remove HCs to focus only on classification between patient groups
    # FÜR DIESE PRÜFUNG: Nur Patienten vergleichen
    patient_mask = y != 'HC'
    X = X[patient_mask]
    y = y[patient_mask]
    
    if y.nunique() < 2 or len(y) < 10:
        return 0.0
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        acc = cross_val_score(clf, X, y, cv=min(3, y.value_counts().min())).mean()
    except:
        acc = 0.0
    
    return acc

def mean_covariate_r2(data, covars, covar_name):
    """Mean ROI R² explained by a covariate."""
    if covar_name not in covars.columns or covars[covar_name].nunique() <= 1:
        return None
    
    # Stellen Sie sicher, dass keine NaNs in der Kovariate sind
    valid_mask = ~covars[covar_name].isna()
    X = sm.add_constant(covars.loc[valid_mask, covar_name])
    data_valid = data.loc[valid_mask]

    r2s = []
    for i in range(data_valid.shape[1]):
        y = data_valid.iloc[:, i]
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

# Site effects
r2_before = site_r2(roi_raw, metadata)
r2_after = site_r2(roi_harmonized, metadata)
acc_before = site_classification_accuracy(roi_raw, metadata)
acc_after = site_classification_accuracy(roi_harmonized, metadata)

# Age
age_r2_before = mean_covariate_r2(roi_raw, metadata, 'Age')
age_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'Age')

# TIV
tiv_r2_before = tiv_r2_after = None
if 'TIV' in metadata.columns:
    tiv_r2_before = mean_covariate_r2(roi_raw, metadata, 'TIV')
    tiv_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'TIV')

# IQR
iqr_r2_before = iqr_r2_after = None
if 'IQR' in metadata.columns:
    iqr_r2_before = mean_covariate_r2(roi_raw, metadata, 'IQR')
    iqr_r2_after = mean_covariate_r2(roi_harmonized, metadata, 'IQR')

# Sex
sex_r2_before = sex_r2_after = None
if sex_col:
    sex_r2_before = mean_covariate_r2(roi_raw, metadata, sex_col)
    sex_r2_after = mean_covariate_r2(roi_harmonized, metadata, sex_col)

# Diagnosis
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

# ... (Drucklogik wie im Originalskript) ...
print("\n=== 🎯 SITE EFFECT REDUCTION (Goal: DECREASE) ===")
# ... (Print Site R² and Accuracy) ...
def print_site_reduction(r2_before, r2_after, acc_before, acc_after):
    print(f"Mean Site R²:")
    print(f"  Before: {r2_before:.4f}")
    print(f"  After:  {r2_after:.4f}")
    change_pct = 100*(r2_after - r2_before)/r2_before if r2_before > 0 else 0
    print(f"  Change: {r2_after - r2_before:+.4f} ({change_pct:+.1f}%)")

    print(f"\nSite Classification Accuracy:")
    print(f"  Before: {acc_before:.3f}")
    print(f"  After:  {acc_after:.3f}")
    
    if r2_after < r2_before * 0.8:
        print("  ✅ GOOD: Site effects substantially reduced!")
    else:
        print("  ✓ OK: Site effects reduced") if r2_after < r2_before else print("  ⚠️  WARNING: Site effects not reduced")

print_site_reduction(r2_before, r2_after, acc_before, acc_after)

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
    else:
        print("  ✓ OK: {name} effect reasonably preserved") if abs(change_pct) < 20 else print(f"  ⚠️  WARNING: {name} effect changed substantially")

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
        print(f"  {dx_type}: Before: {d_before:.3f} | After: {d_after:.3f} ({change_pct:+.1f}%)")


# ============================================================
# VISUALIZATIONS
# ============================================================

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
                                  c=[color_map.get(cat, 'gray')], label=str(cat)[:20], alpha=0.6, s=15)
                ax.set_title(f"{title_prefix}{title}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Legend
            if n_cats <= 20:
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map.get(cat, 'gray'), markersize=8, label=str(cat)[:20])
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

if sex_col:
    plot_pca_umap(roi_raw, roi_harmonized, metadata, color_by=sex_col)


print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)