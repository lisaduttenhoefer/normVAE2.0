#!/usr/bin/env python3
"""
UMAP visualization of brain MRI data
Colored by Dataset, Age, Sex, and Diagnosis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import StandardScaler

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# MRI features
mri_df = pd.read_csv('/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/QC/CAT12_results_final.csv')
print(f"  MRI data: {mri_df.shape}")
print(f"  MRI key column: 'Filename' - {mri_df['Filename'].nunique()} unique")

# Metadata
meta_df = pd.read_csv('/net/data.isilon/ag-cherrmann/lduttenhoefer/project/CAT12_newvals/metadata/metadata_CVAE.csv')
print(f"  Metadata: {meta_df.shape}")
print(f"  Meta key column: 'Filename' - {meta_df['Filename'].nunique()} unique")

# Merge
data = mri_df.merge(
    meta_df[['Filename', 'Dataset', 'Age', 'Sex', 'Diagnosis']], 
    on='Filename', 
    how='inner',
    suffixes=('', '_meta')  # In case Dataset appears in both
)
print(f"  Merged: {data.shape}")
print(f"  Merged subjects: {data['Filename'].nunique()}")

# Use Dataset from metadata if there's a conflict
if 'Dataset_meta' in data.columns:
    print("  [INFO] Using Dataset from metadata file")
    data['Dataset'] = data['Dataset_meta']
    data.drop('Dataset_meta', axis=1, inplace=True)

# Check we have the metadata columns
print("\n  Available metadata columns:")
for col in ['Dataset', 'Age', 'Sex', 'Diagnosis']:
    if col in data.columns:
        print(f"    ✓ {col}")
    else:
        print(f"    ✗ {col} MISSING!")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("\nPreparing features...")

# Get all numeric ROI columns
# Looking for columns with atlas prefixes or volume types
roi_cols = [col for col in data.columns if any(x in col for x in [
    'DK40', 'Neurom', 'AAL', 'SUIT',  # Atlas names
    '_G_', '_T_', '_Vgm', '_Vwm', '_Vcsf'  # Volume types
])]

print(f"  Found {len(roi_cols)} ROI features")

if len(roi_cols) < 10:
    print(f"\n[WARNING] Only found {len(roi_cols)} ROI columns!")
    print("  Trying alternative approach...")
    # Alternative: exclude known metadata columns
    exclude_cols = ['Filename', 'Dataset', 'Diagnosis', 'Age', 'Sex', 'TIV', 
                   'IQR', 'NCR', 'ICR', 'res_RMS', 'GM_vol', 'WM_vol', 
                   'CSF_vol', 'WMH_vol', 'SITE']
    roi_cols = [col for col in data.columns 
                if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]
    print(f"  Found {len(roi_cols)} numeric features")

# Extract features
X = data[roi_cols].values

# Handle missing values
print(f"  Missing values: {np.isnan(X).sum()} / {X.size} ({np.isnan(X).sum()/X.size*100:.2f}%)")
X = np.nan_to_num(X, nan=0.0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Feature matrix: {X_scaled.shape}")

# ============================================================================
# RUN UMAP
# ============================================================================

print("\nRunning UMAP (this may take a few minutes)...")

reducer = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42,
    n_jobs=-1,
    verbose=True
)

embedding = reducer.fit_transform(X_scaled)

print(f"  UMAP embedding: {embedding.shape}")

#===========================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

# ========== Plot 1: Dataset ==========
ax = axes[0]

datasets = data['Dataset'].values
unique_datasets = np.unique(datasets)
n_datasets = len(unique_datasets)

# Use a good colormap
if n_datasets <= 10:
    colors_dataset = plt.cm.tab10(np.linspace(0, 1, n_datasets))
else:
    colors_dataset = plt.cm.tab20(np.linspace(0, 1, n_datasets))

for i, dataset in enumerate(unique_datasets):
    mask = datasets == dataset
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=[colors_dataset[i]],
        label=f"{dataset} (n={mask.sum()})",
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax.set_title('Colored by Dataset', fontsize=14, fontweight='bold', pad=15)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2 if n_datasets > 10 else 1)

# Remove grid and spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# ========== Plot 2: Age ==========
ax = axes[1]

ages = data['Age'].values
ages_clean = np.where(np.isnan(ages), np.nanmedian(ages), ages)

scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=ages_clean,
    cmap='viridis',
    alpha=0.6,
    s=15,
    edgecolors='none'
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Age (years)', fontsize=10, fontweight='bold')

ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax.set_title('Colored by Age', fontsize=14, fontweight='bold', pad=15)

# Remove grid and spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# ========== Plot 3: Sex ==========
ax = axes[2]

sex = data['Sex'].values

# Plot females
mask_f = sex == 0
if mask_f.sum() > 0:
    ax.scatter(
        embedding[mask_f, 0],
        embedding[mask_f, 1],
        c='#FF69B4',
        label=f'Female (n={mask_f.sum()})',
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

# Plot males
mask_m = sex == 1
if mask_m.sum() > 0:
    ax.scatter(
        embedding[mask_m, 0],
        embedding[mask_m, 1],
        c='#1E90FF',
        label=f'Male (n={mask_m.sum()})',
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

# Plot unknown
mask_unknown = ~np.isin(sex, [0, 1])
if mask_unknown.sum() > 0:
    ax.scatter(
        embedding[mask_unknown, 0],
        embedding[mask_unknown, 1],
        c='gray',
        label=f'Unknown (n={mask_unknown.sum()})',
        alpha=0.3,
        s=15,
        edgecolors='none'
    )

ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax.set_title('Colored by Sex', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)

# Remove grid and spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# ========== Plot 4: Diagnosis ==========
ax = axes[3]

diagnoses = data['Diagnosis'].values
unique_diag = np.unique(diagnoses)

# Custom colors for diagnoses
diag_colors = {
    'HC': '#125E8A',
    'SSD': '#3E885B',
    'MDD': '#BEDCFE',
    'CAT': '#2F4B26',
    'CAT-SSD': '#A67DB8',
    'CAT-MDD': '#160C28'
}

for diag in unique_diag:
    mask = diagnoses == diag
    color = diag_colors.get(diag, '#808080')  # Gray for unknown
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=color,
        label=f"{diag} (n={mask.sum()})",
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax.set_title('Colored by Diagnosis', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)

# Remove grid and spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# ============================================================================
# SAVE
# ============================================================================

plt.suptitle('Brain MRI Data - UMAP Visualization', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_file = '/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/UMAP_analysis/brain_mri_umap_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal subjects: {len(data)}")

print(f"\nDatasets:")
for dataset in unique_datasets:
    n = (datasets == dataset).sum()
    print(f"  {dataset:20s}: {n:5d} subjects ({n/len(data)*100:5.1f}%)")

print(f"\nAge:")
print(f"  Mean:  {np.nanmean(ages):5.1f} years")
print(f"  Std:   {np.nanstd(ages):5.1f} years")
print(f"  Range: {np.nanmin(ages):5.1f} - {np.nanmax(ages):.1f} years")
print(f"  Missing: {np.isnan(ages).sum()} subjects")

print(f"\nSex:")
n_female = (sex == 0).sum()
n_male = (sex == 1).sum()
n_unknown = len(sex) - n_female - n_male
print(f"  Female:  {n_female:5d} ({n_female/len(sex)*100:5.1f}%)")
print(f"  Male:    {n_male:5d} ({n_male/len(sex)*100:5.1f}%)")
if n_unknown > 0:
    print(f"  Unknown: {n_unknown:5d} ({n_unknown/len(sex)*100:5.1f}%)")

print(f"\nDiagnoses:")
for diag in unique_diag:
    n = (diagnoses == diag).sum()
    print(f"  {diag:20s}: {n:5d} subjects ({n/len(diagnoses)*100:5.1f}%)")

print("\n" + "="*70)
print("✓ Analysis complete!")
print("="*70)