import pandas as pd
import numpy as np

csv_path = "/net/data.isilon/ag-cherrmann/lduttenhoefer/project/VAE_model/data_training/CAT12_results_NORMALIZED_columnwise_HC.csv"
df = pd.read_csv(csv_path)

roi_cols = [c for c in df.columns if c.startswith(('Vgm_', 'G_', 'T_'))]

print("="*80)
print("CSV HEALTH CHECK")
print("="*80)
print(f"Total rows: {len(df)}")
print(f"Total ROI columns: {len(roi_cols)}")
print(f"\nData Quality:")
print(f"  NaN count: {df[roi_cols].isna().sum().sum()}")
print(f"  Inf count: {np.isinf(df[roi_cols]).sum().sum()}")
print(f"\nValue ranges:")
print(f"  Min: {df[roi_cols].min().min():.3f}")
print(f"  Max: {df[roi_cols].max().max():.3f}")
print(f"  Mean: {df[roi_cols].mean().mean():.6f}")
print(f"  Std: {df[roi_cols].std().mean():.6f}")
print(f"\nFirst subject, first 10 values:")
print(df[roi_cols].iloc[0, :10].values)