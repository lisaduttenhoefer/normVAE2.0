
import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import umap

from module.data_processing_hc import load_mri_data_2D_conditional
from utils.logging_utils import log_and_print_test


def match_unknown_dataset_by_iqr(
    dataset_name: str,
    dataset_iqr: float,
    known_datasets: List[str],
    training_metadata: pd.DataFrame,
    method: str = "closest"
) -> str:
    """
    Match an unknown dataset to a known dataset based on IQR similarity.
    
    Args:
        dataset_name: Name of unknown dataset
        dataset_iqr: IQR value of the subject
        known_datasets: List of dataset names from training
        training_metadata: Training metadata with Dataset and IQR columns
        method: "closest" or "median"
            - "closest": Match to dataset with closest IQR
            - "median": Match to dataset whose median IQR is closest
    
    Returns:
        matched_dataset: Name of matched known dataset
    """
    
    # Calculate median IQR for each known dataset
    dataset_iqr_stats = {}
    for dataset in known_datasets:
        dataset_mask = training_metadata['Dataset'] == dataset
        if dataset_mask.sum() > 0:
            dataset_iqrs = training_metadata.loc[dataset_mask, 'IQR'].values
            dataset_iqr_stats[dataset] = {
                'median': np.median(dataset_iqrs),
                'mean': np.mean(dataset_iqrs),
                'std': np.std(dataset_iqrs),
                'min': np.min(dataset_iqrs),
                'max': np.max(dataset_iqrs)
            }
    
    if not dataset_iqr_stats:
        # Fallback: return first known dataset
        return known_datasets[0]
    
    # Find closest match
    if method == "median":
        # Match to dataset with closest median IQR
        distances = {
            ds: abs(stats['median'] - dataset_iqr)
            for ds, stats in dataset_iqr_stats.items()
        }
    elif method == "closest":
        # Match to dataset where IQR falls within or closest to range
        distances = {}
        for ds, stats in dataset_iqr_stats.items():
            if stats['min'] <= dataset_iqr <= stats['max']:
                # IQR within range - use distance to median
                distances[ds] = abs(stats['median'] - dataset_iqr)
            else:
                # IQR outside range - use distance to nearest boundary
                if dataset_iqr < stats['min']:
                    distances[ds] = stats['min'] - dataset_iqr
                else:
                    distances[ds] = dataset_iqr - stats['max']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get dataset with minimum distance
    matched_dataset = min(distances, key=distances.get)
    distance = distances[matched_dataset]
    
    return matched_dataset, distance, dataset_iqr_stats[matched_dataset]


def create_conditions_tensor_from_metadata(
    metadata_df: pd.DataFrame,
    conditioning_info: dict,
    training_metadata: pd.DataFrame = None,
    match_unknown_datasets: bool = True
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Create conditions tensor for CVAE from metadata using training scalers.
    
    NOW WITH IQR-BASED DATASET MATCHING FOR UNKNOWN DATASETS!
    
    Args:
        metadata_df: DataFrame with Age, Sex, IQR, Dataset columns
        conditioning_info: Dict with scalers and categories from training
        training_metadata: Training metadata (needed for IQR-based matching)
        match_unknown_datasets: If True, match unknown datasets by IQR
    
    Returns:
        conditions_tensor: Tensor [n_samples, condition_dim]
        dataset_mapping_df: DataFrame with original and matched datasets
    """
    
    n_samples = len(metadata_df)
    dataset_categories = conditioning_info['dataset_categories']
    
    log_and_print_test(f"\n{'='*80}")
    log_and_print_test("CREATING CONDITIONS TENSOR WITH DATASET MATCHING")
    log_and_print_test(f"{'='*80}")
    
    # ========== AGE (normalized) ==========
    age_values = metadata_df['Age'].values.reshape(-1, 1)
    age_normalized = (age_values - conditioning_info['age_scaler_mean']) / conditioning_info['age_scaler_scale']
    age_normalized = age_normalized.flatten()
    
    # ========== SEX (binary) ==========
    if metadata_df['Sex'].dtype == 'object':
        sex_values = (metadata_df['Sex'].str.upper() == 'M').astype(float).values
    else:
        sex_values = metadata_df['Sex'].values.astype(float)
    
    # ========== IQR (normalized) ==========
    iqr_values = metadata_df['IQR'].values.reshape(-1, 1)
    iqr_normalized = (iqr_values - conditioning_info['iqr_scaler_mean']) / conditioning_info['iqr_scaler_scale']
    iqr_normalized = iqr_normalized.flatten()
    
    # ========== DATASET (one-hot with IQR-based matching) ==========
    dataset_onehot = np.zeros((n_samples, len(dataset_categories)))
    
    unknown_datasets = {}
    dataset_mapping = []
    
    for i, (dataset, iqr) in enumerate(zip(metadata_df['Dataset'], metadata_df['IQR'])):
        if dataset in dataset_categories:
            # Known dataset - use directly
            idx = dataset_categories.index(dataset)
            dataset_onehot[i, idx] = 1.0
            dataset_mapping.append({
                'Original_Dataset': dataset,
                'Matched_Dataset': dataset,
                'Match_Type': 'exact',
                'IQR': iqr,
                'Match_Distance': 0.0
            })
        else:
            # Unknown dataset
            if dataset not in unknown_datasets:
                unknown_datasets[dataset] = []
            unknown_datasets[dataset].append((i, iqr))
    
    # ========== MATCH UNKNOWN DATASETS ==========
    if unknown_datasets and match_unknown_datasets:
        if training_metadata is None:
            log_and_print_test("⚠️  WARNING: No training metadata provided - cannot match unknown datasets!")
            log_and_print_test("    Unknown datasets will use zero vector (average dataset)")
            
            for dataset, samples in unknown_datasets.items():
                for i, iqr in samples:
                    dataset_mapping.append({
                        'Original_Dataset': dataset,
                        'Matched_Dataset': 'UNKNOWN',
                        'Match_Type': 'zero_vector',
                        'IQR': iqr,
                        'Match_Distance': np.nan
                    })
        else:
            log_and_print_test(f"\n{'='*40}")
            log_and_print_test("IQR-BASED DATASET MATCHING")
            log_and_print_test(f"{'='*40}")
            log_and_print_test(f"Known datasets: {dataset_categories}")
            log_and_print_test(f"Unknown datasets found: {list(unknown_datasets.keys())}")
            
            for dataset, samples in unknown_datasets.items():
                log_and_print_test(f"\n{dataset}: {len(samples)} subjects")
                
                # Get IQRs for this unknown dataset
                dataset_iqrs = [iqr for _, iqr in samples]
                median_iqr = np.median(dataset_iqrs)
                
                log_and_print_test(f"  IQR range: [{min(dataset_iqrs):.3f}, {max(dataset_iqrs):.3f}]")
                log_and_print_test(f"  IQR median: {median_iqr:.3f}")
                
                # Match to known dataset
                matched_dataset, distance, matched_stats = match_unknown_dataset_by_iqr(
                    dataset_name=dataset,
                    dataset_iqr=median_iqr,
                    known_datasets=dataset_categories,
                    training_metadata=training_metadata,
                    method="median"
                )
                
                log_and_print_test(f"  → Matched to: {matched_dataset}")
                log_and_print_test(f"    {matched_dataset} IQR: {matched_stats['median']:.3f} ± {matched_stats['std']:.3f}")
                log_and_print_test(f"    Distance: {distance:.3f}")
                
                # Apply matched dataset to all samples
                matched_idx = dataset_categories.index(matched_dataset)
                for i, iqr in samples:
                    dataset_onehot[i, matched_idx] = 1.0
                    dataset_mapping.append({
                        'Original_Dataset': dataset,
                        'Matched_Dataset': matched_dataset,
                        'Match_Type': 'iqr_matched',
                        'IQR': iqr,
                        'Match_Distance': distance,
                        'Matched_IQR_Median': matched_stats['median'],
                        'Matched_IQR_Std': matched_stats['std']
                    })
            
            log_and_print_test(f"\n✓ Matched {sum(len(s) for s in unknown_datasets.values())} subjects from unknown datasets")
    
    elif unknown_datasets and not match_unknown_datasets:
        log_and_print_test(f"⚠️  WARNING: Found unknown datasets but matching disabled!")
        log_and_print_test(f"    Unknown: {list(unknown_datasets.keys())}")
        log_and_print_test(f"    These will use zero vector (average dataset)")
        
        for dataset, samples in unknown_datasets.items():
            for i, iqr in samples:
                dataset_mapping.append({
                    'Original_Dataset': dataset,
                    'Matched_Dataset': 'UNKNOWN',
                    'Match_Type': 'zero_vector',
                    'IQR': iqr,
                    'Match_Distance': np.nan
                })
    
    # ========== COMBINE ALL ==========
    conditions = np.column_stack([
        age_normalized,
        sex_values,
        iqr_normalized,
        dataset_onehot
    ])
    
    conditions_tensor = torch.FloatTensor(conditions)
    dataset_mapping_df = pd.DataFrame(dataset_mapping)
    
    log_and_print_test(f"\n✓ Created conditions tensor: {conditions_tensor.shape}")
    log_and_print_test(f"  - Age (normalized): 1")
    log_and_print_test(f"  - Sex (binary): 1")
    log_and_print_test(f"  - IQR (normalized): 1")
    log_and_print_test(f"  - Dataset (one-hot): {len(dataset_categories)}")
    
    return conditions_tensor, dataset_mapping_df


def load_and_normalize_test_data(
    model_dir: str,
    raw_mri_csv: str,
    test_metadata: pd.DataFrame,
    atlas_name: list,
    volume_type: list
):
    """
    Load raw test data and apply TRAINING normalization.
    
    Args:
        model_dir: Path to training output directory
        raw_mri_csv: Path to RAW MRI CSV
        test_metadata: Test metadata DataFrame
        atlas_name: Atlas names
        volume_type: Volume types
    
    Returns:
        subjects, data_overview, roi_names
    """
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LOADING AND NORMALIZING TEST DATA")
    log_and_print_test("="*80)
    
    # ========== 1. LOAD NORMALIZATION STATS ==========
    norm_stats_path = os.path.join(model_dir, "data", "normalization_stats.pkl")
    
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_stats_path}\n"
            "Please ensure the model was trained with the updated training script."
        )
    
    with open(norm_stats_path, 'rb') as f:
        normalization_stats = pickle.load(f)
    
    log_and_print_test(f"✓ Loaded training normalization stats")
    log_and_print_test(f"  Volume types: {list(normalization_stats.keys())}")
    
    # ========== 2. LOAD RAW MRI DATA ==========
    log_and_print_test(f"\nLoading raw MRI data from: {raw_mri_csv}")
    
    if not os.path.exists(raw_mri_csv):
        raise FileNotFoundError(f"Raw MRI CSV not found: {raw_mri_csv}")
    
    raw_mri_data = pd.read_csv(raw_mri_csv)
    log_and_print_test(f"✓ Loaded {len(raw_mri_data)} subjects from raw MRI file")
    
    # ========== 3. APPLY TRAINING NORMALIZATION ==========
    log_and_print_test("\nApplying TRAINING normalization to test data...")
    log_and_print_test("(Using HC statistics from training set)")
    
    normalized_test_data = raw_mri_data.copy()
    
    for vtype, stats in normalization_stats.items():
        columns = stats['columns']
        
        if not columns:
            continue
        
        log_and_print_test(f"\nNormalizing {vtype}: {len(columns)} features")
        
        # TIV normalization (if applicable)
        if vtype in ['Vgm', 'Vwm', 'Vcsf'] and 'TIV' in normalized_test_data.columns:
            log_and_print_test(f"  - Applying TIV normalization")
            tiv = normalized_test_data['TIV'].values.reshape(-1, 1)
            normalized_test_data[columns] = normalized_test_data[columns].div(tiv, axis=0)
        
        # Apply TRAINING stats (NOT test stats!)
        medians = stats['median']
        iqr = stats['iqr']
        
        normalized_test_data[columns] = (normalized_test_data[columns] - medians) / iqr
        
        log_and_print_test(f"  ✓ Applied training normalization")
        log_and_print_test(f"    Output range: [{normalized_test_data[columns].min().min():.3f}, {normalized_test_data[columns].max().max():.3f}]")
    
    log_and_print_test("\n✓ Test data normalized using TRAINING HC statistics")
    
    # ========== 4. SAVE TO TEMP FILES AND LOAD WITH EXISTING FUNCTION ==========
    log_and_print_test("\nLoading test subjects...")
    
    # Create temp directory
    temp_dir = os.path.join(model_dir, "temp_test_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_mri_path = os.path.join(temp_dir, "temp_test_normalized_mri.csv")
    temp_meta_path = os.path.join(temp_dir, "temp_test_metadata.csv")
    
    normalized_test_data.to_csv(temp_mri_path, index=False)
    test_metadata.to_csv(temp_meta_path, index=False)
    
    # Load with existing function
    subjects, data_overview, roi_names = load_mri_data_2D_conditional(
        normalized_csv_path=temp_mri_path,
        csv_paths=[temp_meta_path],
        diagnoses=None,  # Load ALL diagnoses
        atlas_name=atlas_name,
        volume_type=volume_type
    )
    
    log_and_print_test(f"✓ Loaded {len(subjects)} test subjects")
    log_and_print_test(f"✓ ROI features: {len(roi_names)}")
    
    # Cleanup temp files
    os.remove(temp_mri_path)
    os.remove(temp_meta_path)
    os.rmdir(temp_dir)
    
    return subjects, data_overview, roi_names


def analyze_volume_type_separately_cvae(
    vtype, bootstrap_models, clinical_data, conditions_tensor, annotations_df,
    roi_names, norm_diagnosis, device, base_save_dir, conditioning_info,
    mri_data_path, atlas_name, metadata_path, custom_colors
):
    """
    Analyze a specific volume type separately (CVAE version).
    
    This is a CVAE-adapted version of analyze_volume_type_separately.
    """
    
    from utils.dev_scores_utils_CVAE import (
        calculate_deviations_cvae,
        calculate_reconstruction_deviation_cvae,
        calculate_kl_divergence_deviation_cvae,
        calculate_latent_deviation_aguila_cvae,
        calculate_combined_deviation,
        compute_hc_latent_stats_cvae,
    )
    from utils.dev_scores_utils import (
        plot_all_deviation_metrics_errorbar,
        analyze_regional_deviations,
    )
    
    log_and_print_test(f"\n{'='*80}")
    log_and_print_test(f"ANALYZING VOLUME TYPE: {vtype} (CVAE)")
    log_and_print_test(f"{'='*80}\n")
    
    # Create subdirectory for this volume type
    vtype_save_dir = os.path.join(base_save_dir, f"volume_type_{vtype}")
    os.makedirs(vtype_save_dir, exist_ok=True)
    os.makedirs(f"{vtype_save_dir}/figures", exist_ok=True)
    os.makedirs(f"{vtype_save_dir}/figures/distributions", exist_ok=True)
    
    # Filter ROI names for this volume type
    vtype_roi_indices = [i for i, name in enumerate(roi_names) if name.startswith(f"{vtype}_")]
    vtype_roi_names = [roi_names[i] for i in vtype_roi_indices]
    
    if not vtype_roi_names:
        log_and_print_test(f"No ROIs found for volume type {vtype}, skipping...")
        return
    
    log_and_print_test(f"Found {len(vtype_roi_names)} ROIs for {vtype}")
    
    # Filter clinical data for this volume type
    vtype_data = clinical_data[:, vtype_roi_indices]
    log_and_print_test(f"Filtered data shape: {vtype_data.shape}")
    
    # ========== CALCULATE DEVIATIONS FOR THIS VOLUME TYPE ==========
    log_and_print_test(f"\nCalculating deviation scores for {vtype}...")
    
    # Bootstrap method
    results_df_vtype = calculate_deviations_cvae(
        normative_models=bootstrap_models,
        data_tensor=vtype_data,
        conditions_tensor=conditions_tensor,
        norm_diagnosis=norm_diagnosis,
        annotations_df=annotations_df,
        device=device,
        roi_names=vtype_roi_names
    )
    
    # Additional metrics (using baseline model)
    baseline_model = bootstrap_models[0]
    
    hc_mask = annotations_df['Diagnosis'] == norm_diagnosis
    hc_data_vtype = vtype_data[hc_mask]
    hc_conditions = conditions_tensor[hc_mask]
    
    # D_MSE
    deviation_recon = calculate_reconstruction_deviation_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        device=device
    )
    
    # D_KL
    deviation_kl = calculate_kl_divergence_deviation_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        device=device
    )
    
    # D_latent
    hc_latent_stats = compute_hc_latent_stats_cvae(
        model=baseline_model,
        hc_data=hc_data_vtype.numpy(),
        hc_conditions=hc_conditions.numpy(),
        device=device
    )
    
    deviation_latent, _ = calculate_latent_deviation_aguila_cvae(
        model=baseline_model,
        data=vtype_data.numpy(),
        conditions=conditions_tensor.numpy(),
        hc_latent_stats=hc_latent_stats,
        device=device
    )
    
    # D_combined
    deviation_combined = calculate_combined_deviation(
        recon_dev=deviation_recon,
        kl_dev=deviation_kl
    )
    
    # Add to results
    results_df_vtype['deviation_score_recon'] = deviation_recon
    results_df_vtype['deviation_score_kl'] = deviation_kl
    results_df_vtype['deviation_score_latent_aguila'] = deviation_latent
    results_df_vtype['deviation_score_combined'] = deviation_combined
    
    log_and_print_test(f"✓ Computed all deviation metrics for {vtype}")
    
    # ========== SAVE RESULTS ==========
    results_file = os.path.join(vtype_save_dir, f"deviation_scores_{vtype}.csv")
    results_df_vtype.to_csv(results_file, index=False)
    log_and_print_test(f"Saved: {results_file}")
    
    # ========== PLOTS ==========
    plot_all_deviation_metrics_errorbar(
        results_df=results_df_vtype,
        save_dir=vtype_save_dir,
        norm_diagnosis=norm_diagnosis,
        custom_colors=custom_colors,
        name=f"{vtype}_Analysis"
    )
    
    # ========== REGIONAL ANALYSIS ==========
    log_and_print_test(f"\nPerforming regional analysis for {vtype}...")
    
    regional_results = analyze_regional_deviations(
        results_df=results_df_vtype,
        save_dir=vtype_save_dir,
        clinical_data_path=mri_data_path,
        volume_type=[vtype],
        atlas_name=atlas_name,
        roi_names=vtype_roi_names,
        norm_diagnosis=norm_diagnosis,
        name=f"{vtype}_analysis",
        add_catatonia_subgroups=False,
        metadata_path=metadata_path,
        merge_CAT_groups=True
    )
    
    if regional_results is not None and not regional_results.empty:
        regional_file = os.path.join(vtype_save_dir, f"regional_effect_sizes_{vtype}.csv")
        regional_results.to_csv(regional_file, index=False)
        log_and_print_test(f"Saved regional results: {regional_file}")
    
    log_and_print_test(f"✓ Completed analysis for {vtype}")


def analyze_latent_space_quality_cvae(
    model, data_tensor, conditions_tensor, annotations_df, 
    save_dir, device='cuda'
):
    """
    Analyze latent space quality with UMAP visualizations (CVAE version).
    Similar to post-training analysis but for test data.
    """
    
    log_and_print_test("\n" + "="*80)
    log_and_print_test("LATENT SPACE QUALITY ANALYSIS")
    log_and_print_test("="*80)
    
    os.makedirs(f"{save_dir}/latent_analysis", exist_ok=True)
    
    # ========== ENCODE DATA TO LATENT SPACE ==========
    model.eval()
    with torch.no_grad():
        data_dev = data_tensor.to(device)
        cond_dev = conditions_tensor.to(device)
        _, mu, _ = model(data_dev, cond_dev)
        latents = mu.cpu().numpy()
    
    log_and_print_test(f"Encoded {len(latents)} samples to latent space (dim={latents.shape[1]})")
    
    # ========== COMPUTE UMAP ==========
    log_and_print_test("Computing UMAP projection...")
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        n_jobs=1
    )
    
    umap_embeddings = reducer.fit_transform(latents)
    log_and_print_test(f"UMAP embedding shape: {umap_embeddings.shape}")
    
    # ========== SILHOUETTE SCORE (Dataset separation) ==========
    silhouette = np.nan
    if 'Dataset' in annotations_df.columns:
        unique_datasets = annotations_df['Dataset'].nunique()
        if unique_datasets > 1:
            try:
                dataset_labels = pd.Categorical(annotations_df['Dataset']).codes
                silhouette = silhouette_score(latents, dataset_labels)
                log_and_print_test(f"\nSilhouette Score (Dataset separation): {silhouette:.3f}")
                
                if silhouette < 0.2:
                    log_and_print_test("✓ EXCELLENT! Datasets well mixed (site effects removed)")
                elif silhouette < 0.4:
                    log_and_print_test("✓ GOOD! Datasets reasonably mixed")
                else:
                    log_and_print_test("⚠️  WARNING! Strong dataset clustering (site effects remain)")
            except Exception as e:
                log_and_print_test(f"Could not compute silhouette score: {e}")
    
    # ========== OVERALL CONDITION CORRELATIONS ==========
    log_and_print_test("\n=== OVERALL CONDITION CORRELATION ===")
    
    # Age correlation (overall)
    age_corr = None
    if 'Age' in annotations_df.columns:
        age_corr = np.abs(np.corrcoef(latents.T, annotations_df['Age'].values)[:-1, -1]).max()
        log_and_print_test(f"Max Age correlation (all subjects): {age_corr:.3f}")
    
    # Sex correlation
    sex_corr = None
    if 'Sex' in annotations_df.columns:
        sex_numeric = (annotations_df['Sex'].str.upper() == 'M').astype(float) if annotations_df['Sex'].dtype == 'object' else annotations_df['Sex']
        sex_corr = np.abs(np.corrcoef(latents.T, sex_numeric)[:-1, -1]).max()
        log_and_print_test(f"Max Sex correlation: {sex_corr:.3f}")
    
    # IQR correlation
    iqr_corr = None
    if 'IQR' in annotations_df.columns:
        iqr_corr = np.abs(np.corrcoef(latents.T, annotations_df['IQR'].values)[:-1, -1]).max()
        log_and_print_test(f"Max IQR correlation: {iqr_corr:.3f}")
    
    if all(c is not None and c < 0.2 for c in [age_corr, sex_corr, iqr_corr]):
        log_and_print_test("✓ EXCELLENT! Conditions well disentangled (overall)")
    elif all(c is not None and c < 0.3 for c in [age_corr, sex_corr, iqr_corr]):
        log_and_print_test("✓ GOOD! Conditions reasonably disentangled (overall)")
    else:
        log_and_print_test("⚠️  WARNING! Some conditions not fully disentangled (overall)")
    
    # ========== AGE CORRELATION BY DIAGNOSIS ==========
    log_and_print_test("\n=== AGE CORRELATION BY DIAGNOSIS ===")
    
    age_corr_by_diagnosis = {}
    if 'Age' in annotations_df.columns and 'Diagnosis' in annotations_df.columns:
        for diagnosis in sorted(annotations_df['Diagnosis'].unique()):
            mask = annotations_df['Diagnosis'] == diagnosis
            n_samples = mask.sum()
            
            if n_samples >= 5:
                diag_latents = latents[mask]
                diag_ages = annotations_df.loc[mask, 'Age'].values
                
                age_corr_diag = np.abs(np.corrcoef(diag_latents.T, diag_ages)[:-1, -1]).max()
                age_corr_by_diagnosis[diagnosis] = age_corr_diag
                
                log_and_print_test(f"  {diagnosis:10s} (n={n_samples:3d}): {age_corr_diag:.3f}")
            else:
                log_and_print_test(f"  {diagnosis:10s} (n={n_samples:3d}): skipped (too few samples)")
        
        # Interpretation
        log_and_print_test("\nInterpretation:")
        hc_corr = age_corr_by_diagnosis.get('HC', None)
        
        if hc_corr is not None:
            if hc_corr < 0.2:
                log_and_print_test(f"  ✓ HC age correlation low ({hc_corr:.3f}) - age confound removed")
            else:
                log_and_print_test(f"  ⚠️  HC age correlation high ({hc_corr:.3f}) - age confound NOT fully removed!")
            
            patient_diagnoses = [d for d in age_corr_by_diagnosis.keys() if d != 'HC']
            if patient_diagnoses:
                high_age_corr_patients = [d for d in patient_diagnoses if age_corr_by_diagnosis[d] > hc_corr + 0.1]
                
                if high_age_corr_patients:
                    log_and_print_test(f"  ✓ Patients with higher age correlation: {high_age_corr_patients}")
                    log_and_print_test(f"    → Indicates pathological age-related changes")
                else:
                    log_and_print_test(f"  → No strong age-related pathological patterns detected")
    
    # ========== SAVE NUMERICAL RESULTS ==========
    results = {
        'silhouette_score': silhouette,
        'max_age_correlation_overall': age_corr if age_corr is not None else np.nan,
        'max_sex_correlation': sex_corr if sex_corr is not None else np.nan,
        'max_iqr_correlation': iqr_corr if iqr_corr is not None else np.nan,
    }
    
    # Add per-diagnosis age correlations
    if age_corr_by_diagnosis:
        for diagnosis, corr_val in age_corr_by_diagnosis.items():
            results[f'age_corr_{diagnosis}'] = corr_val
    
    pd.DataFrame([results]).to_csv(
        f"{save_dir}/latent_analysis/latent_quality_metrics.csv",
        index=False
    )
    log_and_print_test("\n✓ Saved latent quality metrics")
    
    log_and_print_test("="*80)
    
    return umap_embeddings, results