"""
VAE Latent Space Diagnostic Tool
================================
Diagnose why MDD patients have lower KL divergence than HC controls.
This script checks for:
1. Posterior collapse
2. Latent space variance
3. Distribution differences
4. Reconstruction quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd
from torch.utils.data import DataLoader


class VAEDiagnostics:
    def __init__(self, model, device=None):
        """
        Initialize diagnostics tool
        
        Args:
            model: Your trained NormativeVAE_2D model
            device: torch device (will auto-detect if None)
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_latents_and_stats(self, data_loader, group_name=""):
        """
        Extract latent representations and compute statistics
        
        Returns:
            dict with mu, logvar, z, recon_errors, kl_per_sample
        """
        print(f"\n{'='*60}")
        print(f"Extracting latents for: {group_name}")
        print(f"{'='*60}")
        
        all_mu = []
        all_logvar = []
        all_z = []
        all_inputs = []
        all_recons = []
        all_labels = []
        all_names = []
        
        for batch_idx, (measurements, labels, names) in enumerate(data_loader):
            # Handle different input formats
            if isinstance(measurements, list):
                batch_measurements = torch.stack(measurements).to(self.device)
            else:
                batch_measurements = measurements.to(self.device)
            
            # Forward pass through encoder
            encoded = self.model.encoder(batch_measurements)
            mu = self.model.fc_mu(encoded)
            logvar = self.model.fc_var(encoded)
            
            # Sample z
            z = self.model.reparameterize(mu, logvar)
            
            # Get reconstruction
            recon = self.model.decoder(z)
            
            # Store everything
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_z.append(z.cpu())
            all_inputs.append(batch_measurements.cpu())
            all_recons.append(recon.cpu())
            all_labels.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
            all_names.extend(names)
        
        # Concatenate all batches
        mu = torch.cat(all_mu, dim=0)
        logvar = torch.cat(all_logvar, dim=0)
        z = torch.cat(all_z, dim=0)
        inputs = torch.cat(all_inputs, dim=0)
        recons = torch.cat(all_recons, dim=0)
        
        # Compute statistics
        results = {
            'mu': mu.numpy(),
            'logvar': logvar.numpy(),
            'z': z.numpy(),
            'inputs': inputs.numpy(),
            'recons': recons.numpy(),
            'labels': np.array(all_labels),
            'names': all_names,
            'group_name': group_name
        }
        
        # Compute per-sample metrics
        results['kl_per_sample'] = self._compute_kl_per_sample(mu, logvar)
        results['recon_error_per_sample'] = self._compute_recon_error_per_sample(inputs, recons)
        results['l2_norm_per_sample'] = torch.norm(mu, dim=1).numpy()
        
        return results
    
    def _compute_kl_per_sample(self, mu, logvar):
        """Compute KL divergence per sample (not averaged over dimensions)"""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.numpy()
    
    def _compute_recon_error_per_sample(self, inputs, recons):
        """Compute reconstruction error per sample"""
        errors = torch.mean((inputs - recons) ** 2, dim=1)
        return errors.numpy()
    
    def check_posterior_collapse(self, results):
        """
        Check if latent dimensions have collapsed
        
        Signs of collapse:
        - Very low variance in latent dimensions
        - Most dimensions close to 0
        - logvar very negative (high certainty, low variance)
        """
        print(f"\n{'='*60}")
        print(f"POSTERIOR COLLAPSE CHECK - {results['group_name']}")
        print(f"{'='*60}")
        
        mu = results['mu']
        logvar = results['logvar']
        
        # 1. Variance per dimension
        var_per_dim = np.var(mu, axis=0)
        mean_per_dim = np.mean(mu, axis=0)
        std_per_dim = np.std(mu, axis=0)
        
        print(f"\nLatent Space Statistics (across {len(mu)} samples):")
        print(f"  Mean variance per dimension: {var_per_dim.mean():.6f}")
        print(f"  Std variance per dimension: {var_per_dim.std():.6f}")
        print(f"  Min variance: {var_per_dim.min():.6f}")
        print(f"  Max variance: {var_per_dim.max():.6f}")
        
        # 2. Check for collapsed dimensions
        collapsed_threshold = 0.01
        collapsed_dims = np.sum(var_per_dim < collapsed_threshold)
        print(f"\n  Collapsed dimensions (var < {collapsed_threshold}): {collapsed_dims}/{len(var_per_dim)}")
        
        if collapsed_dims > 0:
            print(f"  ⚠️  WARNING: {collapsed_dims} dimensions have collapsed!")
            print(f"  Collapsed dimension indices: {np.where(var_per_dim < collapsed_threshold)[0]}")
        
        # 3. Check logvar (should be around 0 for N(0,1))
        mean_logvar = np.mean(logvar, axis=0)
        print(f"\n  Mean logvar per dimension: {mean_logvar.mean():.4f}")
        print(f"  Std logvar per dimension: {mean_logvar.std():.4f}")
        
        # Very negative logvar = model is very certain = potential collapse
        very_certain_dims = np.sum(mean_logvar < -5)
        if very_certain_dims > 0:
            print(f"  ⚠️  WARNING: {very_certain_dims} dimensions have very negative logvar (< -5)")
        
        # 4. Distance from origin
        l2_norms = results['l2_norm_per_sample']
        print(f"\n  Mean L2 norm (distance to origin): {l2_norms.mean():.4f}")
        print(f"  Std L2 norm: {l2_norms.std():.4f}")
        
        # If all samples cluster near origin → collapse
        if l2_norms.mean() < 0.5:
            print(f"  ⚠️  WARNING: Samples very close to origin (mean norm < 0.5)")
        
        # 5. Active dimensions (contribute meaningfully)
        active_dims = np.sum(var_per_dim > 0.1)
        print(f"\n  Active dimensions (var > 0.1): {active_dims}/{len(var_per_dim)}")
        
        results['collapse_stats'] = {
            'var_per_dim': var_per_dim,
            'collapsed_dims': collapsed_dims,
            'active_dims': active_dims,
            'mean_l2_norm': l2_norms.mean()
        }
        
        return results
    
    def compare_groups(self, hc_results, patient_results):
        """
        Compare HC and patient group statistics
        
        This is the KEY function to understand why MDD has lower KL than HC!
        """
        print(f"\n{'='*60}")
        print(f"GROUP COMPARISON: {hc_results['group_name']} vs {patient_results['group_name']}")
        print(f"{'='*60}")
        
        # 1. KL Divergence comparison
        hc_kl = hc_results['kl_per_sample']
        patient_kl = patient_results['kl_per_sample']
        
        print(f"\nKL Divergence Statistics:")
        print(f"  HC    - Mean: {hc_kl.mean():.4f}, Std: {hc_kl.std():.4f}, Median: {np.median(hc_kl):.4f}")
        print(f"  {patient_results['group_name']:5s} - Mean: {patient_kl.mean():.4f}, Std: {patient_kl.std():.4f}, Median: {np.median(patient_kl):.4f}")
        print(f"  Difference: {patient_kl.mean() - hc_kl.mean():.4f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(hc_kl, patient_kl)
        print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        if patient_kl.mean() < hc_kl.mean():
            print(f"  ⚠️  PROBLEM DETECTED: {patient_results['group_name']} has LOWER KL than HC!")
        
        # 2. Reconstruction Error comparison
        hc_recon = hc_results['recon_error_per_sample']
        patient_recon = patient_results['recon_error_per_sample']
        
        print(f"\nReconstruction Error Statistics:")
        print(f"  HC    - Mean: {hc_recon.mean():.6f}, Std: {hc_recon.std():.6f}")
        print(f"  {patient_results['group_name']:5s} - Mean: {patient_recon.mean():.6f}, Std: {patient_recon.std():.6f}")
        print(f"  Difference: {patient_recon.mean() - hc_recon.mean():.6f}")
        
        t_stat, p_value = stats.ttest_ind(hc_recon, patient_recon)
        print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        # 3. Variance comparison (are patients more or less variable?)
        hc_var = np.var(hc_results['mu'], axis=0).mean()
        patient_var = np.var(patient_results['mu'], axis=0).mean()
        
        print(f"\nLatent Space Variance:")
        print(f"  HC    - Mean variance: {hc_var:.6f}")
        print(f"  {patient_results['group_name']:5s} - Mean variance: {patient_var:.6f}")
        print(f"  Ratio: {patient_var / hc_var:.4f}")
        
        if patient_var < hc_var:
            print(f"  ⚠️  {patient_results['group_name']} has LESS variance → easier to fit to prior!")
        
        # 4. Distance to origin comparison
        hc_dist = hc_results['l2_norm_per_sample']
        patient_dist = patient_results['l2_norm_per_sample']
        
        print(f"\nDistance to Origin (L2 norm):")
        print(f"  HC    - Mean: {hc_dist.mean():.4f}, Std: {hc_dist.std():.4f}")
        print(f"  {patient_results['group_name']:5s} - Mean: {patient_dist.mean():.4f}, Std: {patient_dist.std():.4f}")
        
        # 5. Mahalanobis distance (proper anomaly detection)
        print(f"\n{'='*60}")
        print(f"MAHALANOBIS DISTANCE (proper anomaly detection)")
        print(f"{'='*60}")
        
        hc_mu = hc_results['mu']
        patient_mu = patient_results['mu']
        
        # Compute HC distribution parameters
        hc_mean = np.mean(hc_mu, axis=0)
        hc_cov = np.cov(hc_mu.T)
        
        # Add regularization to prevent singular matrix
        hc_cov_reg = hc_cov + np.eye(hc_cov.shape[0]) * 1e-6
        
        try:
            hc_cov_inv = np.linalg.inv(hc_cov_reg)
            
            # Compute Mahalanobis distance for both groups
            def mahalanobis(x, mean, cov_inv):
                diff = x - mean
                return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
            
            hc_mahal = mahalanobis(hc_mu, hc_mean, hc_cov_inv)
            patient_mahal = mahalanobis(patient_mu, hc_mean, hc_cov_inv)
            
            print(f"  HC    - Mean Mahalanobis: {hc_mahal.mean():.4f}, Std: {hc_mahal.std():.4f}")
            print(f"  {patient_results['group_name']:5s} - Mean Mahalanobis: {patient_mahal.mean():.4f}, Std: {patient_mahal.std():.4f}")
            
            t_stat, p_value = stats.ttest_ind(hc_mahal, patient_mahal)
            print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
            
            if patient_mahal.mean() > hc_mahal.mean():
                print(f"  ✅ GOOD: {patient_results['group_name']} is further from HC distribution")
            else:
                print(f"  ⚠️  PROBLEM: {patient_results['group_name']} is closer to HC distribution")
            
            # Store for plotting
            hc_results['mahalanobis'] = hc_mahal
            patient_results['mahalanobis'] = patient_mahal
            
        except np.linalg.LinAlgError:
            print("  ⚠️  Could not compute Mahalanobis (singular covariance matrix)")
    
    def create_diagnostic_plots(self, hc_results, patient_results, save_path="vae_diagnostics.png"):
        """
        Create comprehensive diagnostic plots
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('VAE Diagnostic Plots: HC vs Patient Group', fontsize=16, fontweight='bold')
        
        hc_name = hc_results['group_name']
        patient_name = patient_results['group_name']
        
        # 1. KL Divergence Distribution
        ax = axes[0, 0]
        ax.hist(hc_results['kl_per_sample'], bins=50, alpha=0.5, label=hc_name, color='blue', density=True)
        ax.hist(patient_results['kl_per_sample'], bins=50, alpha=0.5, label=patient_name, color='red', density=True)
        ax.axvline(hc_results['kl_per_sample'].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(patient_results['kl_per_sample'].mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Density')
        ax.set_title('KL Divergence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Reconstruction Error Distribution
        ax = axes[0, 1]
        ax.hist(hc_results['recon_error_per_sample'], bins=50, alpha=0.5, label=hc_name, color='blue', density=True)
        ax.hist(patient_results['recon_error_per_sample'], bins=50, alpha=0.5, label=patient_name, color='red', density=True)
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Density')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. L2 Norm Distribution
        ax = axes[0, 2]
        ax.hist(hc_results['l2_norm_per_sample'], bins=50, alpha=0.5, label=hc_name, color='blue', density=True)
        ax.hist(patient_results['l2_norm_per_sample'], bins=50, alpha=0.5, label=patient_name, color='red', density=True)
        ax.set_xlabel('L2 Norm (Distance to Origin)')
        ax.set_ylabel('Density')
        ax.set_title('Distance to Origin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Variance per Dimension
        ax = axes[1, 0]
        hc_var = np.var(hc_results['mu'], axis=0)
        patient_var = np.var(patient_results['mu'], axis=0)
        x = np.arange(len(hc_var))
        ax.bar(x - 0.2, hc_var, 0.4, label=hc_name, color='blue', alpha=0.7)
        ax.bar(x + 0.2, patient_var, 0.4, label=patient_name, color='red', alpha=0.7)
        ax.axhline(0.01, color='black', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Variance')
        ax.set_title('Variance per Latent Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Mean per Dimension
        ax = axes[1, 1]
        hc_mean = np.mean(hc_results['mu'], axis=0)
        patient_mean = np.mean(patient_results['mu'], axis=0)
        ax.bar(x - 0.2, hc_mean, 0.4, label=hc_name, color='blue', alpha=0.7)
        ax.bar(x + 0.2, patient_mean, 0.4, label=patient_name, color='red', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean')
        ax.set_title('Mean per Latent Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Logvar Distribution
        ax = axes[1, 2]
        hc_logvar_mean = np.mean(hc_results['logvar'], axis=0)
        patient_logvar_mean = np.mean(patient_results['logvar'], axis=0)
        ax.bar(x - 0.2, hc_logvar_mean, 0.4, label=hc_name, color='blue', alpha=0.7)
        ax.bar(x + 0.2, patient_logvar_mean, 0.4, label=patient_name, color='red', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, label='Target (N(0,1))')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean Log Variance')
        ax.set_title('Log Variance per Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. PCA of Latent Space
        ax = axes[2, 0]
        pca = PCA(n_components=2)
        combined_mu = np.vstack([hc_results['mu'], patient_results['mu']])
        pca_result = pca.fit_transform(combined_mu)
        
        n_hc = len(hc_results['mu'])
        ax.scatter(pca_result[:n_hc, 0], pca_result[:n_hc, 1], 
                  alpha=0.5, s=20, label=hc_name, color='blue')
        ax.scatter(pca_result[n_hc:, 0], pca_result[n_hc:, 1], 
                  alpha=0.5, s=20, label=patient_name, color='red')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA of Latent Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Mahalanobis Distance (if computed)
        ax = axes[2, 1]
        if 'mahalanobis' in hc_results:
            ax.hist(hc_results['mahalanobis'], bins=50, alpha=0.5, label=hc_name, color='blue', density=True)
            ax.hist(patient_results['mahalanobis'], bins=50, alpha=0.5, label=patient_name, color='red', density=True)
            ax.axvline(hc_results['mahalanobis'].mean(), color='blue', linestyle='--', linewidth=2)
            ax.axvline(patient_results['mahalanobis'].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Mahalanobis Distance to HC')
            ax.set_ylabel('Density')
            ax.set_title('Mahalanobis Distance Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Mahalanobis not computed', ha='center', va='center')
            ax.axis('off')
        
        # 9. Scatter: KL vs Reconstruction Error
        ax = axes[2, 2]
        ax.scatter(hc_results['kl_per_sample'], hc_results['recon_error_per_sample'],
                  alpha=0.5, s=20, label=hc_name, color='blue')
        ax.scatter(patient_results['kl_per_sample'], patient_results['recon_error_per_sample'],
                  alpha=0.5, s=20, label=patient_name, color='red')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('KL vs Reconstruction Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Diagnostic plots saved to: {save_path}")
        plt.close()
    
    def generate_report(self, hc_results, patient_results, save_path="diagnostic_report.txt"):
        """
        Generate a text report summarizing findings
        """
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VAE DIAGNOSTIC REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Latent Dimension: {self.model.latent_dim}\n")
            f.write(f"HC Group: {hc_results['group_name']} (n={len(hc_results['mu'])})\n")
            f.write(f"Patient Group: {patient_results['group_name']} (n={len(patient_results['mu'])})\n\n")
            
            f.write("="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")
            
            # Finding 1: KL Divergence
            hc_kl_mean = hc_results['kl_per_sample'].mean()
            patient_kl_mean = patient_results['kl_per_sample'].mean()
            
            f.write("1. KL DIVERGENCE\n")
            f.write(f"   HC Mean KL: {hc_kl_mean:.4f}\n")
            f.write(f"   Patient Mean KL: {patient_kl_mean:.4f}\n")
            f.write(f"   Difference: {patient_kl_mean - hc_kl_mean:.4f}\n")
            
            if patient_kl_mean < hc_kl_mean:
                f.write("   ⚠️  PROBLEM: Patients have LOWER KL than HC!\n")
                f.write("   → Patients appear more 'normal' according to KL metric\n")
                f.write("   → KL divergence to N(0,1) is NOT appropriate for anomaly detection\n\n")
            else:
                f.write("   ✅ Patients have higher KL than HC (as expected)\n\n")
            
            # Finding 2: Posterior Collapse
            hc_collapsed = hc_results['collapse_stats']['collapsed_dims']
            patient_collapsed = patient_results['collapse_stats']['collapsed_dims']
            total_dims = self.model.latent_dim
            
            f.write("2. POSTERIOR COLLAPSE CHECK\n")
            f.write(f"   HC Collapsed Dimensions: {hc_collapsed}/{total_dims}\n")
            f.write(f"   Patient Collapsed Dimensions: {patient_collapsed}/{total_dims}\n")
            
            if hc_collapsed > total_dims * 0.3:
                f.write("   ⚠️  SEVERE COLLAPSE: >30% of dimensions collapsed\n")
                f.write("   → Model is not using latent space effectively\n\n")
            elif hc_collapsed > 0:
                f.write("   ⚠️  PARTIAL COLLAPSE: Some dimensions collapsed\n\n")
            else:
                f.write("   ✅ No collapsed dimensions detected\n\n")
            
            # Finding 3: Reconstruction Quality
            hc_recon = hc_results['recon_error_per_sample'].mean()
            patient_recon = patient_results['recon_error_per_sample'].mean()
            
            f.write("3. RECONSTRUCTION ERROR\n")
            f.write(f"   HC Mean Error: {hc_recon:.6f}\n")
            f.write(f"   Patient Mean Error: {patient_recon:.6f}\n")
            f.write(f"   Difference: {patient_recon - hc_recon:.6f}\n")
            
            if patient_recon > hc_recon:
                f.write("   ✅ Patients have higher reconstruction error (as expected)\n")
                f.write("   → Reconstruction-based anomaly detection may work better!\n\n")
            else:
                f.write("   ⚠️  Patients have LOWER reconstruction error\n")
                f.write("   → Model reconstructs patients better than HC\n\n")
            
            # Finding 4: Mahalanobis Distance
            if 'mahalanobis' in hc_results:
                hc_mahal = hc_results['mahalanobis'].mean()
                patient_mahal = patient_results['mahalanobis'].mean()
                
                f.write("4. MAHALANOBIS DISTANCE TO HC DISTRIBUTION\n")
                f.write(f"   HC Mean Distance: {hc_mahal:.4f}\n")
                f.write(f"   Patient Mean Distance: {patient_mahal:.4f}\n")
                f.write(f"   Difference: {patient_mahal - hc_mahal:.4f}\n")
                
                if patient_mahal > hc_mahal:
                    f.write("   ✅ Patients are further from HC distribution\n")
                    f.write("   → Mahalanobis distance is working correctly!\n\n")
                else:
                    f.write("   ⚠️  Patients are closer to HC distribution\n\n")
            
            # Recommendations
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            if patient_kl_mean < hc_kl_mean:
                f.write("⚠️  PRIMARY ISSUE: KL divergence to N(0,1) is inappropriate for anomaly detection\n\n")
                f.write("RECOMMENDED SOLUTIONS:\n\n")
                f.write("1. Use Mahalanobis distance to HC distribution instead of KL to N(0,1)\n")
                f.write("   - Measures distance from HC, not from standard normal\n")
                f.write("   - More appropriate for normative modeling\n\n")
                
                f.write("2. Use reconstruction error as anomaly score\n")
                f.write("   - Model trained on HC should reconstruct HC better\n")
                f.write("   - Higher reconstruction error = more abnormal\n\n")
                
                f.write("3. Combine multiple metrics:\n")
                f.write("   deviation_score = α * mahalanobis + β * recon_error\n\n")
            
            if hc_collapsed > 0:
                f.write("⚠️  SECONDARY ISSUE: Posterior collapse detected\n\n")
                f.write("RECOMMENDED SOLUTIONS:\n\n")
                f.write("1. Increase β in β-VAE (try β=4-10)\n")
                f.write("2. Use two-stage training\n")
                f.write("3. Try Wasserstein VAE with MMD loss\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✅ Diagnostic report saved to: {save_path}")


def run_full_diagnostics(model, hc_loader, patient_loader, 
                         hc_name="HC", patient_name="MDD",
                         save_dir="./diagnostics"):
    """
    Main function to run complete diagnostics
    
    Args:
        model: Trained NormativeVAE_2D model
        hc_loader: DataLoader for HC training data
        patient_loader: DataLoader for patient test data
        hc_name: Name for HC group
        patient_name: Name for patient group
        save_dir: Directory to save outputs
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING VAE DIAGNOSTICS")
    print("="*80)
    
    # Initialize diagnostics
    diagnostics = VAEDiagnostics(model)
    
    # Extract latent spaces and compute statistics
    hc_results = diagnostics.extract_latents_and_stats(hc_loader, group_name=hc_name)
    patient_results = diagnostics.extract_latents_and_stats(patient_loader, group_name=patient_name)
    
    # Check for posterior collapse
    hc_results = diagnostics.check_posterior_collapse(hc_results)
    patient_results = diagnostics.check_posterior_collapse(patient_results)
    
    # Compare groups
    diagnostics.compare_groups(hc_results, patient_results)
    
    # Create plots
    plot_path = os.path.join(save_dir, "vae_diagnostics.png")
    diagnostics.create_diagnostic_plots(hc_results, patient_results, save_path=plot_path)
    
    # Generate report
    report_path = os.path.join(save_dir, "diagnostic_report.txt")
    diagnostics.generate_report(hc_results, patient_results, save_path=report_path)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print(f"Outputs saved to: {save_dir}")
    
    return hc_results, patient_results
