import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import torchio as tio
from pytorch_msssim import ssim
from tqdm import tqdm

from models.base_model import (
    loss_proportions,
    supervised_contrastive_loss,
    update_performance_metrics,
)
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    process_latent_space_2D,
    save_model,
    save_model_metrics,
)
from utils.logging_utils import (
    end_logging,
    log_and_print,
    log_checkpoint,
    log_early_stopping,
    log_extracting_latent_space,
    log_model_metrics,
    log_training_start,
)
from utils.plotting_utils import (
    latent_space_batch_plot,
    metrics_plot,
    plot_learning_curves,
    plot_bootstrap_metrics,
)


# ============================================================================
# CONDITIONAL DATASET CLASS
# ============================================================================
class ConditionalDataset(Dataset):
    """
    Dataset for Conditional VAE that includes measurements and metadata conditions.
    
    Conditions encoding:
    - Age: Ordinal (normalized) - kontinuierlicher Wert
    - Sex: Binary (0/1) - 1 Feature
    - IQR: Ordinal (normalized) - kontinuierlicher Wert  
    - Dataset: One-Hot - n_datasets Features
    
    NEW: Age Binning for Contrastive Loss
    - age_bins: Quantile-based bins (default: 10 bins)
    """
    def __init__(self, measurements, metadata, dataset_categories=None, fit_scalers=True, n_age_bins=10):
        """
        Args:
            measurements: numpy array [n_samples, n_features]
            metadata: DataFrame mit Spalten ['Age', 'Sex', 'IQR', 'Dataset']
            dataset_categories: Liste der Dataset-Namen (für One-Hot). Falls None, wird automatisch bestimmt
            fit_scalers: Ob neue Scaler gefittet werden sollen (True für Training, False für Val/Test)
            n_age_bins: Number of age bins for contrastive loss (default: 10)
        """
        self.measurements = measurements
        self.metadata = metadata.reset_index(drop=True)
        self.n_age_bins = n_age_bins
        
        # Initialize scalers
        if fit_scalers:
            self.age_scaler = StandardScaler()
            self.iqr_scaler = StandardScaler()
            
            # Fit scalers on training data
            self.age_normalized = self.age_scaler.fit_transform(
                metadata['Age'].values.reshape(-1, 1)
            ).flatten()
            self.iqr_normalized = self.iqr_scaler.fit_transform(
                metadata['IQR'].values.reshape(-1, 1)
            ).flatten()
            
            # ========== NEW: Create Age Bins ==========
            if n_age_bins > 0:
                age_bins_raw = pd.qcut(metadata['Age'], q=n_age_bins, labels=False, duplicates='drop')
                self.age_bins = np.array(age_bins_raw, dtype=np.int32)
                self.age_bin_edges = pd.qcut(metadata['Age'], q=n_age_bins, retbins=True, duplicates='drop')[1]
                
                log_and_print(f"\n=== Age Binning for Contrastive Loss ===")
                log_and_print(f"Number of bins: {n_age_bins}")
                log_and_print(f"Actual bins created: {len(np.unique(self.age_bins))}")
                log_and_print(f"Age bin edges: {self.age_bin_edges}")
                
                # Log bin distribution
                unique_bins, counts = np.unique(self.age_bins, return_counts=True)
                log_and_print(f"Bin distribution:")
                for bin_id, count in zip(unique_bins, counts):
                    age_range_min = self.age_bin_edges[bin_id]
                    age_range_max = self.age_bin_edges[bin_id + 1]
                    log_and_print(f"  Bin {bin_id}: {count} samples (Age {age_range_min:.1f}-{age_range_max:.1f})")
            else:
                self.age_bins = np.zeros(len(metadata), dtype=np.int32)
                self.age_bin_edges = None
                log_and_print("Using continuous age (no binning)")
                
        else:
            # ========== FIX: Initialize even when fit_scalers=False ==========
            # Scalers will be set later via set_scalers_and_bins()
            self.age_scaler = None
            self.iqr_scaler = None
            
            # Initialize with raw values (will be overwritten by set_scalers_and_bins)
            self.age_normalized = metadata['Age'].values.astype(float)  # ← FIX!
            self.iqr_normalized = metadata['IQR'].values.astype(float)  # ← FIX!
            
            self.age_bins = np.zeros(len(metadata), dtype=np.int32)
            self.age_bin_edges = None
        
        # Sex: Binary encoding (1=Male, 0=Female oder wie in deinen Daten)
        if metadata['Sex'].dtype == 'object':
            self.sex = (metadata['Sex'].str.upper() == 'M').astype(float).values
        else:
            self.sex = metadata['Sex'].values.astype(float)
        
        # Dataset: One-Hot Encoding
        if dataset_categories is None:
            self.dataset_categories = sorted(metadata['Dataset'].unique())
        else:
            self.dataset_categories = dataset_categories
        
        # Create one-hot encoding for datasets
        self.dataset_onehot = np.zeros((len(metadata), len(self.dataset_categories)))
        for i, dataset in enumerate(metadata['Dataset']):
            if dataset in self.dataset_categories:
                idx = self.dataset_categories.index(dataset)
                self.dataset_onehot[i, idx] = 1.0
        
        # Combine all conditions: [Age, Sex, IQR, Dataset_1, Dataset_2, ...]
        self.conditions = np.column_stack([
            self.age_normalized,
            self.sex,
            self.iqr_normalized,
            self.dataset_onehot
        ])
        
        # Calculate condition_dim
        self.condition_dim = self.conditions.shape[1]
        
        log_and_print(f"Conditional Dataset created:")
        log_and_print(f"  - Measurements shape: {self.measurements.shape}")
        log_and_print(f"  - Conditions shape: {self.conditions.shape}")
        log_and_print(f"  - Condition breakdown: Age(1) + Sex(1) + IQR(1) + Dataset({len(self.dataset_categories)})")
        log_and_print(f"  - Dataset categories: {self.dataset_categories}")
        
    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
        # Return age bin as label for contrastive loss
        age_bin_label = self.age_bins[idx] if self.age_bins is not None else 0
        
        return (
            torch.FloatTensor(self.measurements[idx]),
            torch.FloatTensor(self.conditions[idx]),
            torch.tensor(age_bin_label, dtype=torch.long),  # ← Age bin as label
            self.metadata.index[idx] if hasattr(self.metadata, 'index') else f"sample_{idx}"
        )
    
    def set_scalers(self, age_scaler, iqr_scaler):
        """Set scalers from training dataset for validation/test sets"""
        self.age_scaler = age_scaler
        self.iqr_scaler = iqr_scaler
        
        # Transform with provided scalers
        self.age_normalized = age_scaler.transform(
            self.metadata['Age'].values.reshape(-1, 1)
        ).flatten()
        self.iqr_normalized = iqr_scaler.transform(
            self.metadata['IQR'].values.reshape(-1, 1)
        ).flatten()
        
        # Recreate conditions with normalized values
        self.conditions = np.column_stack([
            self.age_normalized,
            self.sex,
            self.iqr_normalized,
            self.dataset_onehot
        ])
        
        # Calculate condition_dim
        self.condition_dim = self.conditions.shape[1]
        
        log_and_print(f"Conditional Dataset created:")
        log_and_print(f"  - Measurements shape: {self.measurements.shape}")
        log_and_print(f"  - Conditions shape: {self.conditions.shape}")
        log_and_print(f"  - Condition breakdown: Age(1) + Sex(1) + IQR(1) + Dataset({len(self.dataset_categories)})")
        log_and_print(f"  - Dataset categories: {self.dataset_categories}")
        
    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.measurements[idx]),
            torch.FloatTensor(self.conditions[idx]),
            torch.tensor(0),  # Dummy label (falls nicht gebraucht)
            self.metadata.index[idx] if hasattr(self.metadata, 'index') else f"sample_{idx}"
        )
    
    def set_scalers(self, age_scaler, iqr_scaler):
        """Set scalers from training dataset for validation/test sets"""
        self.age_scaler = age_scaler
        self.iqr_scaler = iqr_scaler
        
        # Transform with provided scalers
        self.age_normalized = age_scaler.transform(
            self.metadata['Age'].values.reshape(-1, 1)
        ).flatten()
        self.iqr_normalized = iqr_scaler.transform(
            self.metadata['IQR'].values.reshape(-1, 1)
        ).flatten()
        
        # Recreate conditions with normalized values
        self.conditions = np.column_stack([
            self.age_normalized,
            self.sex,
            self.iqr_normalized,
            self.dataset_onehot
        ])


# ============================================================================
# CONDITIONAL VAE MODEL
# ============================================================================
class ConditionalVAE_2D(nn.Module):
    """
    Conditional Variational Autoencoder that incorporates metadata (Age, Sex, IQR, Dataset)
    into both encoder and decoder.
    
    Architecture:
    - Encoder: [input + conditions] -> hidden layers -> [mu, logvar]
    - Decoder: [latent + conditions] -> hidden layers -> reconstruction
    """
    def __init__(
        self,
        recon_loss_weight,
        kldiv_loss_weight,
        contr_loss_weight,
        beta=0.01,
        contr_temperature=0.1,
        input_dim: int = None, 
        condition_dim: int = None,  # Wird automatisch berechnet: 3 + n_datasets
        hidden_dim_1=100,
        hidden_dim_2=100,
        latent_dim=50,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device=None,
        dropout_prob=0.05,
        schedule_on_validation=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        kl_warmup_epochs=100,
    ):
        super(ConditionalVAE_2D, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.latent_dim = latent_dim
        self.beta = beta
        
        # KL warmup parameters
        self.kl_warmup_epochs = kl_warmup_epochs
        self.current_epoch = 0
        
        # Encoder: Takes [measurements + conditions] as input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, latent_dim)
        )
        
        self.encoder_feature_dim = int(latent_dim)
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_feature_dim, latent_dim)
        
        # Decoder: Takes [latent + conditions] as input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, input_dim),
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        # Gradient Scaler
        self.scaler = GradScaler()
        
        # Loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.contr_loss_weight = contr_loss_weight
        self.contr_temperature = contr_temperature
        
        # Logging parameters
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        
        # Device setup
        if device is not None:
            self.device = device
            self.to(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    
    def get_kl_weight(self):
        """Linear warmup for KL weight"""
        if self.current_epoch < self.kl_warmup_epochs:
            warmup_factor = self.current_epoch / self.kl_warmup_epochs
            return warmup_factor * self.kldiv_loss_weight 
        else:
            return self.kldiv_loss_weight 
    
    def _init_weights(self, module):
        """Xavier initialization for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, conditions):
        """
        Forward pass with conditions
        
        Args:
            x: Input measurements [batch_size, input_dim]
            conditions: Conditioning variables [batch_size, condition_dim]
        
        Returns:
            recon_x: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encoder: concatenate input with conditions
        encoder_input = torch.cat([x, conditions], dim=1)
        encoded = self.encoder(encoder_input)
        
        # Latent parameters
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decoder: concatenate latent with conditions
        decoder_input = torch.cat([z, conditions], dim=1)
        recon_x = self.decoder(decoder_input)
        
        return recon_x, mu, logvar
    
    def to_latent(self, x, conditions):
        """Extract latent representation given conditions"""
        self.eval()
        with torch.no_grad():
            encoder_input = torch.cat([x, conditions], dim=1)
            encoded = self.encoder(encoder_input)
            mu = self.fc_mu(encoded)
            logvar = self.fc_var(encoded)
        return mu
    
    def reconstruct(self, x, conditions):
        """Reconstruct input given conditions"""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x, conditions)
        return recon
    
    def loss_function(self, recon_x, x, mu, logvar, z=None, age_bin_labels=None, free_bits=0.5):
        """
        β-VAE loss with KL warmup, Free Bits, and optional Contrastive Loss on Age Bins
        
        Args:
            recon_x: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Latent representation (needed for contrastive loss)
            age_bin_labels: Age bin labels for contrastive loss
            free_bits: Minimum KL divergence per dimension
        
        Returns:
            total_loss, recon_loss, kldiv_loss, contr_loss
        """
        # Reconstruction loss (MSE per sample)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL divergence per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Free bits: prevent complete collapse
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        
        # Sum across latent dimensions, mean across batch
        kldiv_loss_raw = kl_per_dim_clamped.sum(dim=1).mean()
        
        # Apply dynamic KL weight with warmup and beta
        current_kl_weight = self.get_kl_weight()
        kldiv_loss = kldiv_loss_raw * current_kl_weight * self.beta
        
        # ========== NEW: Contrastive Loss on Age Bins ==========
        contr_loss = torch.tensor(0.0, device=x.device)
        if self.contr_loss_weight > 0 and z is not None and age_bin_labels is not None:
            # Normalize latent vectors (important for contrastive learning)
            z_normalized = F.normalize(z, p=2, dim=1)
            
            # Apply supervised contrastive loss with age bins as labels
            contr_loss = supervised_contrastive_loss(
                z_normalized, 
                age_bin_labels, 
                temperature=self.contr_temperature
            )
        
        # Total loss with all components
        total_loss = (self.recon_loss_weight * recon_loss + 
                    kldiv_loss + 
                    self.contr_loss_weight * contr_loss)
        
        return total_loss, recon_loss, kldiv_loss, contr_loss
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.current_epoch = epoch
        current_kl_weight = self.get_kl_weight()
        
        self.train()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (measurements, conditions, labels, names) in enumerate(train_loader):
            # Move to device
            batch_measurements = measurements.to(self.device, non_blocking=True)
            batch_conditions = conditions.to(self.device, non_blocking=True)
            batch_labels = labels.to(self.device, non_blocking=True)  # ← Age bin labels
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                recon_data, mu, logvar = self(batch_measurements, batch_conditions)
                
                # Sample z for contrastive loss
                z = self.reparameterize(mu, logvar)
                
                # ========== UPDATED: Pass z and labels to loss function ==========
                b_total_loss, b_recon_loss, b_kldiv_loss, b_contr_loss = self.loss_function(
                    recon_x=recon_data,
                    x=batch_measurements,
                    mu=mu,
                    logvar=logvar,
                    z=z,  # ← NEW
                    age_bin_labels=batch_labels  # ← NEW
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(b_total_loss).backward()
            
            # Handle NaN gradients
            for param in self.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    param.grad.nan_to_num_(0.0)
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()
            
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            del batch_measurements, batch_conditions, batch_labels, recon_data, mu, logvar, z
            del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
        
        # Calculate epoch metrics
        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "t_contr_loss": contr_loss / len(train_loader.dataset),
            "t_recon_loss": recon_loss / len(train_loader.dataset),
            "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
            "kl_weight": current_kl_weight,
            "beta": self.beta,
        }
        
        epoch_props = loss_proportions("train_loss", epoch_metrics)
        
        if epoch <= self.kl_warmup_epochs:
            logging.info(f"KL Warmup: Epoch {epoch}/{self.kl_warmup_epochs}, KL weight: {current_kl_weight:.4f}, β: {self.beta:.2f}")
        
        log_model_metrics(epoch, epoch_props, type="Training Metrics:")
        
        if not self.schedule_on_validation:
            self.scheduler.step(total_loss / len(train_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics

    @torch.no_grad()
    def validate(self, valid_loader, epoch) -> Dict[str, float]:
        """Validate the model"""
        self.current_epoch = epoch
        current_kl_weight = self.get_kl_weight()
        
        self.eval()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (measurements, conditions, labels, names) in enumerate(valid_loader):
            batch_measurements = measurements.to(self.device, non_blocking=True)
            batch_conditions = conditions.to(self.device, non_blocking=True)
            batch_labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                recon_data, mu, logvar = self(batch_measurements, batch_conditions)
                
                # Sample z for contrastive loss
                z = self.reparameterize(mu, logvar)
                
                # ========== UPDATED: Pass z and labels ==========
                b_total_loss, b_recon_loss, b_kldiv_loss, b_contr_loss = self.loss_function(
                    recon_x=recon_data,
                    x=batch_measurements,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    age_bin_labels=batch_labels
                )
            
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()
            
            del batch_measurements, batch_conditions, batch_labels, recon_data, mu, logvar, z
            del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
            
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        epoch_metrics = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_contr_loss": contr_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
            "kl_weight": current_kl_weight,
            "beta": self.beta,
        }
        
        epoch_props = loss_proportions("valid_loss", epoch_metrics)
        log_model_metrics(epoch, epoch_props, type="Validation Metrics:")
        
        if self.schedule_on_validation:
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, valid_loader, epoch) -> Dict[str, float]:
        """Validate the model"""
        self.current_epoch = epoch
        current_kl_weight = self.get_kl_weight()
        
        self.eval()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (measurements, conditions, labels, names) in enumerate(valid_loader):
            batch_measurements = measurements.to(self.device, non_blocking=True)
            batch_conditions = conditions.to(self.device, non_blocking=True)
            batch_labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                recon_data, mu, logvar = self(batch_measurements, batch_conditions)
                
                b_total_loss, b_recon_loss, b_kldiv_loss = self.loss_function(
                    recon_x=recon_data,
                    x=batch_measurements,
                    mu=mu,
                    logvar=logvar
                )
                b_contr_loss = torch.tensor(0.0)
            
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()
            
            del batch_measurements, batch_conditions, batch_labels, recon_data, mu, logvar
            del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
            
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        epoch_metrics = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_contr_loss": contr_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
            "kl_weight": current_kl_weight,
            "beta": self.beta,
        }
        
        epoch_props = loss_proportions("valid_loss", epoch_metrics)
        log_model_metrics(epoch, epoch_props, type="Validation Metrics:")
        
        if self.schedule_on_validation:
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_conditional_model_plots(train_data, valid_data, model, epochs, batch_size, save_best=True, return_history=True):
    """
    Train conditional model with plotting
    
    Args:
        train_data: ConditionalDataset for training
        valid_data: ConditionalDataset for validation
    """
    device = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'contr_loss': [],  # ← NEW
        'kl_weight': [],
        'beta': model.beta,
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Create data loaders
    log_and_print(f"Training with Conditional VAE")
    log_and_print(f"Using β-VAE with β={model.beta:.2f}")
    if model.contr_loss_weight > 0:
        log_and_print(f"Using Contrastive Loss with weight={model.contr_loss_weight:.3f}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    best_model_state = None
    
    for epoch in range(epochs):
        model.current_epoch = epoch
        current_kl_weight = model.get_kl_weight()
        
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_contr_loss = 0.0  # ← NEW
        
        for batch_idx, (data, conditions, age_bin_labels, _) in enumerate(train_loader):  # ← Changed labels to age_bin_labels
            data = data.to(device)
            conditions = conditions.to(device)
            age_bin_labels = age_bin_labels.to(device)  # ← NEW
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data, conditions)
            z = model.reparameterize(mu, log_var)  # ← NEW: Sample z
            
            # ← UPDATED: Pass z and age_bin_labels
            loss, recon_loss, kl_loss, contr_loss = model.loss_function(
                recon_batch, data, mu, log_var, z=z, age_bin_labels=age_bin_labels
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_contr_loss += contr_loss.item()  # ← NEW
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_contr_loss = 0.0  # ← NEW
        
        with torch.no_grad():
            for batch_idx, (data, conditions, age_bin_labels, _) in enumerate(valid_loader):  # ← Changed
                data = data.to(device)
                conditions = conditions.to(device)
                age_bin_labels = age_bin_labels.to(device)  # ← NEW
                
                recon_batch, mu, log_var = model(data, conditions)
                z = model.reparameterize(mu, log_var)  # ← NEW
                
                # ← UPDATED
                loss, recon_loss, kl_loss, contr_loss = model.loss_function(
                    recon_batch, data, mu, log_var, z=z, age_bin_labels=age_bin_labels
                )
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_contr_loss += contr_loss.item()  # ← NEW
        
        # Normalize
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        train_contr_loss /= len(train_loader)  # ← NEW
        
        val_loss /= len(valid_loader)
        val_recon_loss /= len(valid_loader)
        val_kl_loss /= len(valid_loader)
        val_contr_loss /= len(valid_loader)  # ← NEW
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(train_recon_loss)
        history['kl_loss'].append(train_kl_loss)
        history['contr_loss'].append(train_contr_loss)  # ← NEW
        history['kl_weight'].append(current_kl_weight)
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            if save_best:
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs-1:
            log_msg = (f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Recon: {train_recon_loss:.4f}, "
                      f"KL: {train_kl_loss:.4f}")
            if model.contr_loss_weight > 0:
                log_msg += f", Contr: {train_contr_loss:.4f}"
            log_msg += f", KL_weight: {current_kl_weight:.4f}, β: {model.beta:.2f}"
            log_and_print(log_msg)
    
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if return_history:
        return model, history
    return model

def bootstrap_train_conditional_models_plots(train_data, valid_data, model, n_bootstraps, epochs, batch_size, save_dir, save_models=True):
    """
    Bootstrap training for conditional models
    
    Args:
        train_data: ConditionalDataset for training
        valid_data: ConditionalDataset for validation
    """
    n_samples = len(train_data)
    device = model.device
    bootstrap_models = []
    bootstrap_metrics = []
    all_losses = []
    
    # Create directory structure
    figures_dir = os.path.join(save_dir, "figures")
    latent_dir = os.path.join(figures_dir, "latent_space")
    recon_dir = os.path.join(figures_dir, "reconstructions")
    loss_dir = os.path.join(figures_dir, "loss_curves")
    
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    
    log_and_print(f"Starting bootstrap training with {n_bootstraps} iterations")
    log_and_print(f"KL warmup will occur over first {model.kl_warmup_epochs} epochs")
    log_and_print(f"Using Conditional β-VAE with β={model.beta:.2f}")
    
    for i in range(n_bootstraps):
        log_and_print(f"Training bootstrap model {i+1}/{n_bootstraps}")
        
        # Create bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Create bootstrap dataset
        bootstrap_measurements = train_data.measurements[bootstrap_indices]
        bootstrap_metadata = train_data.metadata.iloc[bootstrap_indices].reset_index(drop=True)
        
        bootstrap_train_data = ConditionalDataset(
            measurements=bootstrap_measurements,
            metadata=bootstrap_metadata,
            dataset_categories=train_data.dataset_categories,
            fit_scalers=True
        )
        
        # Set scalers for validation data to match training
        bootstrap_valid_data = ConditionalDataset(
            measurements=valid_data.measurements,
            metadata=valid_data.metadata,
            dataset_categories=train_data.dataset_categories,
            fit_scalers=False
        )
        bootstrap_valid_data.set_scalers(
            bootstrap_train_data.age_scaler,
            bootstrap_train_data.iqr_scaler
        )
        
        # Create fresh model
        bootstrap_model = ConditionalVAE_2D(
            input_dim=model.input_dim,
            condition_dim=model.condition_dim,
            hidden_dim_1=model.hidden_dim_1,
            hidden_dim_2=model.hidden_dim_2,
            latent_dim=model.latent_dim,
            learning_rate=model.learning_rate,
            kldiv_loss_weight=model.kldiv_loss_weight,
            dropout_prob=model.dropout_prob,
            recon_loss_weight=model.recon_loss_weight,
            contr_loss_weight=model.contr_loss_weight,
            kl_warmup_epochs=model.kl_warmup_epochs,
            beta=model.beta,
            device=device
        )
        
        # Train model
        trained_model, history = train_conditional_model_plots(
            train_data=bootstrap_train_data,
            valid_data=bootstrap_valid_data,
            model=bootstrap_model,
            epochs=epochs,
            batch_size=batch_size,
            save_best=True,
            return_history=True
        )
        
        # Extract metrics
        metrics = {
            'bootstrap_id': i,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_recon_loss': history['recon_loss'][-1],
            'final_kl_loss': history['kl_loss'][-1],
            'final_contr_loss': history['contr_loss'][-1],
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'beta': history['beta']
        }
        
        bootstrap_models.append(trained_model)
        bootstrap_metrics.append(metrics)
        all_losses.append(history)
        
        # Save model
        if save_models:
            model_save_path = os.path.join(save_dir, "models", f"bootstrap_model_{i}.pt")
            torch.save(trained_model.state_dict(), model_save_path)
            log_and_print(f"Saved model {i+1} to {model_save_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame(bootstrap_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "models", "bootstrap_metrics.csv"), index=False)
    
    # Create combined loss plots
    plt.figure(figsize=(15, 10))  # ← Changed from (12, 8) to (15, 10)
    
    # Determine if we need 2x2 or 2x3 grid
    has_contr_loss = bootstrap_models[0].contr_loss_weight > 0
    n_plots = 5 if has_contr_loss else 4
    n_cols = 3 if has_contr_loss else 2
    
    plt.subplot(2, n_cols, 1)
    for i, history in enumerate(all_losses):
        plt.plot(history['val_loss'], alpha=0.3, color='blue')
    plt.plot([np.mean([h['val_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Validation Loss')
    plt.title(f'Validation Loss Across Bootstrap Models (β={model.beta:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, n_cols, 2)
    for i, history in enumerate(all_losses):
        plt.plot(history['train_loss'], alpha=0.3, color='green')
    plt.plot([np.mean([h['train_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Training Loss')
    plt.title(f'Training Loss Across Bootstrap Models (β={model.beta:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, n_cols, 3)
    for i, history in enumerate(all_losses):
        plt.plot(history['recon_loss'], alpha=0.3, color='purple')
    plt.plot([np.mean([h['recon_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Reconstruction Loss')
    plt.title(f'Reconstruction Loss Across Bootstrap Models (β={model.beta:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, n_cols, 4)
    for i, history in enumerate(all_losses):
        plt.plot(history['kl_loss'], alpha=0.3, color='orange')
    plt.plot([np.mean([h['kl_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean KL Divergence Loss')
    plt.title(f'KL Divergence Loss Across Bootstrap Models (β={model.beta:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # ← NEW: Add contrastive loss plot if used
    if has_contr_loss:
        plt.subplot(2, n_cols, 5)
        for i, history in enumerate(all_losses):
            plt.plot(history['contr_loss'], alpha=0.3, color='cyan')
        plt.plot([np.mean([h['contr_loss'][e] for h in all_losses]) for e in range(epochs)], 
                 linewidth=2, color='red', label='Mean Contrastive Loss')
        plt.title(f'Contrastive Loss Across Bootstrap Models (weight={model.contr_loss_weight:.3f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "bootstrap_losses.png"))
    plt.close()
    
    plot_bootstrap_metrics(bootstrap_metrics, os.path.join(figures_dir, "bootstrap_metrics_distribution.png"))
    
    return bootstrap_models, bootstrap_metrics


# ============================================================================
# LATENT SPACE EXTRACTION
# ============================================================================
@torch.no_grad()
def extract_latent_space_conditional(model, data_loader, data_type):
    """
    Extract latent space from conditional VAE
    
    Args:
        model: Trained ConditionalVAE_2D
        data_loader: DataLoader with ConditionalDataset
        data_type: String identifier for logging
    """
    log_extracting_latent_space(data_type)
    
    model.eval()
    latent_spaces = []
    sample_names = []
    logvars = []
    conditions_list = []
    
    for batch_idx, (measurements, conditions, labels, names) in enumerate(data_loader):
        batch_measurements = measurements.to(model.device, non_blocking=True)
        batch_conditions = conditions.to(model.device, non_blocking=True)
        
        # Get latent representations
        mu, logvar = model.to_latent(batch_measurements, batch_conditions)
        
        # Move to CPU
        latent_spaces.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        conditions_list.append(conditions.cpu().numpy())
        sample_names.extend(names)
        
        # Clean up
        del batch_measurements, batch_conditions, mu
        
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Create anndata object
    adata = ad.AnnData(np.concatenate(latent_spaces))
    adata.obs_names = sample_names
    
    # Add conditions as observations
    conditions_array = np.concatenate(conditions_list)

    adata.uns['logvar_array'] = np.concatenate(logvars)
    
    # Decompose conditions back to original features
    # Assuming order: [Age, Sex, IQR, Dataset_1, Dataset_2, ...]
    adata.obs['age_normalized'] = conditions_array[:, 0]
    adata.obs['sex'] = conditions_array[:, 1]
    adata.obs['iqr_normalized'] = conditions_array[:, 2]
    
    # Dataset one-hot (rest of columns)
    n_datasets = conditions_array.shape[1] - 3
    for i in range(n_datasets):
        adata.obs[f'dataset_{i}'] = conditions_array[:, 3 + i]
    
    return adata


# ============================================================================
# HELPER FUNCTIONS FOR USAGE
# ============================================================================
def create_conditional_datasets(train_measurements, train_metadata, 
                                valid_measurements, valid_metadata,
                                test_measurements=None, test_metadata=None,
                                n_age_bins=10):
    """
    Create ConditionalDatasets with proper scaler handling and age binning
    """
    
    # Create training dataset (fits scalers and creates age bins)
    train_dataset = ConditionalDataset(
        measurements=train_measurements,
        metadata=train_metadata,
        fit_scalers=True,
        n_age_bins=n_age_bins
    )
    
    log_and_print(f"Training dataset created with {len(train_dataset)} samples")
    
    # Create validation dataset (uses training scalers and bins)
    valid_dataset = ConditionalDataset(
        measurements=valid_measurements,
        metadata=valid_metadata,
        dataset_categories=train_dataset.dataset_categories,
        fit_scalers=False,
        n_age_bins=n_age_bins
    )
    
    # ← FIX: Use set_scalers instead of set_scalers_and_bins
    valid_dataset.set_scalers(
        train_dataset.age_scaler, 
        train_dataset.iqr_scaler
    )
    
    log_and_print(f"Validation dataset created with {len(valid_dataset)} samples")
    
    if test_measurements is not None and test_metadata is not None:
        test_dataset = ConditionalDataset(
            measurements=test_measurements,
            metadata=test_metadata,
            dataset_categories=train_dataset.dataset_categories,
            fit_scalers=False,
            n_age_bins=n_age_bins
        )
        test_dataset.set_scalers(
            train_dataset.age_scaler, 
            train_dataset.iqr_scaler
        )
        
        log_and_print(f"Test dataset created with {len(test_dataset)} samples")
        return train_dataset, valid_dataset, test_dataset
    
    return train_dataset, valid_dataset


def initialize_conditional_vae(input_dim, condition_dim, **kwargs):
    """
    Initialize ConditionalVAE_2D with sensible defaults
    
    Args:
        input_dim: Dimension of input features
        condition_dim: Dimension of conditioning variables (3 + n_datasets)
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model: Initialized ConditionalVAE_2D
    """
    
    default_params = {
        'hidden_dim_1': 100,
        'hidden_dim_2': 100,
        'latent_dim': 20,
        'beta': 4.0,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'dropout_prob': 0.1,
        'recon_loss_weight': 1.0,
        'kldiv_loss_weight': 1.0,
        'contr_loss_weight': 0.0,
        'kl_warmup_epochs': 50,
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
    }
    
    # Update with provided kwargs
    default_params.update(kwargs)
    
    model = ConditionalVAE_2D(
        input_dim=input_dim,
        condition_dim=condition_dim,
        **default_params
    )
    
    return model


