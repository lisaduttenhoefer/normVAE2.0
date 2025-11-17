import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
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
from torch.utils.data import DataLoader, TensorDataset
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

# ContrastVAE_2D Model Class
class NormativeVAE_2D(nn.Module):
    def __init__(
        self,
        recon_loss_weight,
        kldiv_loss_weight,
        contr_loss_weight,
        beta=0.2,  # ← NEU: β-VAE Parameter (Standard: 4.0, für normalen VAE: 1.0)
        contr_temperature=0.1,
        input_dim:int = None, 
        hidden_dim_1=100, #based on Pinaya
        hidden_dim_2=100, #based on Pinaya
        latent_dim=20, #based on Pinaya & tuning
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=None,
        dropout_prob=0.1,
        schedule_on_validation=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        kl_warmup_epochs=50,  # Number of epochs for KL warmup
        
    ):
        super(NormativeVAE_2D, self).__init__()
        
        # Store important parameters for later
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.latent_dim = latent_dim
        
        # β-VAE parameter
        self.beta = beta  # ← NEU: β gespeichert
        
        # KL warmup parameters
        self.kl_warmup_epochs = kl_warmup_epochs
        self.current_epoch = 0  # Track current epoch for warmup
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, latent_dim)
        )
        
        self.encoder_feature_dim = int(latent_dim)
        
        # Mu and logvar projections
        self.fc_mu = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_feature_dim, latent_dim)
        
         # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, input_dim),
            #input data is normalized so no nn.Sigmoid() necessary
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler for learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        # Gradient Scaler for Mixed Precision
        self.scaler = GradScaler()
        
        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.contr_loss_weight = contr_loss_weight
        
        # Additional loss metrics
        self.contr_temperature = contr_temperature
        
        # Record some values for logging
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        
        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def get_kl_weight(self):
        """
        Gradually increase KL weight from 0 to target weight over warmup_epochs.
        This prevents posterior collapse by allowing the encoder to learn 
        meaningful representations before enforcing the N(0,1) prior.
        """
        if self.current_epoch < self.kl_warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.kl_warmup_epochs
            return warmup_factor * self.kldiv_loss_weight * 4.0
        else:
            # Full weight after warmup
            return self.kldiv_loss_weight * 4.0
            
    #Initialize weights for linear layers
    def _init_weights(self, module): 
    
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    #Reparameterization trick for VAE
    def reparameterize(self, mu, logvar):
        # self: The latent space produced by the encode
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    #Forward pass through the model
    def forward(self, x):
        
        x = self.encoder(x)
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        # Decoder
        x = self.decoder(z)
        
        return x, mu, logvar
    
    #Extract latent space representation
    def to_latent(self, x):
        # Encoder
        self.eval()
        with torch.no_grad():
            x = self.encoder(x)
            # Latent space
            mu = self.fc_mu(x)
        
        return mu
    
    #Rekonstruiere einen Input
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
        return recon
    
    # ← GEÄNDERT: β-VAE loss function mit KL warmup UND β
    def loss_function(self, recon_x, x, mu, logvar, free_bits=0.5):
        """
        β-VAE loss with KL warmup and Free Bits.
        
        Beta (β) controls disentanglement:
        - β = 1: Standard VAE (keine zusätzliche Regularisierung)
        - β > 1: Stärkeres Disentanglement (z.B. β=4 ist typisch)
        - β < 1: Schwächere Regularisierung
        
        Free Bits: Don't penalize KL below 'free_bits' per dimension.
        This prevents complete collapse while still allowing meaningful structure.
        """
        # Rekonstruktionsloss (MSE between input and reconstructed feature map)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL-Divergenz per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Free Bits: Clamp each dimension to minimum 'free_bits'
        # This prevents collapse to 0 while maintaining gradients
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        
        # Sum across latent dimensions, mean across batch
        kldiv_loss_raw = kl_per_dim_clamped.sum(dim=1).mean()
        
        # ← GEÄNDERT: Apply dynamic KL weight with warmup UND β
        current_kl_weight = self.get_kl_weight()
        kldiv_loss = kldiv_loss_raw * current_kl_weight * self.beta  # ← β hier angewendet
        
        # total loss
        total_loss = recon_loss + kldiv_loss
        
        return total_loss, recon_loss, kldiv_loss
    

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        # Update current epoch for KL warmup
        self.current_epoch = epoch
        current_kl_weight = self.get_kl_weight()
        
        # Set to train mode
        self.train()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        # Iterate through batches
        for batch_idx, (measurements, labels, names) in enumerate(train_loader):
            # More efficient tensor handling - avoid double stacking
            if isinstance(measurements, list):
                batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
            else:
                batch_measurements = measurements.to(self.device, non_blocking=True)
                
            if isinstance(labels, list):
                batch_labels = torch.stack(labels).to(self.device, non_blocking=True)
            else:
                batch_labels = labels.to(self.device, non_blocking=True)
            # Clear gradients before forward pass
            self.optimizer.zero_grad()

            # Autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                # Forward pass
                recon_data, mu, logvar = self(batch_measurements)
                
                # Calculate loss
                b_total_loss, b_recon_loss, b_kldiv_loss = self.loss_function(
                    recon_x=recon_data,
                    x=batch_measurements,
                    mu=mu,
                    logvar=logvar
                )
                b_contr_loss = torch.tensor(0.0)  # No contrastive loss in this version
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(b_total_loss).backward()
            
            # Remove NaNs from gradients
            for param in self.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    param.grad.nan_to_num_(0.0)
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update loss stats
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

            # Clear cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Delete tensors to free memory
            del batch_measurements, batch_labels, recon_data, mu, logvar
            del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
        
        # Calculate epoch metrics
        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "t_contr_loss": contr_loss / len(train_loader.dataset),
            "t_recon_loss": recon_loss / len(train_loader.dataset),
            "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
            "kl_weight": current_kl_weight,
            "beta": self.beta,  # ← NEU: β zum Logging hinzugefügt
        }
        
        # Calculate loss proportions
        epoch_props = loss_proportions("train_loss", epoch_metrics)
    
        # Log KL warmup progress
        if epoch <= self.kl_warmup_epochs:
            logging.info(f"KL Warmup: Epoch {epoch}/{self.kl_warmup_epochs}, KL weight: {current_kl_weight:.4f}, β: {self.beta:.2f}")
        
        log_model_metrics(epoch, epoch_props, type="Training Metrics:")
        
        # Update learning rate if needed
        if not self.schedule_on_validation:
            self.scheduler.step(total_loss / len(train_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, valid_loader, epoch) -> Dict[str, float]:
        # Update current epoch for KL warmup
        self.current_epoch = epoch
        current_kl_weight = self.get_kl_weight()
        
        # Set to evaluation mode
        self.eval()
        
        # Initialize loss values
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        # Go through validation data
        for batch_idx, (measurements, labels, names) in enumerate(valid_loader):
            if isinstance(measurements, list):
                batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
            else:
                batch_measurements = measurements.to(self.device, non_blocking=True)
                
            if isinstance(labels, list):
                batch_labels = torch.stack(labels).to(self.device, non_blocking=True)
            else:
                batch_labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                recon_data, mu, logvar = self(batch_measurements)
                
                # Calculate loss
                b_total_loss, b_recon_loss, b_kldiv_loss = self.loss_function(
                    recon_x=recon_data,
                    x=batch_measurements,
                    mu=mu,
                    logvar=logvar
                )
                b_contr_loss = torch.tensor(0.0)  # No contrastive loss in this version
            
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()
        
            # Clear memory
            del batch_measurements, batch_labels, recon_data, mu, logvar
            del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
        # Calculate metrics
        epoch_metrics = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_contr_loss": contr_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
            "kl_weight": current_kl_weight,
            "beta": self.beta,  # ← NEU: β zum Logging hinzugefügt
        }
    
        epoch_props = loss_proportions("valid_loss", epoch_metrics)
        
        log_model_metrics(epoch, epoch_props, type="Validation Metrics:")
        
        # Update learning rate if needed
        if self.schedule_on_validation:
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics
    
    
# TRAINING FUNCTION with epoch tracking for warmup
def train_normative_model_plots(train_data, valid_data, model, epochs, batch_size, save_best=True, return_history=True):
    
    device = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'kl_weight': [],
        'beta': model.beta,  # ← NEU: β im History gespeichert
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Create data loaders
    log_and_print(f"Training data shape IM MODEL: {train_data.shape}")
    log_and_print(f"Using β-VAE with β={model.beta:.2f}")  # ← NEU: β loggen
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = torch.utils.data.TensorDataset(valid_data)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    best_model_state = None
    
    for epoch in range(epochs):
        # Update model's epoch counter
        model.current_epoch = epoch
        current_kl_weight = model.get_kl_weight()
        
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(valid_loader):
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Normalize by batch count
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        val_loss /= len(valid_loader)
        val_recon_loss /= len(valid_loader)
        val_kl_loss /= len(valid_loader)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(train_recon_loss)
        history['kl_loss'].append(train_kl_loss)
        history['kl_weight'].append(current_kl_weight)
        
        # Save best model based on validation loss
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            if save_best:
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs-1:
            log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Recon: {train_recon_loss:.4f}, "
                         f"KL: {train_kl_loss:.4f}, KL_weight: {current_kl_weight:.4f}, β: {model.beta:.2f}")  # ← NEU: β loggen
    
    # Load best model if available
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if return_history:
        return model, history
    return model

#TRAINING FUNCTION for multiple models (bootstrapping), returns trained models & metrics & saves loss plots (all models in one graph with mean)
def bootstrap_train_normative_models_plots(train_data, valid_data, model, n_bootstraps, epochs, batch_size, save_dir, save_models=True):
   
    n_samples = train_data.shape[0]
    device = model.device
    bootstrap_models = []
    bootstrap_metrics = []
    all_losses = []
    
    # Create figures directory structure
    figures_dir = os.path.join(save_dir, "figures")
    latent_dir = os.path.join(figures_dir, "latent_space")
    recon_dir = os.path.join(figures_dir, "reconstructions")
    loss_dir = os.path.join(figures_dir, "loss_curves")
    
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    
    log_and_print(f"Starting bootstrap training with {n_bootstraps} iterations")
    log_and_print(f"KL warmup will occur over first {model.kl_warmup_epochs} epochs")
    log_and_print(f"Using β-VAE with β={model.beta:.2f}")  # ← NEU: β loggen
    
    # Combine training and validation data for visualization
    combined_data = torch.cat([train_data, valid_data], dim=0)
    combined_labels = torch.cat([torch.zeros(train_data.shape[0]), torch.ones(valid_data.shape[0])])
    
    for i in range(n_bootstraps):
        log_and_print(f"Training bootstrap model {i+1}/{n_bootstraps}")
        
        # Create bootstrap sample (with replacement)
        bootstrap_indices = torch.randint(0, n_samples, (n_samples,))
        bootstrap_train_data = train_data[bootstrap_indices]
        
        # Create a fresh model instance
        bootstrap_model = NormativeVAE_2D(
            input_dim=model.input_dim,
            hidden_dim_1=model.hidden_dim_1,
            hidden_dim_2=model.hidden_dim_2,
            latent_dim=model.latent_dim,
            learning_rate=model.learning_rate,
            kldiv_loss_weight=model.kldiv_loss_weight,
            dropout_prob=model.dropout_prob,
            recon_loss_weight=model.recon_loss_weight,
            contr_loss_weight=model.contr_loss_weight,
            kl_warmup_epochs=model.kl_warmup_epochs,
            beta=model.beta,  # ← NEU: β übergeben
            device=device
        )
        
        # Train the model and collect metrics
        trained_model, history = train_normative_model_plots(
            train_data=bootstrap_train_data,
            valid_data=valid_data,
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
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'beta': history['beta']  # ← NEU: β in Metriken
        }
        
        bootstrap_models.append(trained_model)
        bootstrap_metrics.append(metrics)
        all_losses.append(history)
        
        # Save model if requested
        if save_models:
            model_save_path = os.path.join(save_dir, "models", f"bootstrap_model_{i}.pt")
            torch.save(trained_model.state_dict(), model_save_path)
            log_and_print(f"Saved model {i+1} to {model_save_path}")
        
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame(bootstrap_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "models", "bootstrap_metrics.csv"), index=False)
    
    # Create combined loss plots from all bootstraps
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i, history in enumerate(all_losses):
        plt.plot(history['val_loss'], alpha=0.3, color='blue')
    plt.plot([np.mean([h['val_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Validation Loss')
    plt.title(f'Validation Loss Across Bootstrap Models (β={model.beta:.2f})')  # ← NEU
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for i, history in enumerate(all_losses):
        plt.plot(history['train_loss'], alpha=0.3, color='green')
    plt.plot([np.mean([h['train_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Training Loss')
    plt.title(f'Training Loss Across Bootstrap Models (β={model.beta:.2f})')  # ← NEU
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for i, history in enumerate(all_losses):
        plt.plot(history['recon_loss'], alpha=0.3, color='purple')
    plt.plot([np.mean([h['recon_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Reconstruction Loss')
    plt.title(f'Reconstruction Loss Across Bootstrap Models (β={model.beta:.2f})')  # ← NEU
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for i, history in enumerate(all_losses):
        plt.plot(history['kl_loss'], alpha=0.3, color='orange')
    plt.plot([np.mean([h['kl_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean KL Divergence Loss')
    plt.title(f'KL Divergence Loss Across Bootstrap Models (β={model.beta:.2f})')  # ← NEU
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "bootstrap_losses.png"))
    plt.close()
    
    plot_bootstrap_metrics(bootstrap_metrics, os.path.join(figures_dir, "bootstrap_metrics_distribution.png"))
    
    return bootstrap_models, bootstrap_metrics

#extract latent space
@torch.no_grad()
def extract_latent_space(self, data_loader, data_type):
    log_extracting_latent_space(data_type)
    
    self.eval()
    latent_spaces = []
    sample_names = []
    
    # Process in smaller chunks to avoid memory issues
    for batch_idx, (measurements, labels, names) in enumerate(data_loader):
        if isinstance(measurements, list):
            batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
        else:
            batch_measurements = measurements.to(self.device, non_blocking=True)
        
        # Get latent representations
        mu = self.to_latent(batch_measurements)
        
        # Move to CPU immediately to free GPU memory
        latent_spaces.append(mu.cpu().numpy())
        sample_names.extend(names)
        
        # Clean up GPU memory
        del batch_measurements, mu
    
        # Clear cache every few batches
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Create anndata object
    adata = ad.AnnData(np.concatenate(latent_spaces))
    adata.obs_names = sample_names        
    return adata