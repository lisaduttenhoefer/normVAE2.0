import math
import os
import random as rd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Literal
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import torchio as tio
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from module.data_processing_hc import load_mri_data_2D
from utils.logging_utils import log_and_print

"""
Plotting functions for all the train_ModelXYZ() functions used in the  run_ModelXYZ.py files.
"""


# This function plots the metrics of a model over epochs. It can plot each metric in a separate subplot, but it can also
# combine all the training and validation losses in two subplots, accompanied by an overall train/validation comparison.
# The function can also plot a rolling average of the metrics, and can skip the first N epochs of plotting (the firs
# epochs of training can mess with the plot scales). The function can save the plots to a specified path, and can also
# show the plots in the console.
def metrics_plot(
    metrics: pd.DataFrame,
    timestamp: str,
    skip_first_n_epochs: int = 0,
    plot_types: List[Literal["separate", "combined"]] = ["separate", "combined"],
    rolling_window: int = None, # Should the rolling average of metrics be plotted instead? If so, what is the window size of the rolling average?
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    metrics_copy = metrics.copy()

    metrics_copy = metrics_copy.loc[:, (metrics_copy != 0).any(axis=0)]

    title_addendum = ""
    file_addendum = ""


    if skip_first_n_epochs > 0:
        if len(metrics_copy) > skip_first_n_epochs:
            # Remove the first N epochs
            metrics_copy = metrics_copy.iloc[skip_first_n_epochs:, :]
            # Add information to the plot title
            title_addendum = f" (first {skip_first_n_epochs} epochs not shown)"

        # If there are not enough epochs to skip
        else:
            # Log that you couldn't skip epochs. Don't change the metrics dataframe.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Cannot skip first {skip_first_n_epochs} epochs in plot, not enough epochs to skip."

            log_and_print(prefix + message)

    # If a rolling average is used
    if rolling_window is not None:
        # If the rolling window is too large, set it to the maximum possible value
        if rolling_window > (len(metrics_copy) - 1):
            # Log that the rolling window is being set to the maximum possible value.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Rolling window of {rolling_window} too large, setting window to {max(len(metrics_copy) - 1, 1)} instead."

            log_and_print(prefix + message)

            # Set the rolling window to the maximum possible value
            rolling_window = max(len(metrics_copy) - 1, 1)

        # If the rolling window is valid, calculate the rolling average
        for metric in metrics_copy.columns:
            # Calculate the rolling average
            metrics_copy[metric] = (
                metrics_copy[metric].rolling(window=rolling_window).mean()
            )

        # Add information to the plot title and file name if a rolling average was used.
        title_addendum += f" (rolling average over {rolling_window} epochs)"
        file_addendum = f"_rolling"

    # Titles and colors for all the metrics we expect to see in the metrics dataframe
    metric_annotations = {
        # annotations for loss components
        "class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for training loss components
        "t_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "t_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "t_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "t_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for validation loss components
        "v_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "v_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "v_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "v_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for general losses
        "conf_loss": {"title": "Confounder Adversarial Loss", "color": "blue"},
        "train_loss": {"title": "Total Training Loss", "color": "green"},
        "valid_loss": {"title": "Total Validation Loss", "color": "purple"},
        # annotations for confusion metrics
        "accuracy": {"title": "Accuracy", "color": "blue"},
        "precision": {"title": "Precision", "color": "orange"},
        "recall": {"title": "Recall", "color": "green"},
        "f1-score": {"title": "f1-Score", "color": "red"},
        # annotation for learning_rate
        "learning_rate": {"title": "Learning Rate", "color": "black"},
        "VAE_learning_rate": {"title": "VAE Learning Rate", "color": "black"},
        "adv_class_lr": {
            "title": "Adversarial Encoder Learning Rate",
            "color": "black",
        },
        "adv_encod_lr": {
            "title": "Adversarial Classifier Learning Rate",
            "color": "black",
        },
    }

    # If the "separate" plot was desired, plot all metrics separately
    if "separate" in plot_types:
        # collect relevant colors and title names
        titles = [
            metric_annotations[metric]["title"] for metric in metrics_copy.columns
        ]
        colors = [
            metric_annotations[metric]["color"] for metric in metrics_copy.columns
        ]

        # Each row of the plot will contain 4 subplots
        nrows = math.ceil(len(metrics_copy.columns) / 4)

        # Plot each metric in a separate subplot using pandas plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, ax = plt.subplots(figsize=(24, 6 * nrows))
            metrics_copy.plot(
                kind="line",
                subplots=True,
                xlabel="Epoch",
                ylabel="Loss",
                layout=(nrows, 4),
                grid=True,
                title=titles,
                color=colors,
                ax=ax,
                use_index=True,
            )

        # Set plot title and layout
        plt.suptitle("Loss Components Over Epochs" + title_addendum, fontsize=30)
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_metrics_separate{file_addendum}.png"
                )
            )
        if show:
            plt.show()

        # Close the plot (for stability)
        plt.close(fig)

    # If the "combined" plot was desired, plot all metrics in three subplots:
    # - one for the total training and validation loss together
    # - one for training loss components (KLD loss, recon loss, class loss, etc)
    # - one for validation loss components (KLD loss, recon loss, class loss, etc)
    if "combined" in plot_types:
        # Plot losses in one figure
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Get the toal valid and total train loss
        main_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("valid_")
            ]
        ]

        # Get the components of the train and valid losses
        train_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("t_")
            ]
        ]
        valid_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("valid_")
                | metrics_copy.columns.str.startswith("v_")
            ]
        ]

        # Plot the losses in the three subplots
        for i, model_losses, title in zip(
            range(3),
            [main_losses, train_losses, valid_losses],
            ["Combined", "Train", "Validation"],
        ):
            # Get the relevant colors and titles for the metrics
            colors = [
                metric_annotations[metric]["color"] for metric in model_losses.columns
            ]
            titles = [
                metric_annotations[metric]["title"] for metric in model_losses.columns
            ]

            # Plot the losses in the chosen subplot
            model_losses.plot(
                kind="line",
                xlabel="Epochs",
                ylabel="Losses",
                ax=ax[i],
                color=colors,
                label=titles,
                title=f"{title} Loss Over Epochs" + title_addendum,
                use_index=True,
            )

        # Set the plot title and layout
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_losses_combined{file_addendum}.png"
                )
            )
        if show:
            plt.show()

        plt.close(fig)

# This function plots the latent space of the model, broken down by diagnosis. Each diagnosis will be plotted in a separate
# column, and each row will be one of the datasets in the data. The first row is all the datasets combined, as an overview.
# The function can save the plots to a specified path, and can also show the plots in the console.
def latent_space_batch_plot(
    data: ad.AnnData, # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data_type: str, # The type of data being plotted, such as "training" or "validation". Used for the plot title and file name.
    save_path: str,
    timestamp: str,
    epoch: int,
    show: bool = True,
    save: bool = False,
    descriptor: str = "", # A descriptor to add to the end of file name of the plot.
    size: int = 30, # The size of the points in the plot.
):
    # Get the unique diagnoses and datasets in the data
    diagnoses = data.obs["Diagnosis"].unique()
    datasets = data.obs["Dataset"].unique()

    nrows = len(datasets) + 1
    ncols = len(diagnoses)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    palette = {dataset: sns.color_palette()[i] for i, dataset in enumerate(datasets)}

    for dia, diagnosis in enumerate(diagnoses):

        diagnosis_data = data[data.obs["Diagnosis"] == diagnosis].copy()
        sc.pl.umap(
            diagnosis_data,
            ax=axs[0, dia],
            title=f"{diagnosis} - All",
            color="Dataset",
            palette=palette,
            size=size,
            show=False,
            return_fig=False,
        )

        # For each dataset; dat is used for the row index, and dataset is the dataset name
        for dat, dataset in enumerate(data.obs["Dataset"].unique(), start=1):

            # Get the data for this dataset and diagnosis
            subset_data = data[
                (data.obs["Diagnosis"] == diagnosis) & (data.obs["Dataset"] == dataset)
            ].copy()

            # If there is no data for this dataset and diagnosis, skip this subplot
            if len(subset_data) == 0:
                axs[dat, dia].axis("off")
                continue

            # plot grey background of all the data in this diagnosis
            sc.pl.umap(
                diagnosis_data,
                ax=axs[dat, dia],
                show=False,
                return_fig=False,
                size=size,
            )

            # plot the data for this dataset and diagnosis in color
            sc.pl.umap(
                subset_data,
                ax=axs[dat, dia],
                title=f"{diagnosis} - {dataset}",
                color="Dataset",
                palette=palette,
                size=size,
                show=False,
                return_fig=False,
            )

    plt.suptitle(
        f"Batch Effect Breakdown of {data_type.capitalize()} Data (UMAP)",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.90)

    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_batch_{data_type}{descriptor}.png",
            )
        )

    plt.show() if show else plt.close()


def plot_learning_curves(train_losses, val_losses, kl_losses, recon_losses, save_path):
    
    #Plot training and validation loss curves

    plt.figure(figsize=(15, 10))
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # Reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # KL divergence loss
    plt.subplot(2, 2, 3)
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # Log scale for total loss
    plt.subplot(2, 2, 4)
    plt.semilogy(train_losses, label='Training Loss (log scale)')
    plt.semilogy(val_losses, label='Validation Loss (log scale)')
    plt.title('Total Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bootstrap_metrics(bootstrap_metrics, save_path):
    
    #Plot metrics from bootstrap models
    
    df = pd.DataFrame(bootstrap_metrics)
    
    plt.figure(figsize=(15, 10))
    # Final loss distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['final_val_loss'], kde=True)
    plt.title('Distribution of Final Validation Loss')
    plt.xlabel('Validation Loss')
    plt.grid(True)
    # Final KL loss distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['final_kl_loss'], kde=True)
    plt.title('Distribution of Final KL Divergence Loss')
    plt.xlabel('KL Loss')
    plt.grid(True)
    # Final reconstruction loss distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['final_recon_loss'], kde=True)
    plt.title('Distribution of Final Reconstruction Loss')
    plt.xlabel('Reconstruction Loss')
    plt.grid(True)
    # Best epoch distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['best_epoch'], kde=False, discrete=True)
    plt.title('Distribution of Best Epochs')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

