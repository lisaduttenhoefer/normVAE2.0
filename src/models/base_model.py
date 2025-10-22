from typing import Dict, List
import pandas as pd
import torch
import torch.nn.functional as F

from utils.logging_utils import log_and_print

"""
These are basic functions ever model can make use of. -> prior Code to base the other models on 
"""

# This function takes the existing model metrics as a DataFrame and the metrics of the current epoch as a list of dictionaries.
# Each dictionary can contain metrics for multiple losses, e.g. {"valid_loss": 0.5, "v_recon_loss": 0.3, "v_kldiv_loss": 0.2}.
# It then updates the data frame with the new metrics and returns it.
def update_performance_metrics(
    # The existing model metrics as a DataFrame
    model_metrics: pd.DataFrame,
    # The metrics of the current epoch as a list of dictionaries
    epoch_metrics: List[Dict[str, float]],
) -> pd.DataFrame:

    # Combine metrics dicts into one
    epoch_losses_comb = epoch_metrics[0]

    for i in range(1, len(epoch_metrics)):
        epoch_losses_comb.update(epoch_metrics[i])

    # Convert combined metrics to DataFrame
    epoch_losses_df = pd.DataFrame(epoch_losses_comb, index=[0])

    # Append to model_metrics
    model_metrics = pd.concat([model_metrics, epoch_losses_df], axis=0)
    model_metrics.reset_index(drop=True, inplace=True)

    # Return updated model_metrics
    return model_metrics


# This function takes a dictionary of losses and returns a dictionary of loss proportions. By specifying the total_loss_key,
# the function will calculate the proportion of each loss relative to the total loss. If the total loss is 0, the function will
# return the input dictionary unchanged. The function will also add the total loss to the output dictionary. If there are metrics
# that are not losses in the input dictionary, they will be added to the output dictionary unchanged.
def loss_proportions(
        # The dict key of the total loss
        total_loss_key: str, 
        # The dictionary of losses
        losses_dict: Dict[str, float]) -> Dict[str, float]:

    # Get the total loss
    total_loss = losses_dict[total_loss_key]

    # If the total loss is 0, we cannot calculate proportions
    if total_loss == 0:
        log_and_print("Total loss is 0, cannot calculate proportions.")
        return losses_dict

    # Create a dictionary to store the loss proportions
    prop_dict = {}
    # Add the total loss to the dictionary to start
    prop_dict[total_loss_key] = total_loss

    # Calculate the proportion of each loss relative to the total loss
    for key, value in losses_dict.items():
        # ignore the total loss key
        if key == total_loss_key:
            continue

        # Calculate the proportion of the loss
        # only calculate proportions for losses, not for other metrics
        if "loss" in key:
            # replace "_loss" with "_prop"
            key = key.split("_")[-2] + "_prop"
            # store the proportion of the loss	
            prop_dict[key] = value / total_loss
        # for other metrics...
        else:
            # ... store the metric unchanged
            prop_dict[key] = value
    
    # Return the dictionary of loss proportions
    return prop_dict


# Supervised contrastive loss as described in https://arxiv.org/pdf/2004.11362#page=4.15, the L-sup-out variant.
# Labels are expected to be integers, not one-hot encoded. They must have the same order as the input data.
# Z is a tensor with dimensions (batch_size, embedding_size) containing the latent vectors of the samples.
# Temperature is a hyperparameter that scales the similarity between samples.
# For every sample in z, (i.e.) every row in z), we calculate the Loss, then we mean it all.
def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature=0.1) -> torch.Tensor:
    # Normalize the latent vectors by dividing them by their length (L2 norm / Euclidian distance)
    # This is done to make the similarity calculation more stable, and its cosine similarity now
    z = F.normalize(z, p=2, dim=1)

    # Compute similarity matrix S via dot product of every latent vector with every other latent vector
    # So each row is one samples similarity to every other one.
    # Each pairwise distance is S(ij). S(ij) = z(i) * z(j) / temperature
    similarity_matrix = torch.matmul(z, z.T) / temperature

    # Mask to remove self-comparisons, in all the sums of dot products, self similarity is always excluded
    mask = torch.eye(z.shape[0], device=z.device).bool()

    # Create a mask for positive pairs, which are pairs of samples that have the same label
    labels = labels.contiguous().view(-1, 1)
    mask_labels = (labels == labels.T).float()

    # Compute log probability.
    # So here for every pairwise distance exp(S(ij)), were norming it by dividing it by the sum of all pairwise distances for i.
    # So for every sample (=row) we're calculating: exp(S(ij))/sum(exp(S(ij)))
    # So we're dividing every row by the sum of the row.
    # We use the log-sum-exp trick to make the calculation more stable.
    # log(exp(S)/Sum(exp(S)) = log(exp(S)) - log(Sum(exp(S)) = S - log(Sum(exp(S))
    exp_similarity_matrix = torch.exp(similarity_matrix)
    exp_similarity_matrix = exp_similarity_matrix.masked_fill(mask, 0)
    log_prob = similarity_matrix - torch.log(
        exp_similarity_matrix.sum(dim=1, keepdim=True)
    )

    # Compute mean log-likelihood over positive pairs
    # Only positive pairs are summed in the formula, hence the mask.
    # We norm by the number of positive pairs, as per the formula.
    mean_log_prob_pos = (mask_labels * log_prob).sum(dim=1) / mask_labels.sum(dim=1)

    # We combine the loss of every sample into one scalar by taking the negative mean of the log probabilities.
    # In the original paper, the loss is defined as the negative SUM of the log probabilities, but mean is more stable here.
    loss = -mean_log_prob_pos.mean()

    # Return calculated loss
    return loss