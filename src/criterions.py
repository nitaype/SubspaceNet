'''

criterions for evaluating the model:
1. RMSPE for doa estimation - validation
2. SISNR for speech enhancement
3. spectrum_loss for doa estimation - train

'''

import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # Calculate RMSE over all permutations
                rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            # Choose minimal error from all permutations
            rmspe_min = torch.min(rmspe_tensor)
            rmspe.append(rmspe_min)
        result = torch.mean(torch.stack(rmspe, dim=0))
        return result

class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio (SISNR) loss function.
    This loss function calculates the SISNR between the MVDR-filtered signal and the target (clean)
    signal.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(filtered_signal: torch.Tensor, clean_signal: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self):
        super(SISNRLoss, self).__init__()

    def forward(self, filtered_signal: torch.Tensor, clean_signal: torch.Tensor):
        """
        Compute the SISNR loss between the MVDR-filtered signal and the target (clean) signal.
        Args:
            filtered_signal (torch.Tensor): MVDR-filtered signal tensor of shape (batch_size, num_snapshots).
            clean_signal (torch.Tensor): Clean signal tensor of shape (batch_size, num_snapshots).

        Returns:
            torch.Tensor: The computed SISNR loss.
        """

        epsilon = 1e-8

        # Zero-mean normalization
        filtered_signal = filtered_signal - filtered_signal.mean(dim=-1, keepdim=True)
        clean_signal = clean_signal - clean_signal.mean(dim=-1, keepdim=True)

        # print("filtered_signal mean (should be ~0):", filtered_signal.mean().item())
        # print("clean_signal mean (should be ~0):", clean_signal.mean().item())

        # Compute power of clean (reference)
        reference_pow = clean_signal.pow(2).mean(dim=-1, keepdim=True)
        # print("reference_pow shape:", reference_pow.shape)
        # print("reference_pow mean:", reference_pow.mean().item())

        # Compute projection (scale factor)
        mix_pow = (filtered_signal * clean_signal).mean(dim=-1, keepdim=True)
        # print("mix_pow shape:", mix_pow.shape)
        # print("mix_pow mean:", mix_pow.mean().item())

        scale = mix_pow / (reference_pow + epsilon)
        # print("scale mean:", scale.mean().item())

        # Project reference onto estimated
        scaled_reference = scale * clean_signal
        error = filtered_signal - scaled_reference

        # Power of projected reference and error
        reference_pow = scaled_reference.pow(2).mean(dim=-1)
        error_pow = error.pow(2).mean(dim=-1)

        # print("scaled reference power mean:", reference_pow.mean().item())
        # print("error power mean:", error_pow.mean().item())

        # SI-SNR calculation
        si_snr = 10 * torch.log10(reference_pow / (error_pow + epsilon))
        # print("SI-SNR mean:", si_snr.mean().item())

        # Return negative for loss
        return -si_snr.mean()

class Spectrum_Loss(nn.Module):
    """
    Loss based on MUSIC spectrum values at true DOA locations.

    The loss penalizes low spectrum peaks at true DOAs.

    Args:
        None

    Forward Args:
        inverse_spectrum (torch.Tensor): MUSIC spectrum, shape (B, Q)
        doa (torch.Tensor): True DOA angles in radians, shape (B, M)

    Returns:
        loss (torch.Tensor): Scalar loss value
    """
    def __init__(self):
        super(Spectrum_Loss, self).__init__()

    def forward(self, inverse_spectrum: torch.Tensor, doa: torch.Tensor):
        B, M = doa.shape
        _, Q = inverse_spectrum.shape
        # print(B, M, Q)
        # F = inverse_spectrum.shape[0] // B

        # # Expand DOA
        # doa_expanded = doa.unsqueeze(1).expand(B, F, M)  # shape: (B, F, M)
        # doa = doa_expanded.reshape(B * F, M)    # shape: (B*F, M)

        # Now: 
        # - inverse_spectrum shape: (B*F, Q)
        # - doa_expanded shape: (B*F, M)

        # Create angle grid over Q bins (from -π/2 to π/2)
        angle_grid = torch.linspace(-np.pi / 2, np.pi / 2, Q, device=inverse_spectrum.device)

        # Find nearest angle index in the grid for each DOA
        doa_indices = torch.argmin(torch.abs(doa.unsqueeze(-1) - angle_grid), dim=-1)  # (B, M)

        # Gather
        selected_inverse_spectrum = torch.gather(inverse_spectrum, dim=1, index=doa_indices)  # (B, M, 1)

        # Remove last dimension
        selected_inverse_spectrum = selected_inverse_spectrum.squeeze(-1)  # (B, M)

        # Compute loss: sum over M (sources), then mean over B (batch)
        # loss = selected_inverse_spectrum.sum(dim=1).mean()
        loss = selected_inverse_spectrum.mean()

        return loss

def set_criterions(criterion_name:str):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    if criterion_name.startswith("rmse"):
        criterion = RMSPELoss()
        subspace_criterion = RMSPE
    elif criterion_name.startswith("mse"):
        criterion = MSPELoss()
        subspace_criterion = MSPE
    elif criterion_name.startswith("sisnr"):
        criterion = SISNRLoss()
        subspace_criterion = SISNRLoss
    elif criterion_name.startswith("csisnr"):
        criterion = CSISNRLoss()
        subspace_criterion = CSISNRLoss
    elif criterion_name.startswith("Spectrum_Loss"):
        criterion = Spectrum_Loss()
        subspace_criterion = Spectrum_Loss
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"Loss measure = {criterion_name}")
    return criterion, subspace_criterion

if __name__ == "__main__":
    print('')