'''

This is code of utils functions.
gram_diagonal_overload - for diagonal oveloading of matrices.
set_unified_seed - for a unifies randomness seed.

'''
# Imports
import numpy as np
import torch
import random
import scipy
import warnings

# Constants
R2D = 180 / np.pi
D2R = 1 / R2D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def gram_diagonal_overload(Kx: torch.Tensor, eps: float) -> torch.Tensor:
def gram_diagonal_overload(Kx: torch.Tensor, eps: float):
    """
    Multiply a matrix Kx with its Hermitian conjugate (gram matrix),
    and add eps to the diagonal to ensure a Hermitian and PSD matrix.

    Args:
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N].
        eps (float): Value added to each diagonal element.

    Returns:
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].
    """
    # Ensure Tensor
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)
    Kx = Kx.to(device)

    # Compute Gram matrix
    Kx_gram = torch.bmm(Kx.conj().transpose(1, 2), Kx)

    # Add epsilon to diagonal
    eps_addition = eps * torch.eye(Kx_gram.shape[-1], device=Kx.device).unsqueeze(0)  # (1, N, N)
    Kx_Out = Kx_gram + eps_addition

    # Check Hermitian: A^H = A
    mask = (torch.abs(Kx_Out - Kx_Out.conj().transpose(1, 2)) > 1e-6)

    # Fix batch_mask
    batch_mask = mask.view(mask.shape[0], -1).any(dim=1)

    if batch_mask.any():
        warnings.warn(f"gram_diagonal_overload: {batch_mask.sum().item()} matrices in the batch aren't Hermitian; averaging R and R^H.")
        Kx_Out[batch_mask] = 0.5 * (Kx_Out[batch_mask] + Kx_Out[batch_mask].conj().transpose(1, 2))

    return Kx_Out

def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    print('')
