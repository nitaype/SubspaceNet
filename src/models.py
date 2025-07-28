'''

this code defines the SubspaceNet model
also, mvdr diff method and inverse music spectrum calculation

'''

# Imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
from src.utils import gram_diagonal_overload, device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.simplefilter("ignore")
# Constants
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type = None
        self.diff_method = None
        self.tau = None

    def set_tau(self, tau: int = None):
        """
        Set the value of tau parameter for SubspaceNet model.

        Parameters:
            tau (int): The number of lags.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If tau parameter is not provided for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if not isinstance(tau, int):
                raise ValueError(
                    "ModelParams.set_tau: tau parameter must be provided for SubspaceNet model"
                )
            self.tau = tau
        return self

    def set_diff_method(self, diff_method: str = "root_music"):
        """
        Set the differentiation method for SubspaceNet model.

        Parameters:
            diff_method (str): The differantiable subspace method ("esprit" or "root_music").

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If the diff_method is not defined for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if diff_method not in ["esprit", "root_music", "mvdr", "music"]:
                raise ValueError(
                    f"ModelParams.set_diff_method: {diff_method} is not defined for SubspaceNet model"
                )
            self.diff_method = diff_method
        return self

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelParams.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def set_model(self, system_model_params):
        """
        Set the model based on the model type and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the model type is not defined.
        """
        if self.model_type.startswith("SubspaceNet"):
            self.model = SubspaceNet(
                tau=self.tau, M=system_model_params.M, diff_method=self.diff_method
            )
        else:
            raise Exception(
                f"ModelGenerator.set_model: Model type {self.model_type} is not defined"
            )
        return self

class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, M: int, diff_method: str = "mvdr"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.M = M
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        # Set the subspace method for training
        self.set_diff_method(diff_method)
        # self._init_weights()  # 👈 call weight initialization - not working good

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        elif diff_method.startswith("mvdr"):
            self.diff_method = mvdr
        elif diff_method.startswith("music"):
            self.diff_method = music_spectrum
        else:
            raise Exception(
                f"SubspaceNet.set_diff_method: Method {diff_method} is not defined for SubspaceNet"
            )

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):  # If you add fully connected layers later
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, X: torch.Tensor, Rx_tau: torch.Tensor, A: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm
        if self.diff_method == mvdr:
            enhanced = self.diff_method(X, Rz, A)
            return enhanced
        if self.diff_method == music_spectrum:
            inverse_spectrum = self.diff_method(self, Rz, A)
            return inverse_spectrum, Rz # B_F x angles

def mvdr(signals: torch.Tensor,
                           covariances: torch.Tensor,
                           steering_vectors: torch.Tensor) -> torch.Tensor:
    """
    Apply MVDR beamforming on a batch of narrowband signals.
    
    Args:
        signals: Tensor of shape (B, C, T) - narrowband multichannel time-domain signals
        covariances: Tensor of shape (B, C, C) - spatial covariance matrices
        steering_vectors: Tensor of shape (B, C, 1) - steering vectors toward desired DoA
    
    Returns:
        Tensor of shape (B, T) - beamformed signals
    """
    B, C, T = signals.shape
    signals = signals.to(torch.complex64)
    covariances = covariances.to(torch.complex64)
    steering_vectors = steering_vectors.to(torch.complex64)  # (B, C, 1)
    # print(f"signals shape: {signals.shape}, covariances shape: {covariances.shape}, steering_vectors shape: {steering_vectors.shape}")
    
    # Invert covariance matrices (B, C, C)
    R_inv = torch.linalg.pinv(covariances)

    # Compute numerator: (B, C, 1)
    R_inv_a = torch.bmm(R_inv, steering_vectors)
    # print(f"R_inv_a shape: {R_inv_a.shape}")

    # Compute denominator: (B, 1, 1)
    denom = torch.bmm(steering_vectors.conj().transpose(1, 2), R_inv_a)
    denom = denom.real  # Ensure real-valued denominator
    # print(f"denom shape: {denom.shape}")

    # MVDR weights: (B, C, 1)
    w = R_inv_a / denom

    # Apply beamforming: output shape (B, T)
    y = torch.sum(w.conj().transpose(1, 2) @ signals, dim=1)

    return y

def music_spectrum(self, Rz: torch.Tensor, A: torch.Tensor):
    """Applies the MUSIC operation for generating spectrum (batched & vectorized, clean einsum style)

    Args:
        Rz (torch.Tensor): Covariance matrices, shape (B, C, C)
        A (torch.Tensor): Steering vectors, shape (B, C, Q)

    Returns:
        torch.Tensor: Inverse spectrum values, shape (B, Q)
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Rz)  # (B, C, C)
    sorted_idx = torch.argsort(torch.abs(eigenvalues), descending=True)
    sorted_eigvectors = torch.gather(eigenvectors, 2,
                                    sorted_idx.unsqueeze(-1).expand(-1, -1, Rz.shape[-1]).transpose(1, 2))
    Un = sorted_eigvectors[:, :, self.M:]        # (B, C, C-M)

    # Transpose steering vectors
    A_H = torch.conj(A).transpose(1, 2)     # (B, Q, C)

    # Compute projections using einsum
    # var1: (B, Q, C-M)
    var1 = torch.einsum("bqc, bcs -> bqs", A_H, Un)  # (B, Q, C-M)

    # Norm squared
    inverse_spectrum = torch.norm(var1, dim=-1) ** 2  # (B, Q)

    return inverse_spectrum