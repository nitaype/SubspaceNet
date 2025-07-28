'''

This is the code for the validation loop. it has mainly two options:
1. evaluating for speech enhancement task with mvdr and SISNR loss.
2. evaluating for doa estimation task with RMSPE loss.
Also, there is the test function that test both of the options and the classic algorithms.

'''

# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, SISNRLoss, Spectrum_Loss
from src.utils import *
from src.models import SubspaceNet, mvdr, music_spectrum
from pesq import pesq
from pystoi import stoi
import scipy.signal
import wandb
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    figures: dict = None,
    model_type: str = "SubspaceNet",
):
    """
    Evaluate the DNN model on a given dataset.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    data_crop = len(dataset)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= data_crop:
                break
            # ---- validation for speech enhancement training ----
            if isinstance(criterion, SISNRLoss):
                noisy_stft, R, clean, steering = data
                noisy_stft = noisy_stft.to(device)
                R = R.to(device).to(dtype=torch.float32)
                clean = clean.to(device)
                steering = steering.to(device)
                # stft shape: (B, C, T, F)
                # clean shape: (B, T)
                # steering shape: (B, C, 1, F)
                # R shape: (B, tau, 2C, C, F)
                # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
                B, tau, CC, C, F = R.shape
                _, _, T, _ = noisy_stft.shape

                x = noisy_stft.permute(0, 3, 1, 2).reshape(B * F, C, T)             # (B*F, C, T)
                A = steering.permute(0, 3, 1, 2).reshape(B * F, C, 1)               # (B*F, C, 1)
                Rx = R.permute(0, 4, 1, 2, 3).reshape(B * F, tau, 2 * C, C)  # (B*F, tau, 2C, C)
                filtered_signal = model(x, Rx, A)   # (B*F, T)

                filtered_stft = filtered_signal.view(B, F, T)
                enhanced = torch.istft(
                    filtered_stft, 
                    n_fft=512, 
                    hop_length=256, 
                    win_length=512,
                    return_complex=False
                )  # shape: (B, T)
                eval_loss = criterion(enhanced, clean)

            # ---- validation for DOA training ----
            if isinstance(criterion, Spectrum_Loss):
                noisy_stft, R, clean, steering, doa = data
                noisy_stft = noisy_stft.to(device)
                R = R.to(device).to(dtype=torch.float32)
                clean = clean.to(device)
                steering = steering.to(device)
                doa = doa.to(device)
                doa = doa.squeeze(1)
                # stft shape: (B, C, T, F)
                # clean shape: (B, T)
                # steering shape: (B, C, Q, F)
                # R shape: (B, tau, 2C, C, F)
                # doa shape: (B, M=2)
                # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
                B, tau, CC, C, F = R.shape
                _, _, Q, _ = steering.shape
                _, _, T, _ = noisy_stft.shape
                _, M = doa.shape
                # Reshape inputs for vectorized model execution
                x = noisy_stft.permute(0, 3, 1, 2).reshape(B * F, C, T)             # (B*F, C, T)
                A = steering.permute(0, 3, 1, 2).reshape(B * F, C, Q)               # (B*F, C, Q)
                Rx = R.permute(0, 4, 1, 2, 3).reshape(B * F, tau, 2 * C, C)  # (B*F, tau, 2C, C)

                inverse_spectrum, R = model(x, Rx, A)   # (B*F, Q)
                inverse_spectrum = inverse_spectrum.view(B, F, -1)  # (B, F, Q)
                epsilon = 0
                spectrum_per_f = 1 / (inverse_spectrum + epsilon)
                # print(spectrum_per_f.shape)
                spectrum = spectrum_per_f.sum(dim=1)

                angle_grid = torch.linspace(-np.pi / 2, np.pi / 2, Q, device=inverse_spectrum.device)

                peaks = torch.zeros(B, M, dtype=torch.int64, device=spectrum.device)
                # print(spectrum.shape)
                for batch in range(B):
                    music_spectrum = spectrum[batch].cpu().detach().numpy().reshape(-1)
                    # print(music_spectrum.shape)
                    # Find local maxima
                    peaks_tmp = scipy.signal.find_peaks(music_spectrum)[0]

                    if len(peaks_tmp) < M:
                        warnings.warn(f"Not enough peaks found! Filling with top values instead.")
                        # Fill missing peaks using top global spectrum indices
                        extra_indices = torch.topk(torch.from_numpy(music_spectrum), M - len(peaks_tmp), largest=True).indices.cpu().detach().numpy()
                        peaks_tmp = np.concatenate((peaks_tmp, extra_indices))

                    # Sort peaks by value descending
                    sorted_peaks = peaks_tmp[np.argsort(music_spectrum[peaks_tmp])[::-1]]
                    peaks[batch] = torch.from_numpy(sorted_peaks[:M]).to(spectrum.device)

                # Map indices to angles
                angle_grid = torch.linspace(-torch.pi / 2, torch.pi / 2, Q, device=spectrum.device)
                doa_predictions = angle_grid[peaks]  # (B, M)

                eval_criterion = RMSPELoss()
                eval_loss = eval_criterion(doa_predictions, doa)

            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / data_crop
    return overall_loss

def test_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    model_type: str = "SubspaceNet",
    wandb_name: str = "test"
):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        The test for both doa estimation and speech enhancement tasks
    """

    wandb.init(project="SubspaceNet", name=wandb_name)

    # ------ test only for speech enhancement task -----
    if isinstance(criterion, SISNRLoss):
        # Initialize values
        overall_loss = 0.0
        test_length = 0
        stoi_noisy = np.array([])
        si_sdr_noisy = np.array([])
        pesq_noisy = np.array([])
        stoi_enhanced = np.array([])
        si_sdr_enhanced = np.array([])
        pesq_enhanced = np.array([])
        # Set model to eval mode
        model.eval()
        # Gradients calculation isn't required for evaluation
        print("start testing")
        for i, data in enumerate(dataset):
            noisy_stft, R, clean, steering = data
            noisy_stft = noisy_stft.to(device)
            R = R.to(device).to(dtype=torch.float32)
            clean = clean.to(device)
            steering = steering.to(device)
            # stft shape: (B, C, T, F)
            # clean shape: (B, T)
            # steering shape: (B, C, 1, F)
            # R shape: (B, tau, 2C, C, F)
            # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
            B, tau, CC, C, F = R.shape
            _, _, T, _ = noisy_stft.shape

            x = noisy_stft.permute(0, 3, 1, 2).reshape(B * F, C, T)             # (B*F, C, T)
            A = steering.permute(0, 3, 1, 2).reshape(B * F, C, 1)               # (B*F, C, 1)
            Rx = R.permute(0, 4, 1, 2, 3).reshape(B * F, tau, 2 * C, C)  # (B*F, tau, 2C, C)
            filtered_signal = model(x, Rx, A)   # (B*F, T)

            filtered_stft = filtered_signal.view(B, F, T)
            enhanced = torch.istft(
                filtered_stft, 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                return_complex=False
            )  # shape: (B, T)
            
            noisy = torch.istft(
                noisy_stft[:, 0, :, :].permute(0, 2, 1), 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                return_complex=False
            )  # shape: (B, T)

            enhanced = enhanced/enhanced.max()
            clean = clean/clean.max()
            noisy = noisy/noisy.max()

            s1 = clean.squeeze(0).detach().cpu()
            s_hat = enhanced.squeeze(0).detach().cpu()
            y = noisy.squeeze(0).detach().cpu()

            # Convert only for functions that need numpy
            s1_np = s1.cpu().numpy()
            y_np = y.cpu().numpy()
            s_hat_np = s_hat.cpu().numpy()

            stoi_noisy = np.append(stoi_noisy, stoi(s1_np, y_np, 16000, extended=False))
            si_sdr_noisy = np.append(si_sdr_noisy, -criterion(s1.unsqueeze(0), y.unsqueeze(0)))
            pesq_noisy = np.append(pesq_noisy, pesq(16000, s1_np, y_np, mode="wb"))

            stoi_enhanced = np.append(stoi_enhanced, stoi(s1_np, s_hat_np, 16000, extended=False))
            si_sdr_enhanced = np.append(si_sdr_enhanced, -criterion(s1.unsqueeze(0), s_hat.unsqueeze(0)))
            pesq_enhanced = np.append(pesq_enhanced, pesq(16000, s1_np, s_hat_np, mode="wb"))


            if i % 20 == 19:  # print every 20 iterations
                print(i)
        
        wandb.log({
            "test/stoi_noisy": float(stoi_noisy.mean()),
            "test/pesq_noisy": float(pesq_noisy.mean()),
            "test/si_sdr_noisy": float(si_sdr_noisy.mean()),
            "test/stoi_enhanced": float(stoi_enhanced.mean()),
            "test/pesq_enhanced": float(pesq_enhanced.mean()),
            "test/si_sdr_enhanced": float(si_sdr_enhanced.mean()),
        })

        wandb.log({
            "audio/clean": wandb.Audio(s1.numpy(), sample_rate=16000),
            "audio/enhanced": wandb.Audio(s_hat.numpy(), sample_rate=16000),
            "audio/noisy": wandb.Audio(y.numpy(), sample_rate=16000),
        })
        wandb.finish()

        # print(f"Reference metrics for distorted speech at {snr_dbs[0]}dB are\n")
        print(f"STOI: {stoi_noisy.mean()}")
        print(f"PESQ: {pesq_noisy.mean()}")
        print(f"SI-SDR: {si_sdr_noisy.mean()}")
        print(f"STOI: {stoi_enhanced.mean()}")
        print(f"PESQ: {pesq_enhanced.mean()}")
        print(f"SI-SDR: {si_sdr_enhanced.mean()}")
        plt.show()
        return 

    # ----- testing both doa estimation and mvdr speech enhancement -----
    if isinstance(criterion, Spectrum_Loss):
        overall_loss = 0.0
        test_length = 0
        stoi_noisy = np.array([])
        si_sdr_noisy = np.array([])
        pesq_noisy = np.array([])
        stoi_enhanced = np.array([])
        si_sdr_enhanced = np.array([])
        pesq_enhanced = np.array([])
        stoi_enhanced_classic = np.array([])
        si_sdr_enhanced_classic = np.array([])
        pesq_enhanced_classic = np.array([])
        model.eval()
        all_rmspe_losses = []
        all_rmspe_losses_classic = []

        print("start testing")
        for i, data in enumerate(dataset):
            noisy_stft, R, clean, steering, doa = data
            noisy_stft = noisy_stft.to(device)
            R = R.to(device).to(dtype=torch.float32)
            clean = clean.to(device)
            steering = steering.to(device)
            doa = doa.to(device)
            doa = doa.squeeze(1)
            # stft shape: (B, C, T, F)
            # clean shape: (B, T)
            # steering shape: (B, C, Q, F)
            # R shape: (B, tau, 2C, C, F)
            # doa shape: (B, M=2)
            # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
            B, tau, CC, C, F = R.shape
            _, _, Q, _ = steering.shape
            _, _, T, _ = noisy_stft.shape
            _, M = doa.shape

            # Reshape inputs for vectorized model execution
            x = noisy_stft.permute(0, 3, 1, 2).reshape(B * F, C, T)             # (B*F, C, T)
            A = steering.permute(0, 3, 1, 2).reshape(B * F, C, Q)               # (B*F, C, Q)
            Rx = R.permute(0, 4, 1, 2, 3).reshape(B * F, tau, 2 * C, C)  # (B*F, tau, 2C, C)

            inverse_spectrum, R = model(x, Rx, A)   # (B*F, Q)
            inverse_spectrum = inverse_spectrum.view(B, F, -1)  # (B, F, Q)
            epsilon = 0
            spectrum_per_f = 1 / (inverse_spectrum + epsilon)
            spectrum = spectrum_per_f.sum(dim=1)

            sample_autocorrelation = Rx[:, 0, :, :]
            sample_autocorrelation_real = sample_autocorrelation[:, :C, :]
            sample_autocorrelation_img = sample_autocorrelation[:, C:, :]
            sample_autocorrelation = torch.complex(sample_autocorrelation_real, sample_autocorrelation_img)  # Shape: [B*F, C, C])

            Rx_classic = sample_autocorrelation

            inverse_spectrum_classic = spectrum_calc(M, Rx_classic, A)
            inverse_spectrum_classic = inverse_spectrum_classic.view(B, F, -1)  # (B, F, Q)
            spectrum_per_f_classic = 1 / (inverse_spectrum_classic + epsilon)
            spectrum_classic = spectrum_per_f_classic.sum(dim=1)

            # ----- mvdr test - model and classic -----
            angle_grid = torch.linspace(-np.pi / 2, np.pi / 2, Q, device=inverse_spectrum.device)
            doa_expanded = doa.unsqueeze(1).expand(B, F, M)  # shape: (B, F, M)
            doa = doa_expanded.reshape(B * F, M)    # shape: (B*F, M)

            first_doa = doa[:, 0]  # shape: (B*F,)
            diff = torch.abs(first_doa.unsqueeze(1) - angle_grid.unsqueeze(0))
            doa_indices = torch.argmin(diff, dim=1)  # shape: (B*F,)

            BF, _, _ = A.shape
            doa_indices_exp = doa_indices.view(BF, 1).expand(-1, C)  # (B*F, C)
            V = torch.gather(A, dim=2, index=doa_indices_exp.unsqueeze(2))  # (B*F, C, 1)
            
            filtered_signal = mvdr(x, R, V)
            filtered_signal_classic = mvdr(x, Rx_classic, V)

            filtered_stft = filtered_signal.view(B, F, T)
            filtered_stft_classic = filtered_signal_classic.view(B, F, T)

            enhanced = torch.istft(
                filtered_stft, 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                return_complex=False
            )  # shape: (B, T)
            enhanced_classic = torch.istft(
                filtered_stft_classic, 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                return_complex=False
            )  # shape: (B, T)
            
            noisy = torch.istft(
                noisy_stft[:, 0, :, :].permute(0, 2, 1), 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                return_complex=False
            )  # shape: (B, T)

            enhanced = enhanced/enhanced.max()
            enhanced_classic = enhanced_classic/enhanced_classic.max()
            clean = clean/clean.max()
            noisy = noisy/noisy.max()

            # print(enhanced.shape, clean.shape, noisy.shape)

            min_len = min(enhanced.shape[-1], enhanced_classic.shape[-1], clean.shape[-1], noisy.shape[-1])

            # Trim all signals to the same length
            enhanced = enhanced[..., :min_len]
            enhanced_classic = enhanced_classic[..., :min_len]
            clean = clean[..., :min_len]
            noisy = noisy[..., :min_len]

            s1 = clean.squeeze(0).detach().cpu()
            s_hat = enhanced.squeeze(0).detach().cpu()
            s_hat_classic = enhanced_classic.squeeze(0).detach().cpu()
            y = noisy.squeeze(0).detach().cpu()

            # Convert only for functions that need numpy
            s1_np = s1.cpu().numpy()
            y_np = y.cpu().numpy()
            s_hat_np = s_hat.cpu().numpy()
            s_hat_np_classic = s_hat_classic.cpu().numpy()

            loss = SISNRLoss()

            stoi_noisy = np.append(stoi_noisy, stoi(s1_np, y_np, 16000, extended=False))
            si_sdr_noisy = np.append(si_sdr_noisy, -loss(s1.unsqueeze(0), y.unsqueeze(0)))
            pesq_noisy = np.append(pesq_noisy, pesq(16000, s1_np, y_np, mode="wb"))

            stoi_enhanced = np.append(stoi_enhanced, stoi(s1_np, s_hat_np, 16000, extended=False))
            si_sdr_enhanced = np.append(si_sdr_enhanced, -loss(s1.unsqueeze(0), s_hat.unsqueeze(0)))
            pesq_enhanced = np.append(pesq_enhanced, pesq(16000, s1_np, s_hat_np, mode="wb"))

            stoi_enhanced_classic = np.append(stoi_enhanced_classic, stoi(s1_np, s_hat_np_classic, 16000, extended=False))
            si_sdr_enhanced_classic = np.append(si_sdr_enhanced_classic, -loss(s1.unsqueeze(0), s_hat_classic.unsqueeze(0)))
            pesq_enhanced_classic = np.append(pesq_enhanced_classic, pesq(16000, s1_np, s_hat_np_classic, mode="wb"))

            if i % 20 == 19:  # print every 20 iterations
                print(i)

            peaks = torch.zeros(B, M, dtype=torch.int64, device=spectrum.device)
            # print(spectrum.shape)
            for batch in range(B):
                music_spectrum = spectrum[batch].cpu().detach().numpy().reshape(-1)
                # print(music_spectrum.shape)
                # Find local maxima
                peaks_tmp = scipy.signal.find_peaks(music_spectrum)[0]

                if len(peaks_tmp) < M:
                    warnings.warn(f"Not enough peaks found! Filling with top values instead.")
                    # Fill missing peaks using top global spectrum indices
                    extra_indices = torch.topk(torch.from_numpy(music_spectrum), M - len(peaks_tmp), largest=True).indices.cpu().detach().numpy()
                    peaks_tmp = np.concatenate((peaks_tmp, extra_indices))

                # Sort peaks by value descending
                sorted_peaks = peaks_tmp[np.argsort(music_spectrum[peaks_tmp])[::-1]]
                peaks[batch] = torch.from_numpy(sorted_peaks[:M]).to(spectrum.device)

            # Map indices to angles
            angle_grid = torch.linspace(-torch.pi / 2, torch.pi / 2, Q, device=spectrum.device)
            doa_predictions = angle_grid[peaks]  # (B, M)

            peaks = torch.zeros(B, M, dtype=torch.int64, device=spectrum.device)
            # print(spectrum.shape)
            for batch in range(B):
                music_spectrum = spectrum_classic[batch].cpu().detach().numpy().reshape(-1)
                # print(music_spectrum.shape)
                # Find local maxima
                peaks_tmp = scipy.signal.find_peaks(music_spectrum)[0]

                if len(peaks_tmp) < M:
                    warnings.warn(f"Not enough peaks found! Filling with top values instead.")
                    # Fill missing peaks using top global spectrum indices
                    extra_indices = torch.topk(torch.from_numpy(music_spectrum), M - len(peaks_tmp), largest=True).indices.cpu().detach().numpy()
                    peaks_tmp = np.concatenate((peaks_tmp, extra_indices))

                # Sort peaks by value descending
                sorted_peaks = peaks_tmp[np.argsort(music_spectrum[peaks_tmp])[::-1]]
                peaks[batch] = torch.from_numpy(sorted_peaks[:M]).to(spectrum_classic.device)

            # Map indices to angles
            angle_grid = torch.linspace(-torch.pi / 2, torch.pi / 2, Q, device=spectrum_classic.device)
            doa_predictions_classic = angle_grid[peaks]  # (B, M)

            eval_criterion = RMSPELoss()
            eval_loss = eval_criterion(doa_predictions, doa)
            eval_loss_classic = eval_criterion(doa_predictions_classic, doa)

            all_rmspe_losses.append(eval_loss.item())
            all_rmspe_losses_classic.append(eval_loss_classic.item())

    # Compute mean across all test samples
    mean_rmspe = sum(all_rmspe_losses) / len(all_rmspe_losses)
    mean_rmspe_classic = sum(all_rmspe_losses_classic) / len(all_rmspe_losses_classic)

    # Log to wandb
    wandb.log({"test/mean_eval_RMSPE_model": mean_rmspe})
    wandb.log({"test/mean_eval_RMSPE_classic": mean_rmspe_classic})

    print("Mean RMSPE for model:", mean_rmspe)
    print("Mean RMSPE for classic:", mean_rmspe_classic)

    wandb.log({
        "test/stoi_noisy": float(stoi_noisy.mean()),
        "test/pesq_noisy": float(pesq_noisy.mean()),
        "test/si_sdr_noisy": float(si_sdr_noisy.mean()),
        "test/stoi_enhanced": float(stoi_enhanced.mean()),
        "test/pesq_enhanced": float(pesq_enhanced.mean()),
        "test/si_sdr_enhanced": float(si_sdr_enhanced.mean()),
        "test/stoi_enhanced_classic": float(stoi_enhanced_classic.mean()),
        "test/pesq_enhanced_classic": float(pesq_enhanced_classic.mean()),
        "test/si_sdr_enhanced_classic": float(si_sdr_enhanced_classic.mean())
    })

    wandb.log({
        "audio/clean": wandb.Audio(s1.numpy(), sample_rate=16000),
        "audio/enhanced": wandb.Audio(s_hat.numpy(), sample_rate=16000),
        "audio/enhanced_classic": wandb.Audio(s_hat_classic.numpy(), sample_rate=16000),
        "audio/noisy": wandb.Audio(y.numpy(), sample_rate=16000),
    })
    wandb.finish()

    # print(f"Reference metrics for distorted speech at {snr_dbs[0]}dB are\n")
    print("noisy")
    print(f"STOI: {stoi_noisy.mean()}")
    print(f"PESQ: {pesq_noisy.mean()}")
    print(f"SI-SDR: {si_sdr_noisy.mean()}")

    print("enhanced-model")
    print(f"STOI: {stoi_enhanced.mean()}")
    print(f"PESQ: {pesq_enhanced.mean()}")
    print(f"SI-SDR: {si_sdr_enhanced.mean()}")

    print("enhanced-classic")
    print(f"STOI: {stoi_enhanced_classic.mean()}")
    print(f"PESQ: {pesq_enhanced_classic.mean()}")
    print(f"SI-SDR: {si_sdr_enhanced_classic.mean()}")

def spectrum_calc(M, Rz: torch.Tensor, A: torch.Tensor):
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
    Un = sorted_eigvectors[:, :, M:]        # (B, C, C-M)

    # Hermitian (conjugate transpose)
    Un_H = torch.conj(Un).transpose(1, 2)   # (B, C-M, C)

    # Transpose steering vectors
    A_H = torch.conj(A).transpose(1, 2)     # (B, Q, C)

    # Compute projections using einsum
    # var1: (B, Q, C-M)
    # var1 = chunked_einsum(A_H, Un, chunk_size=50)  # you can adjust chunk_size
    var1 = torch.einsum("bqc, bcs -> bqs", A_H, Un)  # (B, Q, C-M)

    # Norm squared
    inverse_spectrum = torch.norm(var1, dim=-1) ** 2  # (B, Q)

    return inverse_spectrum