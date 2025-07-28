"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, MSPELoss, SISNRLoss, Spectrum_Loss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *
from src.models import SubspaceNet, mvdr, music_spectrum
from src.plotting import plot_spectrum
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
                
                # noisy_speech, R, clean, steering, doa = data
                # x = noisy_speech.to(device)
                # Rx = R.to(device).to(dtype=torch.float32)
                # clean = clean.to(device)
                # A = steering.to(device)
                # doa = doa.to(device) # radians
                # doa = doa.squeeze(1)
                # # noisy shape: (B, C, T)
                # # clean shape: (B, T)
                # # steering shape: (B, C, Q)
                # # R shape: (B, tau, 2C, C)
                # # doa shape: (B, M)
                # # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
                # B, tau, CC, C = R.shape
                # _, _, Q = steering.shape
                # _, _, T = x.shape
                # M = 1

                inverse_spectrum, R = model(x, Rx, A)   # (B*F, Q)
                inverse_spectrum = inverse_spectrum.view(B, F, -1)  # (B, F, Q)
                epsilon = 0
                spectrum_per_f = 1 / (inverse_spectrum + epsilon)
                # print(spectrum_per_f.shape)
                spectrum = spectrum_per_f.sum(dim=1)

                # inverse_spectrum = inverse_spectrum_f.sum(dim=1)
                # spectrum_per = 1 / (inverse_spectrum + epsilon)

                # spectrum = spectrum_per_f
                angle_grid = torch.linspace(-np.pi / 2, np.pi / 2, Q, device=inverse_spectrum.device)
                # print(angle_grid.shape)
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
                # print(f"doa hat: {doa_predictions}")
                # print(f"doa: {doa}")

                # B, M = doa.shape
                # F = inverse_spectrum.shape[0] // B

                # # Expand DOA
                # doa_expanded = doa.unsqueeze(1).expand(B, F, M)  # shape: (B, F, M)
                # doa = doa_expanded.reshape(B * F, M)    # shape: (B*F, M)

                eval_criterion = RMSPELoss()
                eval_loss = eval_criterion(doa_predictions, doa)
                # b = 0
                # spec_np = spectrum[b].cpu().detach().numpy()
                # angle_grid_deg = angle_grid.cpu().detach().numpy() * 180 / np.pi
                # pred_deg = doa_predictions[b].cpu().detach().numpy() * 180 / np.pi
                # true_deg = doa[b].cpu().detach().numpy() * 180 / np.pi

                # import matplotlib.pyplot as plt

                # plt.figure(figsize=(10, 5))
                # plt.plot(angle_grid_deg, spec_np, label="Spectrum")

                # # Predicted DOAs
                # for pd in pred_deg:
                #     plt.axvline(pd, color="r", linestyle="--", label="Pred DOA")

                # # True DOAs
                # for td in true_deg:
                #     plt.axvline(td, color="g", linestyle=":", label="True DOA")

                # plt.xlabel("Angle (deg)")
                # plt.ylabel("Summed MUSIC spectrum")
                # plt.title("Spectrum for first sample in batch")
                # plt.legend(loc="upper right")
                # plt.grid(True)
                # plt.show()

            # if model_type.startswith("DA-MUSIC"):
            #     # Deep Augmented MUSIC
            #     DOA_predictions = model_output
            # elif model_type.startswith("DeepCNN"):
            #     # Deep CNN
            #     if isinstance(criterion, nn.BCELoss):
            #         # If evaluation performed over validation set, loss is BCE
            #         DOA_predictions = model_output
            #         # find peaks in the pseudo spectrum of probabilities
            #         DOA_predictions = (
            #             get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
            #         )
            #         DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
            #     elif isinstance(criterion, [RMSPELoss, MSPELoss]):
            #         # If evaluation performed over testset, loss is RMSPE / MSPE
            #         DOA_predictions = model_output
            #     else:
            #         raise Exception(
            #             f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
            #         )
            # if model_type.startswith("SubspaceNet"):
            #     # Default - SubSpaceNet
            #     filtered_signal = enhanced
            # else:
            #     raise Exception(
            #         f"evaluate_dnn_model: Model type {model_type} is not defined"
            #     )
            # # Compute prediction loss
            # if model_type.startswith("DeepCNN") and isinstance(criterion, RMSPELoss):
            #     eval_loss = criterion(DOA_predictions.float(), DOA.float())
            # else:
            #     eval_loss = criterion(filtered_signal, clean)

            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / data_crop
    # Plot spectrum for SubspaceNet model
    # if plot_spec and model_type.startswith("SubspaceNet"):
    #     DOA_all = model_output[1]
    #     roots = model_output[2]
    #     plot_spectrum(
    #         predictions=DOA_all * R2D,
    #         true_DOA=DOA[0] * R2D,
    #         roots=roots,
    #         algorithm="SubNet+R-MUSIC",
    #         figures=figures,
    #     )
    return overall_loss


def evaluate_augmented_model(
    model: SubspaceNet,
    dataset,
    system_model,
    criterion=RMSPE,
    algorithm: str = "music",
    plot_spec: bool = False,
    figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model),
        "esprit": Esprit(system_model),
        "r-music": RootMUSIC(system_model),
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, DOA = data
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                predictions, M = method_output[0], method_output[-1]
                # If the amount of predictions is less than the amount of sources
                predictions = add_random_predictions(M, predictions, algorithm)
                # Calculate loss criterion
                loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
    return np.mean(hybrid_loss)


def evaluate_model_based(
    dataset: list,
    system_model,
    criterion: RMSPE,
    plot_spec=False,
    algorithm: str = "music",
    figures: dict = None,
):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
        float: The average evaluation loss.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    loss_list = []
    for i, data in enumerate(dataset):
        X, doa = data
        X = X[0]
        # Root-MUSIC algorithms
        if "r-music" in algorithm:
            root_music = RootMUSIC(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            else:
                # Conventional
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="sample"
                )
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions_all,
                    true_DOA=doa[0] * R2D,
                    roots=roots,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        # MUSIC algorithms
        elif "music" in algorithm:
            music = MUSIC(system_model)
            if algorithm.startswith("bb"):
                # Broadband MUSIC
                predictions, spectrum, M = music.broadband(X=X)
            elif algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, spectrum, M = music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            elif algorithm.startswith("music"):
                # Conventional
                predictions, spectrum, M = music.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )

        # ESPRIT algorithms
        elif "esprit" in algorithm:
            esprit = Esprit(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
            else:
                # Conventional
                predictions, M = esprit.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

        # MVDR algorithm
        elif algorithm.startswith("mvdr"):
            mvdr = MVDR(system_model)
            # Conventional
            _, spectrum = mvdr.narrowband(X=X, mode="sample")
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=None,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        else:
            raise Exception(
                f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
            )
    return np.mean(loss_list)


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions

def test_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    figures: dict = None,
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
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    wandb.init(project="SubspaceNet", name=wandb_name)
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
                print(f"STOI: {stoi_noisy.mean()}")
                print(f"PESQ: {pesq_noisy.mean()}")
                print(f"SI-SDR: {si_sdr_noisy.mean()}")
                print(f"STOI: {stoi_enhanced.mean()}")
                print(f"PESQ: {pesq_enhanced.mean()}")
                print(f"SI-SDR: {si_sdr_enhanced.mean()}")
        
            wandb.log({
                "test/step": i,
                "test/stoi_noisy": float(stoi_noisy.mean()),
                "test/pesq_noisy": float(pesq_noisy.mean()),
                "test/si_sdr_noisy": float(si_sdr_noisy.mean()),
                "test/stoi_enhanced": float(stoi_enhanced.mean()),
                "test/pesq_enhanced": float(pesq_enhanced.mean()),
                "test/si_sdr_enhanced": float(si_sdr_enhanced.mean())
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
        # Gradients calculation isn't required for evaluation
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
            # M = 1
            # Reshape inputs for vectorized model execution
            x = noisy_stft.permute(0, 3, 1, 2).reshape(B * F, C, T)             # (B*F, C, T)
            A = steering.permute(0, 3, 1, 2).reshape(B * F, C, Q)               # (B*F, C, Q)
            Rx = R.permute(0, 4, 1, 2, 3).reshape(B * F, tau, 2 * C, C)  # (B*F, tau, 2C, C)

            # ---- for sine wave data ----
            # noisy_speech, R, clean, steering, doa = data
            # x = noisy_speech.to(device)
            # Rx = R.to(device).to(dtype=torch.float32)
            # clean = clean.to(device)
            # steering = steering.to(device)
            # doa = doa.to(device) # radians
            # doa = doa.squeeze(1)
            # # noisy shape: (B, C, T)
            # # clean shape: (B, T)
            # # steering shape: (B, C, Q)
            # # R shape: (B, tau, 2C, C)
            # # doa shape: (B, M)
            # # print(noisy_stft.shape, R.shape, clean.shape, steering.shape)
            # B, tau, CC, C = R.shape
            # _, _, Q = steering.shape
            # _, _, T = x.shape

            inverse_spectrum, R = model(x, Rx, A)   # (B*F, Q)
            inverse_spectrum = inverse_spectrum.view(B, F, -1)  # (B, F, Q)
            epsilon = 0
            spectrum_per_f = 1 / (inverse_spectrum + epsilon)
            # print(spectrum_per_f.shape)
            spectrum = spectrum_per_f.sum(dim=1)

            sample_autocorrelation = Rx[:, 0, :, :]
            sample_autocorrelation_real = sample_autocorrelation[:, :C, :]
            sample_autocorrelation_img = sample_autocorrelation[:, C:, :]
            sample_autocorrelation = torch.complex(sample_autocorrelation_real, sample_autocorrelation_img)  # Shape: [B*F, C, C])

            # ---- diagonal overloading methods ----
            # Rx_classic = gram_diagonal_overload(
            #     Kx=sample_autocorrelation, eps=1
            # )

            # Kx = sample_autocorrelation
            # eps = 0.01
            # if not isinstance(Kx, torch.Tensor):
            #     Kx = torch.tensor(Kx)
            # Kx = Kx.to(device)

            # # Add epsilon to diagonal
            # eps_addition = eps * torch.eye(Kx.shape[-1], device=Kx.device).unsqueeze(0)  # (1, N, N)
            # Kx_Out = Kx + eps_addition
            # Rx_classic = Kx_Out

            Rx_classic = sample_autocorrelation

            inverse_spectrum_classic = spectrum_calc(M, Rx_classic, A)
            inverse_spectrum_classic = inverse_spectrum_classic.view(B, F, -1)  # (B, F, Q)
            spectrum_per_f_classic = 1 / (inverse_spectrum_classic + epsilon)
            spectrum_classic = spectrum_per_f_classic.sum(dim=1)

            # inverse_spectrum = inverse_spectrum_f.sum(dim=1)
            # spectrum_per = 1 / (inverse_spectrum + epsilon)

            # spectrum = spectrum_per_f

            # ----- mvdr test -----
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
                # print(f"STOI: {stoi_noisy.mean()}")
                # print(f"PESQ: {pesq_noisy.mean()}")
                # print(f"SI-SDR: {si_sdr_noisy.mean()}")
                # print(f"STOI: {stoi_enhanced.mean()}")
                # print(f"PESQ: {pesq_enhanced.mean()}")
                # print(f"SI-SDR: {si_sdr_enhanced.mean()}")
        
        #     wandb.log({
        #         "test/step": i,
        #         "test/stoi_noisy": float(stoi_noisy.mean()),
        #         "test/pesq_noisy": float(pesq_noisy.mean()),
        #         "test/si_sdr_noisy": float(si_sdr_noisy.mean()),
        #         "test/stoi_enhanced": float(stoi_enhanced.mean()),
        #         "test/pesq_enhanced": float(pesq_enhanced.mean()),
        #         "test/si_sdr_enhanced": float(si_sdr_enhanced.mean())
        #     })

            # print(angle_grid.shape)
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
            # print(f"doa hat: {doa_predictions}")
            # print(f"doa: {doa}")

            # B, M = doa.shape
            # F = inverse_spectrum.shape[0] // B

            # # Expand DOA
            # doa_expanded = doa.unsqueeze(1).expand(B, F, M)  # shape: (B, F, M)
            # doa = doa_expanded.reshape(B * F, M)    # shape: (B*F, M)

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

            # if i % 20 == 19:  # print every 20 iterations
            #     print("model loss is:", eval_loss.item())
            #     print("classic loss is:", eval_loss_classic.item())

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

def evaluate(
    model: nn.Module,
    model_type: str,
    model_test_dataset: list,
    criterion: nn.Module,
    subspace_criterion,
    figures: dict,
    plot_spec: bool = True,
    augmented_methods: list = None,
    subspace_methods: list = None,
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        model (nn.Module): The DNN model.
        model_type (str): Type of the model.
        model_test_dataset (list): Test dataset for the model.
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        subspace_criterion: Loss criterion for subspace method evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    # Set default methods for SubspaceNet augmentation
    # if not isinstance(augmented_methods, list) and model_type.startswith("SubspaceNet"):
    #     augmented_methods = [
    #         # "mvdr",
    #         "r-music",
    #         "esprit",
    #         # "music",
    #     ]
    # # Set default model-based subspace methods
    # if not isinstance(subspace_methods, list):
    #     subspace_methods = [
    #         "esprit",
    #         # "music",
    #         "r-music",
    #         # "mvdr",
    #         # "sps-r-music",
    #         # "sps-esprit",
    #         # "sps-music"
    #         # "bb-music",
    #     ]
    # Evaluate SubspaceNet + differentiable algorithm performances
    print("start testing")
    model_test_loss = evaluate_dnn_model(
        model=model,
        dataset=model_test_dataset,
        criterion=criterion,
        plot_spec=plot_spec,
        figures=figures,
        model_type=model_type,
    )
    print(f"{model_type} Test loss = {model_test_loss}")
    # Evaluate SubspaceNet augmented methods
    # for algorithm in augmented_methods:
    #     loss = evaluate_augmented_model(
    #         model=model,
    #         dataset=model_test_dataset,
    #         system_model=system_model,
    #         criterion=subspace_criterion,
    #         algorithm=algorithm,
    #         plot_spec=plot_spec,
    #         figures=figures,
    #     )
    #     print("augmented {} test loss = {}".format(algorithm, loss))
    # # Evaluate classical subspace methods
    # for algorithm in subspace_methods:
    #     loss = evaluate_model_based(
    #         generic_test_dataset,
    #         system_model,
    #         criterion=subspace_criterion,
    #         plot_spec=plot_spec,
    #         algorithm=algorithm,
    #         figures=figures,
    #     )
    #     print("{} test loss = {}".format(algorithm.lower(), loss))

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