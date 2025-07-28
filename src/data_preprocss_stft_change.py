'''

This code preprocss the data from mic signals to stft samples. 
Alse generates the autocorrelation matrix for the model input for each frequency of the stft.
There is also control over how many bins the stft has.

'''

import torch
import soundfile as sf
import scipy.io
from pathlib import Path
import os
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing as mp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# print(os.cpu_count())

def pad_to_length(signal, length):
    current_length = signal.shape[-1]
    if current_length >= length:
        return signal[..., :length]
    pad_width = length - current_length
    return torch.nn.functional.pad(signal, (0, pad_width), value=0)

def read_and_process_sample(duration, tau, sample_dir, sample_rate=16000, max_length_sec=6):
    sample_number = int(sample_dir.name.split("_")[-1])
    suffix = f"{sample_number:05d}"

    array_file = sample_dir / f"p.wav"
    direct_file = sample_dir / f"pDirect.wav"
    mat_file = sample_dir / f"phi.mat"

    if not (array_file.exists() and direct_file.exists() and mat_file.exists()):
        print(f"Missing files for {sample_dir.name}")
        return None

    noisy, _ = sf.read(array_file)  # (T, C)
    clean, _ = sf.read(direct_file)  # (T, C)

    dur = sample_rate * duration    # duration x sec
    dur = int(round(dur / 256) * 256)    # round for 256 multiple
    start = len(noisy) // 2    # start from the middle
    end = start + dur

    noisy = noisy[start:end, :]
    clean = clean[start:end, :]

    # V_k = scipy.io.loadmat(mat_file)["V"] # (C, F, Q)
    doa = scipy.io.loadmat(mat_file)["phi_pair"] # (1, 2)
    noisy_speech = torch.tensor(noisy.T, dtype=torch.float32)  # (C, T)
    clean_speech = torch.tensor(clean.T, dtype=torch.float32).unsqueeze(0)  # (C, T)
    # V_k = torch.tensor(V_k, dtype=torch.torch.complex64)
    doa = torch.tensor(doa, dtype=torch.float32)

    # only for 2 speakers dataset
    # C, F_full = V_k.shape
    # F_pos = F_full // 2 + 1
    # V_k_pos = V_k[:, :F_pos]  # Shape: [C, F]
    # V_k = V_k_pos.unsqueeze(1)  # [C, 1, F]

    # V_k = V_k.permute(0, 2, 1)  # from (C, F, Q) → (C, Q, F)
    # V_k = V_k[..., ::2] # reduce from 257 to 129

    # print(noisy.shape, clean.shape, V_k.shape)
    # print("done reading", datetime.now())

    max_length = sample_rate * max_length_sec
    #first = (clean_speech[0, :].abs() > 1e-3).nonzero()[0][0].item()
    #last = first + max_length
    #noisy_speech = pad_to_length(noisy_speech[:, first:last], max_length)
    #clean_speech = pad_to_length(clean_speech[0, first:last], max_length)
    # trim_start = int(sample_rate * 1.5)
    # trim_end = trim_start + 62*256
    # noisy_speech = noisy_speech[:, trim_start:trim_end]
    # clean_speech = clean_speech[:, trim_start:trim_end]
    noisy_speech = noisy_speech.to(device)
    #clean_speech = clean_speech[0, :].squeeze(0)  # (T)
    clean_speech = clean_speech[0,0]  # (T)

    hop_length = 128   # -> time between frames = 8 ms
    win_length = 2 * hop_length
    n_fft = win_length
    window = torch.hann_window(win_length)
    noisy_tf = torch.stft(noisy_speech, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True).transpose(1, 2)

    # print("after stft", datetime.now())

    _, _, F = noisy_tf.shape
    Rx_list = []
    for f in range(F):
        # print(f)
        # Get the current frequency bin
        x = noisy_tf[:, :, f].squeeze()
        Rx_f = create_autocorrelation_tensor(x, tau=tau)  # (tau, 2C, C)
        Rx_list.append(Rx_f)

    Rx = torch.stack(Rx_list, dim=-1) # (tau, 2C, C, F)
    # print(Rx.shape)
    # print(noisy_tf.shape, Rx.shape, clean_speech.shape, V_k.shape)

    return {
        "noisy_stft": noisy_tf,         # (C, T, F)
        "Rx": Rx,                       # (tau, 2C, C, F)
        "ref_channel": clean_speech,    # (T,)
        # "V_k": V_k,                     # (C, Q, F)
        "doa": doa
    }

def safe_read_and_save(sample_dir_tuple):
    duration, tau, sample_dir, sample_rate, max_length_sec, output_root = sample_dir_tuple
    sample_number = int(sample_dir.name.split("_")[-1])
    suffix = f"{sample_number:05d}"

    try:
        data = read_and_process_sample(duration, tau, sample_dir, sample_rate, max_length_sec)
        if data is not None:
            torch.save(data, output_root / f"sample_{suffix}.pt")
            
    except Exception as e:
        print(f"Error processing {sample_dir.name}: {e}")

def convert_and_save_dataset(duration, tau, input_root, output_root, sample_rate=16000, max_length_sec=6):
    print("start to load data", datetime.now())
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if str(input_root).endswith("si_tr_s"):
        mat_file = input_root / "V.mat"
        V_k = scipy.io.loadmat(mat_file)["A"] # (C, F, Q)
        V_k = torch.tensor(V_k, dtype=torch.torch.complex64)
        V_k = V_k.permute(0, 2, 1)  # from (C, F, Q) → (C, Q, F)
        V_k = V_k[..., ::2] # reduce from 257 to 129
        torch.save(V_k, output_root / "steering.pt")

    # start_idx = 0
    # end_idx = None
    sample_dirs = sorted(input_root.glob("ex_*"))
    # if end_idx is None:
    #     end_idx = len(sample_dirs)
    # sample_dirs = sample_dirs[start_idx:end_idx]
    # print(f"Processing samples {start_idx} to {end_idx - 1} ({len(sample_dirs)} samples).")

    # for d in sample_dirs:
    #     safe_read_and_save((d, sample_rate, max_length_sec, output_root))

    # Build args for multiprocessing
    args = [(duration, tau, d, sample_rate, max_length_sec, output_root) for d in sample_dirs]
    # Use multiprocessing
    with Pool(processes=49) as pool:
        for _ in tqdm(pool.imap_unordered(safe_read_and_save, args), total=len(args)):
            pass

def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        mu = torch.mean(X)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        Rx_lag += torch.matmul(x1 - mu, x2 - mu).to(device)
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag

def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr

def main(all=False, train=False, validation=False, test=False):
    mp.set_start_method("spawn", force=True)  # <--- necessary when using CUDA

    duration = 0.5 # sec
    tau = 1

    if all:
        train = validation = True
        test = False

    if train:
        convert_and_save_dataset(
        duration,
        tau,
        input_root="/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_tr_s",
        output_root=f"/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_{tau}tau_{duration:.1f}sec/train_preprocessed"
        )

    if validation:
        convert_and_save_dataset(
        duration,
        tau,
        input_root="/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_dt_05",
        output_root=f"/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_{tau}tau_{duration:.1f}sec/validation_preprocessed"
        )
    
    if test:
        convert_and_save_dataset(
        duration,
        tau,
        input_root="/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_et_05",
        output_root=f"/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_{tau}tau_{duration:.1f}sec/test_preprocessed"
        )
    


if __name__ == "__main__":
    main(all=True)


#runai-cmd --name pre -g 1 --cpu-limit 50 -- "conda activate SubspaceNetEnv && python /gpfs0/bgu-br/users/tatarjit/model-based-nir/mic2stft_miki.py"

