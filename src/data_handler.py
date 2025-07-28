'''

Functions for loading the data

'''

# Imports
import torch
import numpy as np
import itertools
from tqdm import tqdm
from pathlib import Path
from src.system_model import SystemModelParams
import os
import scipy.io
import soundfile as sf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimDS(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.root_folder = Path(root_folder)
        self.files = sorted(self.root_folder.glob("sample_*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        data = torch.load(sample_path)

        return data["noisy_stft"], data["Rx"], data["ref_channel"], data["V_k"]

class SimDSdoa(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.root_folder = Path(root_folder)
        self.files = sorted(self.root_folder.glob("sample_*.pt"))

        # Load global V.pt once
        v_file = self.root_folder / "steering.pt"
        assert v_file.exists(), f"steering.pt not found at {v_file}"

        self.V_k = torch.load(v_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        data = torch.load(sample_path)

        # return data["noisy_stft"], data["Rx"], data["ref_channel"], data["V_k"], data["doa"]
        return data["noisy_stft"], data["Rx"], data["ref_channel"], self.V_k, data["doa"]
        # return data["noisy_speech"], data["Rx"], data["ref_channel"], data["V_k"], data["doa"]


