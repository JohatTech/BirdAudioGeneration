import torch  
import torchaudio 
import torch.optim as optim
from torch.utils.data import Dataset
from torchaudio.models import WaveRNN
import os
from torchaudio.utils import download_asset
import pandas as pd 
SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
print(waveform)
print(spectrogram)
