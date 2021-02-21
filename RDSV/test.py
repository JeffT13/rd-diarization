import os, csv
import torch
import webrtcvad
import librosa
import struct
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Union, List
from time import perf_counter as timer

from VoiceEncoder.hparams import *
from VoiceEncoder.audio import preprocess_wav
from VoiceEncoder.voice_encoder import VoiceEncoder

from utils import RefAudioLibrary, Diarize
from param import *

print(mel_window_length, audio_norm_target_dBFS )
print('---')
print(audio_path, c_set, d_set)
print('complete')
