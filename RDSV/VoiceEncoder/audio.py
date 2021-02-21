from scipy.ndimage.morphology import binary_dilation
from VoiceEncoder.hparams import *
from VoiceEncoder.util import getDiary
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], case_rttm: Optional[str]=None, sd_path: Optional[str]=None, source_sr: Optional[int]=None):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that  
    The waveform will be resampled to match the data hyperparameters.
    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    # process speaker labels
    if case_rttm is not None:
        with open(sd_path) as json_file: 
          sd = json.load(json_file)
        diary = getDiary(case_rttm)
        wav_labels = label_wav(len(wav), diary, source_sr, sd)
        wav_m, labels_m, mask = trim_long_silences(wav, wav_labels)
        return wav_m, labels_m, (wav[:len(mask)], wav_labels[:len(mask)], mask)
    else:
        wav, mask = trim_long_silences(wav)
        return wav, mask


#audio.py file function edits
def trim_long_silences(wav, labels=None):
  """
  Ensures that segments without voice in the waveform remain no longer than a 
  threshold determined by the VAD parameters in params.py.
  :param wav: the raw waveform as a numpy array of floats 
  :return: the same waveform with silences trimmed away (length <= original wav length)
  """
  # Compute the voice detection window size
  samples_per_window = (vad_window_length * sampling_rate) // 1000

  # Trim the end of the audio to have a multiple of the window size
  idx = len(wav) - (len(wav) % samples_per_window)
  wav = wav[:idx]

  # Convert the float waveform to 16-bit mono PCM
  pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

  # Perform voice activation detection
  voice_flags = []
  vad = webrtcvad.Vad(mode=3)
  for window_start in range(0, len(wav), samples_per_window):
      window_end = window_start + samples_per_window
      voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                        sample_rate=sampling_rate))
  voice_flags = np.array(voice_flags)

  # Smooth the voice detection with a moving average
  def moving_average(array, width):
      array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
      ret = np.cumsum(array_padded, dtype=float)
      ret[width:] = ret[width:] - ret[:-width]
      return ret[width - 1:] / width

  audio_mask = moving_average(voice_flags, vad_moving_average_width)
  audio_mask = np.round(audio_mask).astype(np.bool)

  # Dilate the voiced regions
  audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
  audio_mask = np.repeat(audio_mask, samples_per_window)

  if labels is not None:
    labels = labels[:idx]
    return wav[audio_mask == True], labels[audio_mask == True], audio_mask
  else:
    return wav[audio_mask == True], audio_mask

def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

# new functions
def label_wav(wav_len, casetimes, sr, spkrdict):
  mask = np.zeros(wav_len)
  st = 0
  for entry in casetimes:
    temp = entry[0].split(' ')
    time, spk = temp[4], temp[7]
    idx = int(float(time)*sr)+st
    if idx<wav_len:
      mask[st:idx] = spkrdict[spk]
    else:
      mask[st:]=spkrdict[spk]
    st = idx
  return mask
  
def wav_label_for_melspec(wav, labels, hop=160, window=400, overlap_label=999):
  mel_lab = np.zeros(int(len(wav)/hop) + 1)
  for i  in range(len(mel_lab)):
    idx = (i*hop)
    lab = np.array(labels[idx:idx+window])
    if len(np.unique(lab))==1:
      mel_lab[i]=lab[0]
    else:
      mel_lab[i]=overlap_label
  return mel_lab
