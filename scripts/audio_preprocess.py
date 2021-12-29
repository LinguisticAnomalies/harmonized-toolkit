"""
This script generates pre-processing audio parameters for WLS or DB with user inputs
In general, the pre-rpcessing for audio file has the following steps
    1. load audio file from .wav file
    2. resample and convert to stereo
    3. feature extraction using FTT or MFCC
"""
import sys
import os
import librosa
import sklearn
import numpy as np
from pydub import AudioSegment
from util_fun import read_json, return_bool


def convert_to_base_wav(input_path):
    """
    convert .mp3 to .wav,
    the converted files will be saved to the same local path
    
    :param input_path: the local directory containing .mp3 files
    :type input_path: str
    """
    for subdir, _, files in os.walk(input_path):
        for file in files:
            sys.stdout.write("Currently converting {} to wav file.\n".format(file))
            if file.endswith(".mp3"):
                out_file = file.split(".")[0] + ".wav"
                out_file = os.path.join(subdir, out_file)
                file_content = os.path.join(subdir, file)
                sound = AudioSegment.from_mp3(file_content)
                sound.export(out_file, format="wav")
            else:
                continue
    sys.stdout.write("Finished!\n")


def load_base_wav(input_path_to_file, sample_rate):
    """
    load base wav file with customized sample rate

    :param input_path_to_file: the local path to a specifc .wav file
    :type input_path: str
    :param sample_rate: targeting sample rate
    :type sample_rate: int

    :return: audio time series
    :rtype: np.ndarray
    """
    signal = librosa.load(input_path_to_file,
                          sr=sample_rate,
                          duration=librosa.get_duration(input_path_to_file))[0]
    return signal


def enable_fft(signal, sample_rate, window_size):
    """
    return Fourier transform the input audio signal,

    :param signal: the loaded wav audio signal
    :type signal: np.ndarray
    :param sample_rate: targeting sample rate
    :type sample_rate: int
    :param window_size: FFT window size
    :type window_size: int

    :return: the fft audio array
    :rtype: np.ndarray
    """
    return librosa.fft_frequencies(signal, sr=sample_rate, n_fft=window_size)


def enable_mfcc(signal, sample_rate, n_mfcc, scaled):
    """
    return scaled Mel-frequency cepstral coefficients (MFCCs) to the input audio signal

    :param signal: the loaded wav audio signal
    :type signal: np.ndarray
    :param sample_rate: the targeting sample rate
    :type sample_rate: int
    :param n_mfcc: number of MFCCs
    :type n_mfcc: int
    :param scaled: indicator if MCFFs are scaled
    :type scaled: bool

    :return: the MFCC audio array
    :rtype: np.ndarray
    """
    mfcc_audio = librosa.feature.mfcc(signal, sr=sample_rate, n_mfcc=n_mfcc)
    if return_bool(scaled):
        mfcc_audio = sklearn.preprocessing.scale(mfcc_audio, axis=1)
    return mfcc_audio


def parse_dirs():
    """
    extract features from audio files
    """
    param_dict = read_json("audio_process.json")
    # convert .mp3 to .wav
    convert_to_base_wav(param_dict["input_path"])
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            if file.endswith(".wav"):
                loaded_file = os.path.join(subdir, file)
                original_signal = load_base_wav(loaded_file, param_dict["sample_rate"])
                # feature extraction
                if param_dict["feature_extract"].lower == "none":
                    out_file = file.split(".")[0] + "_og.npy"
                    with open(os.path.join(subdir, out_file)) as out_f:
                        np.save(out_f, original_signal, allow_pickle=False)
                elif param_dict["feature_extract"].lower == "ftt":
                    ftt_signal = enable_fft(original_signal,
                                            param_dict["sample_rate"],
                                            int(param_dict["n_feature"]))
                    out_file = file.split(".")[0] + "_ftt.npy"
                    with open(os.path.join(subdir, out_file)) as out_f:
                        np.save(out_f, ftt_signal, allow_pickle=False)
                elif param_dict["feature_extract"].lower == "mfcc":
                    mfcc_signal = enable_mfcc(original_signal,
                                              param_dict["sample_rate"],
                                              int(param_dict["n_feature"]))
                    out_file = file.split(".")[0] + "_mfcc.npy"
                    with open(os.path.join(subdir, out_file)) as out_f:
                        np.save(out_f, mfcc_signal, allow_pickle=False)
                else:
                    raise ValueError("wrong feature extraction parameter")


if __name__ == "__main__":
    parse_dirs()
