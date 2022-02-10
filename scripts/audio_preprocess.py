"""
This script generates pre-processing audio parameters for WLS or DB with user inputs
In general, the pre-rpcessing for audio file has the following steps
    1. load audio file from .wav file
    2. resample and convert to stereo
    3. feature extraction using FTT or MFCC
"""
from datetime import datetime
import sys
import re
import os
import librosa
import soundfile as sf
import sox
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
    sys.stdout.write("Start to convert .mp3 to .wav\n")
    for subdir, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mp3") and re.match(r"\d+-\d", file):
                sys.stdout.write(f"Currently converting {file} to wav file.\n")
                out_file = file.split(".")[0] + ".wav"
                out_file = os.path.join(subdir, out_file)
                file_content = os.path.join(subdir, file)
                sound = AudioSegment.from_mp3(file_content)
                sound.export(out_file, format="wav")
            else:
                continue
    sys.stdout.write("Finished!\n")


def resample_audio(input_path, resample_rate):
    """
    resample the audio file with 16kHz

    :param input_path: the local path to .wav files
    :type input_path: str
    :param resample_rate: the resample rate for .wav files
    :type resample_rate: int
    """
    sys.stdout.write("Start to resample audio to target sample rate...\n")
    for subdir, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".wav"):
                sys.stdout.write(f"Resample {file}\n")
                loaded_file = os.path.join(subdir, file)
                og_signal, sample_rate = librosa.load(loaded_file)
                new_signal = librosa.resample(
                    og_signal, orig_sr=sample_rate, target_sr=resample_rate)
                # write to local file
                sf.write(loaded_file, new_signal, resample_rate)
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
    signal, _ = librosa.load(input_path_to_file,
                          sr=sample_rate)
    return signal


def trim_base_wav(input_path_to_file, sample_rate):
    """
    load and trim base wav file with customized sample rate

    :param input_path_to_file: the local path to a specifc .wav file
    :type input_path: str
    :param sample_rate: targeting sample rate
    :type sample_rate: int

    :return: audio time series
    :rtype: np.ndarray
    """
    try:
        text_param = read_json("text_process.json")
    except FileNotFoundError:
        sys.stdout.write("File not found, please run text preprocessing script first.\n")
    file_name = input_path_to_file.split(".")[2].split("/")[-1]
    time_steps = text_param[file_name]
    # trim process
    sys.stdout.write(f"Trimming {input_path_to_file} to local.\n")
    tfm = sox.Transformer()
    signal_out = tfm.build_array(
        input_filepath=input_path_to_file, sample_rate_in=sample_rate)
    for interval in time_steps:
        tfm.trim(float(interval[0]), float(interval[1]))
    tfm.compand()
    tfm.build_file(
        input_array=signal_out, sample_rate_in=sample_rate,output_filepath=input_path_to_file)
    return signal_out


def enable_fft(signal, window_size):
    """
    return Fourier transform the input audio signal,

    :param signal: the loaded wav audio signal
    :type signal: np.ndarray
    :param window_size: FFT window size
    :type window_size: int

    :return: the fft audio array
    :rtype: np.ndarray
    """
    fourier = np.fft.fft(signal)
    freq = np.fft.fftfreq(n=signal.size, d = window_size)
    return freq


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
    param_dict["sample_rate"] = int(param_dict["sample_rate"])
    # convert .mp3 to .wav
    convert_to_base_wav(param_dict["input_path"])
    # resample .wav files
    resample_audio(param_dict["input_path"], int(param_dict["sample_rate"]))
    features = param_dict["feature_extract"]
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            if file.endswith(".wav"):
                loaded_file = os.path.join(subdir, file)
                # trim audio for DB
                if param_dict["dataset_choice"] == "db":
                    original_signal = trim_base_wav(
                        loaded_file, int(param_dict["sample_rate"]))
                else:
                    original_signal = load_base_wav(
                        loaded_file, int(param_dict["sample_rate"]))
                # feature extraction
                if features.lower() == "none":
                    out_file = file.split(".")[0] + "-og.npy"
                    np.save(
                        str(os.path.join(subdir, out_file)), original_signal)
                elif features.lower() == "ftt":
                    ftt_signal = enable_fft(
                        original_signal, int(param_dict["n_feature"]))
                    out_file = file.split(".")[0] + "-ftt.npy"
                    np.save(
                        str(os.path.join(subdir, out_file)), ftt_signal)
                elif features.lower() == "mfcc":
                    mfcc_signal = enable_mfcc(
                        original_signal, int(param_dict["sample_rate"]),
                        int(param_dict["n_feature"]), param_dict["scale"])
                    out_file = file.split(".")[0] + "-mfcc.npy"
                    np.save(
                        str(os.path.join(subdir, out_file)), mfcc_signal)
                else:
                    raise ValueError("wrong feature extraction parameter")


if __name__ == "__main__":
    start_time = datetime.now()
    parse_dirs()
    sys.stdout.write(
        "Total time running :{}\n".format(datetime.now() - start_time))