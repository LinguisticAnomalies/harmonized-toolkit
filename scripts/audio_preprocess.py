"""
This script generates pre-processing audio parameters for WLS or DB with user inputs
In general, the pre-rpcessing for audio file has the following steps
    1. load audio file from .wav file
    2. resample and convert to stereo
    3. feature extraction using FTT or MFCC
"""
from datetime import datetime
import glob
import sys
import re
import os
import librosa
import soundfile as sf
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
            if file.endswith(".mp3"):
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
    :param data_type: whether it's wls or db
    :type data_type: str
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


def trim_base_wav(input_path_to_file, out_path, data_type):
    """
    load and trim base wav file with customized sample rate

    :param input_path_to_file: the local path to a specifc .wav file
    :type input_path: str
    :param out_path: the local path to store the trimmed audio pieces
    :type out_path: str
    :param data_type: the type of the data, i.e., db or wls
    :type data_type: str
    """
    try:
        text_param = read_json("text_process.json")
    except FileNotFoundError:
        sys.stdout.write("File not found, please run text preprocessing script first.\n")
    if data_type == "db":
        file_name = re.findall(r"\d{3}-\d{1}", input_path_to_file)[0]
    elif data_type == "wls":
        file_name = re.findall(r"\d{5}", input_path_to_file)[0]
        file_name = '20000'+str(file_name)
    else:
        raise ValueError("Cannot find the waveform")
    try:
        time_stamps = text_param[file_name]
        sys.stdout.write(f"Trim {file_name}.wav\n")
        # # get the timestamps for the specific file
        # time_stamps = text_param[file_name]
        # # trim audio from participants with specific intervals
        if len(time_stamps) > 0:
            for i, interval in enumerate(time_stamps):
                out_file = f"{file_name}-{i}.wav"
                out_file = os.path.join(out_path, out_file)
                start = interval[0]
                end = interval[1]
                sox_command = f"sox {input_path_to_file} {out_file} trim {start} ={end}"
                os.system(sox_command)
        else:
            pass
    except KeyError:
        # file does not have qualified intervals
        pass


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
    _ = np.fft.fft(signal)
    freq = np.fft.fftfreq(n=signal.size, d=window_size)
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
    mfcc_audio = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    if return_bool(scaled):
        mfcc_audio = sklearn.preprocessing.scale(mfcc_audio, axis=1)
    return mfcc_audio


def parse_dirs():
    """
    extract features from audio files
    """
    param_dict = read_json("audio_process.json")
    param_dict["sample_rate"] = int(param_dict["sample_rate"])
    # check if they are already converted and resampled
    # only for debug
    if len(glob.glob(param_dict["input_path"]+"*.wav")) == 0:
        # convert .mp3 to .wav
        convert_to_base_wav(param_dict["input_path"])
        # resample .wav files
        resample_audio(param_dict["input_path"], int(param_dict["sample_rate"]))
    features = param_dict["feature_extract"]
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            if param_dict["dataset_choice"].lower() == "db":
                # make sure to load the correct db waveform
                if file.endswith(".wav") and re.match(r"\d{3}-\d{1}.wav", file):
                    loaded_file = os.path.join(subdir, file)
                    trim_base_wav(
                        loaded_file, param_dict["out_path"],
                        param_dict["dataset_choice"].lower())
                    original_signal = load_base_wav(
                        loaded_file, int(param_dict["sample_rate"]))
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
                else:
                    continue
            elif param_dict["dataset_choice"].lower() == "wls":
                if file.endswith(".wav") and re.match(r"\d{5}.wav", file):
                    loaded_file = os.path.join(subdir, file)
                    trim_base_wav(
                        loaded_file, param_dict["out_path"],
                        param_dict["dataset_choice"].lower())
                    original_signal = load_base_wav(
                        loaded_file, int(param_dict["sample_rate"]))
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
            else:
                raise ValueError("Wrong dataset")


if __name__ == "__main__":
    start_time = datetime.now()
    param_dict = read_json("audio_process.json")
    parse_dirs()
    sys.stdout.write(
        f"Total time running :{datetime.now() - start_time}\n")
