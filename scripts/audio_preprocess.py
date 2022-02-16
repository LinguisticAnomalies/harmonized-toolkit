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
                sys.stdout.write(f"Currently converting and {file} to wav file.\n")
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
            if file.endswith(".wav") and re.match(r"\d{3}-\d{1}.wav", file):
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


def trim_base_wav(input_path_to_file):
    """
    load and trim base wav file with customized sample rate

    :param input_path_to_file: the local path to a specifc .wav file
    :type input_path: str

    :return: audio time series
    :rtype: np.ndarray
    """
    try:
        text_param = read_json("text_process.json")
    except FileNotFoundError:
        sys.stdout.write("File not found, please run text preprocessing script first.\n")
    file_name = input_path_to_file.split(".")[2].split("/")[-1]
    sys.stdout.write(f"Trim {input_path_to_file}\n")
    time_steps = text_param[file_name]
    dur = librosa.get_duration(filename=input_path_to_file)
    # INV has multiple talking
    if len(time_steps) >= 1:
        out_files = []
        # add beginning and end
        # if INV does not start talking at very beginning
        if time_steps[0][0] != 0.0:
            time_steps.insert(0, [0.0, 0.0])
        time_steps.append([dur, dur])
        # generate non-INV time steps
        trim_ts = [[x[1], y[0]] for x, y in zip(time_steps, time_steps[1:])]
        trim_ts = [item for item in trim_ts if item[0]!=item[1]]
        trim_ts = [item for item in trim_ts if item[0]<item[1]]
        for i, interval in enumerate(trim_ts):
            out_file = ".." + input_path_to_file.split(".")[2] + f"-{i}.wav"
            out_files.append(out_file)
            start = interval[0]
            end = interval[1]
            # add = to specify milliseconds
            sox_command = f"sox {input_path_to_file} {out_file} trim {start} ={end}"
            os.system(sox_command)
        # concatenate multiple wav files into one
        os.remove(input_path_to_file)
        sox_command = "sox "
        for item in out_files:
            sox_command += f"{item} "
        sox_command += f" {input_path_to_file}"
        os.system(sox_command)
        for item in out_files:
            os.remove(item)
    # No INV taking at all
    else:
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
    # convert .mp3 to .wav
    convert_to_base_wav(param_dict["input_path"])
    # resample .wav files
    resample_audio(param_dict["input_path"], int(param_dict["sample_rate"]))
    features = param_dict["feature_extract"]
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            if file.endswith(".wav") and re.match(r"\d{3}-\d{1}.wav", file):
                loaded_file = os.path.join(subdir, file)
                if param_dict["dataset_choice"].lower() == "db":
                    trim_base_wav(loaded_file)
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
    param_dict = read_json("audio_process.json")
    parse_dirs()
    sys.stdout.write(
        f"Total time running :{datetime.now() - start_time}\n")
