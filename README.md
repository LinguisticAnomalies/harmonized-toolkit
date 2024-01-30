# TRESTLE

This repository contains code developed for TRESTLE (Toolkit for Reproducible Execution of Speech Text and Language Experiments), an open source platform that focuses on text and audio preprocessing for corpora that follow CHAT and Praat's TextGrid protocols. TRESTLE is initially designed to execute preprocessing for two datasets from the TalkBank repository with dementia detection as an illustrative domain. It can be easily adapted to other corpora that support CHAT and TextGrid protocols. The TextGrid functionalities are supported by the [textgrid](https://github.com/kylebgorman/textgrid) package.


## Setup
Please install dependency packages using `pip install -r requirements.txt` or `conda install --yes --file requirements.txt` with Python version at least 3.8.

Please also make sure you have [FFmpeg](https://ffmpeg.org/) installed.

## Usage

### Text Preprocessing

TRESTLE supports user-defined regex patterns for text-prerpcessing. To start, one needs to define regex patterns by initializing a python dictionary:

```python
# Define regex patterns for text preprocessing
txt_patterns = {
        r'\([^)]*\)': "",
        r'(\w)\1\1': '',
        r'\[.*?\]': "",
        r'&-(\w+)': r'\1',
        r'&+(\w+)': r'\1',
        r'<(\w+)>': r'\1',
        r'\+...': "",
        r'[^A-Za-z\n \']': '',
        r'\s+': ' ',
    }
```

Users also need to define a dictionary containing pointers to the input text/audio and save locations for output. In this dictionary, users also need to specify the format of input text  (i.e., `.cha` or `.TextGrid`), the audio format (i.e., `.mp3` or `.wav`). The `speaker` is the required field for CHAT transcripts to indicate the utterance to be preprocessed. If it has multiple tasks in a CHAT transcript, users can use `content` to specify the subset for further preprocessing.

Please note that the paths should be full path.

```python
sample = {
  "format": ".cha",
  "text_input_path":"/path/to/input/txt",
  "audio_input_path": "/path/to/audio/recordings",
  "text_output_path": "/path/to/output/txt",
  "audio_output_path": "/path/to/output/audio/recording",
  "audio_type": ".mp3",
  "speaker": "*PAR",
  "content": r'@Bg:	Activity\n.*?@Eg:	Activity'
  }
```

Users can start preprocessing by:

```python
from TRESTLE import TextWrapperProcessor
wrapper_processor = TextWrapperProcessor(
    data_loc=sample, txt_patterns=txt_patterns)
wrapper_processor.process()
```

The utterance-level transcript will be saved to the `text_output_path` as a .jsonline file with the following structure:

```jsonline
{"start": 0.0, "end": 6875.0, "text": "one does not simply walk in to mordor", "audio": "/path/to/the/recording/file_name.wav"}
```

#### Note
TRESTLE assumes that the the file names for the text transcript and audio recording for each participant stay the same. If a .TextGrid corpus that presents special tier naming convention, one can add a special `data_type` key-value pair in the following setup:

```python
sample = {
  "format": ".TextGrid",
  "text_input_path":"/path/to/input/txt",
  "audio_input_path": "/path/to/audio/recordings",
  "text_output_path": "/path/to/output/txt",
  "audio_output_path": "/path/to/output/audio/recording",
  "audio_type": ".mp3",
  "data_type": "a_special_mark",
  }
```
Then modify the processing rule in the `clean_textgird` and `process_special_tier` functions in the `TextGridProcessor` class accordingly.

### Audio Preprocessing

To preprocess the audio recordings, users need to run text preprocessing first to get the utterance-level transcripts. The audio preprocessing module is designed to resample the input audio recordings and slice them into utterance-level audio clips. Users can start audio preprocessing by:

```python
processor = AudioProcessor(
  data_loc=sample, sample_rate=16000)
processor.process_audio()
```

The audio output folder with the following structure:

```
├── audio_output_path
│   ├── metadata.csv
│   ├── file_name_1.wav
│   ├── file_name_2.wav
│   ├── file_name_3.wav
```

The metadata.csv files stores the audio clips and corrsponding transcript with the following structure:

```
file_name,transcription
file_name_1.wav,one does not simply walk into mordor
```


## AAAI 2022 Hackallenge

Read more [here](hackallenge.md).


## Citation

If you use TRESTLE in your research, please cite our paper via:

```bib
@article{li2023trestle,
  title={TRESTLE: Toolkit for Reproducible Execution of Speech, Text and Language Experiments},
  author={Li, Changye and Xu, Weizhe and Cohen, Trevor and Michalowski, Martin and Pakhomov, Serguei},
  journal={AMIA Summits on Translational Science Proceedings},
  volume={2023},
  pages={360},
  year={2023},
  publisher={American Medical Informatics Association}
}
```
