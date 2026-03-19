# TRESTLE

This repository contains code developed for TRESTLE (Toolkit for Reproducible Execution of Speech Text and Language Experiments), an open source platform that focuses on text and audio preprocessing for corpora that follow the CHAT protocol.

## Setup

TRESTLE is built with python 3.12. The environment can be setup with `uv sync` with the `pyproject.toml` file.

By default, TRESTLE's dependencies include pytorch with cuda 12.6, which the cuda version can be easily updated in `pyproject.toml` file.



## TRESTLE Structure

TRESTLE has 4 modules: a) audio; b) text, c) config; and d)io. The structure of TRESTLE is shown below.

```
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── trestle
│   │   ├── audio
│   │   │   ├── audio_processor.py
│   │   │   ├── __init__.py
│   │   ├── configs
│   │   │   └── config.ini
│   │   ├── __init__.py
│   │   ├── io
│   │   │   ├── audio_wrapper.py
│   │   │   ├── batch_wrapper.py
│   │   │   ├── config.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── __init__.py
│   │   │   └── text_wrapper.py
│   │   └── text
│   │       ├── cha_processor.py
│   │       ├── __init__.py
└── uv.lock
```

The `config.ini` under the config module should follow the following structures, where `corpus_1` and `system` are placeholder for the TalkBank subsite and corpus.

```
[system]
corpus_1 = /path/to/system/corpus_1 # path to corpus file with audio and text as separate sub folders
corpus_2 = /path/to/system/corpus_2

[outputs]
text = /path/to/output/text
audio = /path/to/output/audio
clips = /path/to/output/clips
asr = /path/to/output/asr
meta = /path/to/output/meta # preprocessing patterns and ASR decoding configs are saved here as .json files
```

### Audio Module
The audio module supports the following functionalities:

- Resample the audio files (by default, in .mp3) into 16kHz .wav files.
- Segment resampled audio files into utterance-level clips for downstream ASR pipeline. Note that text preprocessing is required for this functionality.
  - The sementation saves the `metadata.parquet` file under each child folder for matting the source audio file and audio clip file
- ASR pipeline to generate transcripts on the utterance level, using the data obtained in the audio clipper function
    - The ASR pipeline currently supports inference with Wav2Vec2, HuBERT, and Whisper. The decoding strategies for whisper is a required argument and will be saved to `output/meta folder`

The audio clip metdata follows the following structure:

```
clip_path: the path to the this clip audio, naming conversion follows {task}_{pid}_{n}.wav, where n representing the n-th clip from the task with participant unique identifier of pid
pid: participant unique identifier
text: the utterance for this clip
source_audio: the source audio file
```

### Text Module

The text module preprocesses .cha files with user defined regex patterns, where the patterns are saved to `output/meta folder`
  - Users can defind a task boundary for task-wise preprocessing
  - If the task boundary is not defined, the text module preprocesses **all** utterances in the .cha files
  - This module generates two output files as per task: `{task}_utterance.{format}` and `{task}_participant.{format}`, where the supported formats are .jsonl, .csv, and .parquet
  
The output files follows the following structure:

```
start: start_in_ms for the utterance
end: end_in_ms for the utterance
text: the utterance in the .cha file
pid: unique identider for participant,
audio_path: path_to_audio
```
  
### Config Module

This module reads the `config.ini` file stores under this module for reading the input and output paths

### IO module

This module supports intermediate functionalities for preprocessing.

For input and output structure, and TRESTLE usage, please refer to [here](trestle/README.md)

## Misc

For TRESTLE 1.0, please refer to [here](v1).

For a sample analysis code, please refer to [here](analysis)

## Changelogs
- [x] rework .cha processor
- [x] rework audio preprocessor
- [x] rework audio sliding processor
- [x] add ASR pipeline
- [x] better downstream feature pipeline API
- [x] rewrite readme 
- [ ] add textgrid processor
