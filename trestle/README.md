# TRESTLE

This repository contains code developed for TRESTLE (Toolkit for Reproducible Execution of Speech Text and Language Experiments), an open source platform that focuses on text and audio preprocessing for corpora that follow CHAT. 



## Input Structure

In particular, TRESTLE supports the following structure of **input** data.


```
├── system
    ├── corpus_1
    │   ├── audio
    │   │   ├── session_1
    │   │   └── session_2
    │   └── text
    │       ├── session_1
    │       └── session_2
    └── corpus_2
        ├── audio
        │   ├── session_1
        │   └── session_2
        │       ├── sub_session_1
        │       └── sub_session_2
        └── text
            ├── session_1
            └── session_2
                ├── sub_session_1
                └── sub_session_2
```

## Output Structure

The output of TRESTLE follows the following structure:

```
├── asr
│   ├── corpus_1
│       ├── session_1
│       └── session_2
│   ├── corpus_2
│       ├── session_1
│       └── session_2
│           ├── sub_session_1
│           └── sub_session_2
├── audio
│   ├── corpus_1
│       ├── session_1
│       └── session_2
│   ├── corpus_2
│       ├── session_1
│       └── session_2
│           ├── sub_session_1
│           └── sub_session_2
├── clips
│   ├── corpus_1
│       ├── session_1
│       └── session_2
│   ├── corpus_2
│       ├── session_1
│       └── session_2
│           ├── sub_session_1
│           └── sub_session_2
├── meta
└── text
    ├── corpus_1
        ├── session_1
        └── session_2
    ├── corpus_2
        ├── session_1
        └── session_2
            ├── sub_session_1
           └── sub_session_2
```

## Usage

### Audio

```python
# load config file
from trestle.io import load_config
cfg = load_config("configs/config.ini")

# preprocessing audio files to 16K Hz .wav files
from trestle.audio import AudioClipper, AudioClipDataset, AudioWrapper
root = Path(cfg[system][corpus])
converter = AudioWrapper(
    corpus=corpus,
    audio_root=root,
    out_root=Path(cfg['outputs']['audio']),
    dry_run=False
)
# audio clipping
clipper = AudioClipper(
    corpus=corpus,
    text_root=Path(cfg['outputs']['text']),
    out_root=Path(cfg['outputs']['clips']),
    dry_run=False
)
clipper.run()

# ASR pipeline
from trestle.audio import CTCPipeline, Seq2SeqPipeline
# hubert/wav2vec2
pipeline = CTCPipeline(
    model_name='facebook/wav2vec2-large-960h',
    # model_name="facebook/hubert-large-ls960-ft",
    corpus=corpus,
    root=Path(cfg['outputs']['clips']),
    out_root=Path(cfg['outputs']['asr']),
    batch_size=32,
    dry_run=False,
    use_flash_attn2=False
)
pipeline.run(AudioClipDataset)

# whisper
gen_config = dict(
    num_beams=5,
    do_sample=True,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    repetition_penalty=1.0,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    initial_prompt="Uhm, I, you know, like, uhm it's, it is, you know what I mean?",
)
pipeline = Seq2SeqPipeline(
    model_name="openai/whisper-large-v3",
    corpus=corpus,
    root=Path(cfg['outputs']['clips']),
    out_root=Path(cfg['outputs']['asr']),
    meta_root=Path(cfg['outputs']['meta']),
    batch_size=16,
    dry_run=False,
    use_flash_attn2=False,
    gen_config=gen_config,
)
pipeline.run(AudioClipDataset)
```


```python
# preprocessing .cha files with user-defined patterns
from trestle.io import TaskBoundary
from trestle.io import ChaTextWrapper
from trestle.text import ChaProcessor

cha_txt_patterns = {
    # remove the parentheses and keep the content only if it's English letters
    r'\([^a-zA-Z]*\)': '',
    r'\(([^a-zA-Z]*)\)': "",
    r'[()]': "",
    # remove special form markers
    r'@\w': '',
    # remove unitelligent words
    r'(\w)\1\1': '',
    # replace single underscore as a whitespace
    r'(?<!_)_(?!_)': ' ',
    r'\[.*?\]': "",
    # keep repetitions
    r'&-(\w+)': r'\1',
    # keep invited interruptions
    r'&\+(\w+)': r'\1',
    # remove gestures
    r'&=(\w+)': "",
    # keep phrase revision
    r'\<([^<>]*)\>': r'\1',
    # removing trailling off utterances
    r'\+..': "",
    # remove non-ascii characters
    r'[^\x00-\x7F]+': '',
    # remove addtional whitespace between the last word and the last punctuation
    r'\s+([.,!?;:])|([.,!?;:])\s+': r'\1\2',
    # remove additional whitespace
    r'\s+': ' ',
    # remove control characters
    r"[\x00-\x1F]+": ""
}

TOPSY_TASKS = [
    TaskBoundary(
        name=task,
        content_mark=lambda t=task: rf"@Bg:\s+{t}([\s\S]*?)@Eg:\s+{t}"
    )
    for task in ["task_1", "task_2", "task_3"]
]
wrapper = ChaTextWrapper(
    corpus=corpus,
    text_root=Path(ROOT_MAP[corpus]),
    out_root=Path(cfg["outputs"]["text"]),
    audio_root=Path(cfg["outputs"]["audio"]),
    task_boundaries=TASK_BOUNDARY_MAP[corpus],
    meta_root=Path(cfg["outputs"]["meta"]),
    dry_run=False,
)

for fmt in ["parquet", "jsonl", "csv"]:
    wrapper.run(
        cha_processor_cls=ChaProcessor,
        txt_patterns=cha_txt_patterns,
        format=fmt,
    )
```
