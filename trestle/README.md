# TRESTLE

This repository contains code developed for TRESTLE (Toolkit for Reproducible Execution of Speech Text and Language Experiments), an open source platform that focuses on text and audio preprocessing for corpora that follow CHAT and Praat's TextGrid protocols.


## TalkBank Data Structure

```
├── DementiaBank
│   ├── pitt
│   │   ├── control
│   │   │   ├── audio
│   │   │   └── text
│   │   └── dementia
│   │       ├── audio
│   │       └── text
│   └── wls
│       ├── audio
│       └── text
└── PsychosisBank
    ├── discourse
    │   ├── audio
    │   │   ├── Baseline
    │   │   └── Followup
    │   └── text
    │       ├── Baseline
    │       └── Followup
    └── topsy
        ├── audio
        │   ├── TOPSY-0
        │   └── TOPSY-1
        │       ├── 12M
        │       ├── 18M
        │       ├── 24M
        │       ├── 30M
        │       └── 6M
        └── text
            ├── TOPSY-0
            └── TOPSY-1
                ├── 12M
                ├── 18M
                ├── 24M
                ├── 30M
                └── 6M
```

## TRESTLE Structure

```
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── trestle
│   │   ├── audio
│   │   │   ├── audio_processor.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   ├── configs
│   │   │   └── config.ini
│   │   ├── __init__.py
│   │   ├── io
│   │   │   ├── audio_wrapper.py
│   │   │   ├── batch_wrapper.py
│   │   │   ├── config.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   └── text_wrapper.py
│   │   ├── __pycache__
│   │   └── text
│   │       ├── cha_processor.py
│   │       ├── __init__.py
│   │       └── __pycache__
│   └── trestle.egg-info
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
├── tests
│   ├── __pycache__
│   ├── test_audio.py
│   ├── test_cha.py
│   └── test_features.py
└── uv.lock
```

## Changelogs
- [x] rework .cha processor
- [x] rework audio preprocessor
- [ ] rework audio sliding processor
- [ ] rework textgrid processor
- [ ] add ASR pipeline
- [ ] better downstream feature pipeline API
- [ ] rewrite readme 
