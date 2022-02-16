# The Harmonized Pre-processing Toolkit for Dementia Bank

This repository contains code developed for the harmonized pre-processing toolkit for Dementia Bank dataset

While the data of Dementia Bank is publicly available, we are not able to redistribute any of these data as per Data Use agreement with Dementia Bank. Individual investigators need to contact the [Dementia Bank](https://dementia.talkbank.org/access/) to request access to the data.

## Setup

It is recommended to use a virtual environment (i.e.,  [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), or [conda](https://docs.conda.io/en/latest/)) to use and develop this toolkit. Please install dependency packages using ```pip install -r requirements.txt``` or ```conda install --yes --file requirements.txt``` with python version at least 3.8.

For audio preprocessing, please also install [FFmpeg](https://github.com/FFmpeg/FFmpeg) and [sox](https://github.com/rabitt/pysox) on your local system.

## Usage

### Pre-processing

This toolkit has the following file structure

```
- audio folder
    - pitt
        - dementia
        - control
    - wls
- transcript folder
    - pitt
        - dementia
        - control
    - wls
- scripts
	- generate_text_preprocess.sh
	- text_preprocess.py
	- generate_audio_preprocess.sh
	- audio_preprocess.py
```

Both `generate_text_preprocess.sh` and `generate_audio_preprocess.sh` allow to select the data type, pre-processing parameters and input data folder to generate the corresponding cleaned dataset.

### Supported preprocessing arguments

For more details, please check the corresponding `.sh` file.

#### Audio

- Convert `.mp3` to `.wav`
- Resample with user-defined sample rate
- Trim out audio samples from investigators (**Note**: this feature only supports Dementia Bank Pitt corpus)
- Feature extraction with Fourier transform
- Feature extraction Mel-frequency cepstral coefficients
- No feature extraction - keep the original audio time series array
- Save the audio time series array to local `.npy` files

To load `.npy` file:

```python
import numpy as np
data = np.load("data.npy", allow_pickle=False)
```

#### Text

- Remove clear thoat indicator
- Remove open parenthese or brackets
- Remove disfluencies, unword, pauses
- Remove noise indicator
- Capitalize the first character
- Add newline at the end of sentence
- Save the pre-processed transcripts as a .tsv file

## Defining classes and data selection

Participants are required to define their own label to the existing Dementia Bank and WLS dataset. More specifically, participants need to **define** each data samples as one of two categories "positive" vs. "negative" with provided metadata.

## Analysis pipeline

Participants are required to design their own analysis pipeline with the given pre-processing pipeline.

## Reporting and Submitting Manifest

While we do not ask participants to upload or share their analysis pipeline, participants need to report the details of their pre-processing, data seleciton and analysis pipeline in a **.json** file.

We provide the baseline manifest as follows:

```json
{
"pre_process": "scripts/text_process.json",
"data_uids": ["001-2", "005-2", "006-4", "010-3"],
"postive_uids": ["001-2", "005-2", "018-0"],
"negative_uids": ["006-4", "013-0", "015-3"],
"training_uids": ["001-2", "005-2", "006-4"],
"test_uids": ["035-1", "045-0", "049-1"],
"method": "fine-tuning BERT base model with 10 epochs and 8 batch size on ADReSS training set, validatinng on ADReSS test set",
"evaluation":
{
    "ACC@EER": 0.83,
    "AUC@EER": 0.91,
    "ACC": 0.77,
    "AUC": 0.77
}
}
```

## Known Issues

Some .cha files did not align very well, which some audio samples from invetigators did not marked and thus did not get removed by this toolkit.
