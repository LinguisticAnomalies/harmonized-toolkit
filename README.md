# The Harmonized Pre-processing Toolkit for Dementia Bank

This repository contains code developed for the harmonized pre-processing toolkit for Dementia Bank dataset

While the data of Dementia Bank is publicly available, we are not able to redistribute any of these data as per Data Use agreement with Dementia Bank. Individual investigators need to contact the [Dementia Bank](https://dementia.talkbank.org/access/) to request access to the data.

## Setup

It is recommended to use a virtual environment (i.e.,  [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), or [conda](https://docs.conda.io/en/latest/)) to use and develop this toolkit. Please install dependency packages using ```pip install -r requirements.txt``` or ```conda install --yes --file requirements.txt``` with python version at least 3.8.

For audio preprocessing, please also install [FFmpeg](https://github.com/FFmpeg/FFmpeg).

## Usage

### Pre-processing

This toolkit has the following file structure

```
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
- User defined sample rate
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
- Save the preprocessed text transcirpt as Pandas dataframe

## Defining classes and data selection

Participants are required to define their own label to the existing Dementia Bank and WLS dataset. More specifically, participants need to **define** each data samples as one of two categories "positive" vs. "negative" with provided metadata.

## Analysis pipeline

Participants are required to design their own analysis pipeline with the given pre-processing pipeline.

## Reporting and Submitting manifest

While we do not ask participants to upload or share their analysis pipeline, participants need to report the details of their pre-processing, data seleciton and analysis pipeline.

We provide the baseline manifest as follows:

```
{
  "pre_process": scripts/generate_text_preprocess.sh,
	"data_uids": [(001,2), (005,2), (006,4), (010,3), ..., (709, 0)],
  "postive_uids": [(001,2), (005,2), (018,0), ..., (705, 0)],
  "negative_uids": [(006,4), (013, 0), (015,3), ..., (709,0)],
  "training_uids": [(001,2), (005,2), (006,4), ..., (705,0)],
  "test_uids": [(035,1), (045,0), (049,1),...,(709,0)],
  # one-line description of methods
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

Where the ids of ADReSS set consist of a tuple with two parts: a) participant id, and b) transcript id, which marks the $n$-th visit. Both information can be found in the Pitt's corpus metadata file.
