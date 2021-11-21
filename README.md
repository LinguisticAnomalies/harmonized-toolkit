# The Harmonized Pre-processing Toolkit for Dementia Bank

This repository contains code developed for the harmonized pre-processing toolkit for Dementia Bank dataset

While the data of Dementia Bank is publicly available, we are not able to redistribute any of these data as per Data Use agreement with Dementia Bank. Individual investigators need to contact the [Dementia Bank](https://dementia.talkbank.org/access/) to request access to the data.

## Setup

It is recommended to use a virtual environment (i.e.,  [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), or [conda](https://docs.conda.io/en/latest/)) to use and develop this toolkit. Please install dependency packages using ```pip install -r requirements.txt``` or ```conda install --yes --file requirements.txt``` with python version at least 3.8

**For M1 user**: Please install the dependency packages with miniforge. Please visit [miniforge GitHub link](https://github.com/conda-forge/miniforge) for more details

## Usage

### Pre-processing

Ideally the toolkit should have the following structure

```
- Dementia Bank folder contains .cha files
- WLS folder contains .cha files
- scripts
	- generate_text_preprocess.sh
	- text_preprocess.py
	- generate_audio_preprocess.sh
	- audio_preprocess.py
```

Both `generate_text_preprocess.sh` and `generate_audio_preprocess.sh` allow to select the data type, pre-processing parameters and input data folder to generate the corresponding cleaned dataset.

### Defining classes and data selection

Participants are required to define their own label to the existing Dementia Bank and WLS dataset. More specifically, participants need to **define** each data samples as one of two categories "positive" vs. "negative" with provided metadata.  We will provide clinical criteria to help participants better understand the metadata. We will also provide scripts to help participants to perform data selection.

### Analysis pipeline

Participants are required to design their own analysis pipeline with the given pre-processing and data selection pipelines.

### Reporting and Submitting manifest

While we do not ask participants to upload or share their analysis pipeline, participants need to report the details of their analysis pipeline, i.e., the corresponding pre-processing json generated by `generate_x_preprocess.sh` files, the corresponding data selection json generated by `data_selection.sh` file, and the evaluation process. 

Ideally, the final manifest should be a `.json` file with similar structure as follows:

```json
{
  "pre_process": the .json generated by generate_x_.preprocess.sh,
	"data_selection": the .json generated by data_selection.sh,
	"evaluation":{
		"metrics": AUC/ACC/F1
		# there are many ways for evaluation
		"evaluation": {
			"train/test": Y/N {
				"train_set": local path to training set,
				"test_set": local path to test set
			}
			"cv": "leave_one_out"/"kfold",
			"n_fold": n
		}
	}
}
```

## Contributing

If you find any problems, please feel free to open an issue

If you can fix an issue you've found, or another issue, please open a pull request with the following steps:

1. For this repository on GitHub to start making your changes
2. Pass test to show the issue is fixed
3. Submit the pull request
