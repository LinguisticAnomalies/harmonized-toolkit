# use this shell script to generate pre-processing parameters
# !/bin/bash
read -p "Which dataset you are pre-processing? wls or db?: "  dataset_choice
read -p "Where are the .mp3 files located?: " input_path
read -p "Where do you want to store the trimmed audio segments? " out_path
read -p "Enter sample rate: " sample_rate
read -p "Feature extraction methods, selecting from FTT or MFCC or NONE: " feature_extract
read -p "Enter number of FTT windows size or MFCC, 0 for NONE: " n_feature
read -p "Scaling MFCC? y/n: " scale
echo '{"dataset_choice": "'$dataset_choice'",
       "sample_rate": "'$sample_rate'",
       "input_path": "'$input_path'",
       "out_path": "'$out_path'",
       "feature_extract": "'$feature_extract'",
       "n_feature": "'$n_feature'",
       "scale": "'$scale'" }' > audio_process.json
echo "Your audio pre-processing json file has been generated!\n"
echo "Running audio pre-processing script now...\n"
python audio_preprocess.py
echo "Your dataset is now pre-processed!\n"