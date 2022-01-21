# use this shell script to generate pre-processing parameters
# !/bin/bash
read -p "Which dataset you are pre-processing? wls or db?: "  dataset_choice
read -p "Where are the .cha files located?: " input_path
read -p "Remove 'clear throat'? (Y/N): " clear_thoat
read -p "Remove open parentheses e.g, (be)coming? (Y/N): " open_parenthese
read -p "Remove open square brackets eg. [: overflowing]? (Y/N): " open_brackets
read -p "Remove disfluencies prefixed with '&'? (Y/N): " disfluencies
read -p "Remove unitelligible words? (Y/N): " unword
read -p "Remove pauses eg. (.) or (..)? (Y/N): " pauses
read -p "Remove forward slashes in square brackets? (Y/N): " slashes
read -p "Remove noise indicators e.g. &=breath? (Y/N): ": noise_indicators
read -p "Remove square brackets indicating an error code? (Y/N): " brackets_error
read -p "Remove all non alpha characters? (Y/N): " non_alpha_char
read -p "Remove multiple spaces with a single space? (Y/N): " single_space
read -p "Capitalize the first character? (Y/N): " cap_char
read -p "Add period at the end of every sentence? (Y/N): " eos_period
read -p "Add newline at the end of every sentence? (Y/N): " eos_newline
read -p "You data will be stored as .tsv file. Please enter the output path and file name for your pre-processed transcripts: " out_path
echo "Please stand by, your pre-processing script will be generated shortly...\n"
echo '{"dataset_choice": "'$dataset_choice'",
       "input_path": "'$input_path'",
       "clear_thoat": "'$clear_thoat'",
       "open_parenthese": "'$open_parenthese'",
       "open_brackets": "'$open_brackets'",
       "disfluencies": "'$disfluencies'",
       "unword": "'$unword'",
       "pauses": "'$pauses'",
       "slashes": "'$slashes'",
       "noise_indicators": "'$noise_indicators'",
       "brackets_error": "'$brackets_error'",
       "non_alpha_char": "'$non_alpha_char'",
       "single_space": "'$single_space'",
       "cap_char": "'$cap_char'",
       "eos_period": "'$eos_period'",
       "eos_newline": "'$eos_newline'",
       "out_path": "'$out_path'" }' > text_process.json
echo "Your text pre-processing json file has been generated!\n"
echo "Running text pre-processing script now...\n"
python text_preprocess.py
echo "Your dataset is now pre-processed!\n"