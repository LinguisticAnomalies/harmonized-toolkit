"""
This script generates pre-processing parameters for WLS or DB with user inputs
"""

import os
import re
import json
import pandas as pd

def read_json():
    """
    read the user-generated json file and save it as dict,
    return the dict
    """
    with open("text_process.json", "r") as json_file:
        prep_param = json.load(json_file)
    return prep_param


def return_bool(value):
    """
    given a value, i.e., y or Y, return the boolean value

    :param value: the value from user-generated pre-processing parameters
    :type value: str
    :return: the boolean value
    :rtype: bool
    """
    if value.lower() in ("y", "yes"):
        return True
    elif value.lower() in ("n", "no"):
        return False
    else:
        raise ValueError("Wrong response, please double check...")


def clean_text(text_chunk, param_dict):
    """
    clean the text to keep the participant transcript only, based on users' selection
    return the cleaned text

    :param text_chunk: qualified text chunk selected from original transcript
    :type text_chunk: str
    :return: the cleaned participant transcript
    :rtype: str
    """
    all_sents = text_chunk.splitlines()
    tran = ""
    for line in all_sents:
        if re.match(r"\*PAR:", line):
            line = re.sub(r"\*PAR:", "", line)
            # throat clears
            if return_bool(param_dict["clear_thoat"]):
                line = re.sub(r'\&\=clears\s+throat', r' ', line)
            # open parentheses e.g, (be)coming
            if return_bool(param_dict["open_parenthese"]):
                line = re.sub(r'\((\w+)\)(\w+)', r'\1\2', line)
            # open square brackets eg. [: overflowing] - error replacements
            if return_bool(param_dict["open_brackets"]):
                line = re.sub(r'\s+\w+\s+\[\:\s+([^\]]+)\]', r' \1 ', line)
            # remove disfluencies prefixed with "&"
            if return_bool(param_dict["disfluencies"]):
                line = re.sub(r'\&\w+\s+', r' ', line)
            # remove unitelligible words
            if return_bool(param_dict["unword"]):
                line = re.sub(r'xxx', r' ', line)
            # remove pauses eg. (.) or (..)
            if return_bool(param_dict["pauses"]):
                line = re.sub(r'\(\.+\)', r' ', line)
            # remove forward slashes in square brackets
            if return_bool(param_dict["slashes"]):
                line = re.sub(r'\[\/+\]', r' ', line)
            # remove noise indicators eg. &=breath
            if return_bool(param_dict["noise_indicators"]):
                line = re.sub(r'\&\=\S+\s+', r' ', line)
            if return_bool(param_dict["brackets_error"]):
                line = re.sub(r'\[(\*|\+|\%)[^\]]+\]', r' ', line)
                line = re.sub(r'\[(\=\?)[^\]]+\]', r' ', line)
            # finally remove all non alpha characters
            # line = "<s> "+ line + "</s>" # format with utterance start and end symbols
            if return_bool(param_dict["non_alpha_char"]):
                line = re.sub(r'[^A-Za-z\n \']', '', line)
            # replace multiple spaces with a single space
            if return_bool(param_dict["single_space"]):
                line = re.sub(r'\s+', ' ', line)
            # capitalize the first character
            if return_bool(param_dict["cap_char"]):
                line = line.capitalize()
            # if line is not empty
            if line.strip():
                tran += line.strip()
            # period at the end of sentence
            if return_bool(param_dict["eos_period"]):
                tran += ". "
            # newline at the end of sentence
            if return_bool(param_dict["eos_newline"]):
                tran += "\n"
    return tran


def parse_dirs():
    """
    extract task specifc transcript from raw dataset,
    pre-processed as users' selections,
    save it to local path
    """
    param_dict = read_json()
    task_df = pd.DataFrame(columns=["file", "text"])
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            with open(os.path.join(subdir, file), mode="r", errors="ignore") as file_content:
                all_tran = file_content.read()
                if param_dict["dataset_choice"].lower() == "wls":
                    try:
                        text = re.search(r'@Bg:	Activity\n.*?@Eg:	Activity',
                                        all_tran, re.DOTALL).group()
                        tran = clean_text(text, param_dict)
                        tran_row = {"file": "20000" + file.split(".")[0],
                                    "text": tran}
                        task_df = task_df.append(tran_row, ignore_index=True)
                    except AttributeError:
                        # if no qualified transcript
                        pass
                elif param_dict["dataset_choice"].lower() == "db":
                    tran = clean_text(all_tran, param_dict)
                    tran_row = {"file": file.split(".")[0],
                                "text": tran}
                    task_df = task_df.append(tran_row, ignore_index=True)
                else:
                    raise ValueError("Dataset is not supported, please double check...")
    # merge to get uids
    if param_dict["dataset_choice"].lower() == "wls":
        if return_bool(param_dict["meta"]):
            wls_meta = pd.read_csv("../wls_final_meta.tsv", sep="\t")
            task_df = pd.merge(wls_meta, task_df, left_on="ids", right_on="file")
        else:
            wls_ids = pd.read_csv("../wls_ids.tsv", sep="\t")
            task_df = pd.merge(wls_ids, task_df, left_on="ids", right_on="file")
    elif param_dict["dataset_choice"].lower() == "db":
        if return_bool(param_dict["meta"]):
            pitt_meta = pd.read_csv("../pitt_final_meta.tsv", sep="\t")
            task_df = pd.merge(pitt_meta, task_df, left_on="file", right_on="file")
        else:
            pitt_ids = pd.read_csv("../db_ids.tsv", sep="\t")
            task_df = pd.merge(pitt_ids, task_df, left_on="file", right_on="file")
    task_df.to_csv(param_dict["out_path"], sep="\t", index=False)


if __name__ == "__main__":
    parse_dirs()
