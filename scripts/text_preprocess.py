"""
This script generates pre-processing text parameters for WLS or DB with user inputs
"""

import os
import json
import re
import pandas as pd
from util_fun import read_json, return_bool


def get_inv_slots(text_chunk):
    """
    return all timestamps belongs to INV and 

    :param text_chunk: qualified text chunk from .cha files
    :type text_chunk: str
    """
    all_sents = text_chunk.splitlines()
    inv_ts = []
    for line in all_sents:
        if re.match(r"\*INV", line):
            pattern = re.findall(r"\d+_\d+", line)
            if pattern:
                ts_intervals = pattern[0].split("_")
                # convert millisecond to second
                ts_intervals = [int(item)/1000 for item in ts_intervals]
                # add timestamp to the full list
                inv_ts.append(ts_intervals)
        else:
            pass
    return inv_ts


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
            # only add non-empty sentences
            # if line is not empty
            if len(line.strip()) > 0:
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
    param_dict = read_json("text_process.json")
    task_df = pd.DataFrame(columns=["file", "text"])
    for subdir, _, files in os.walk(param_dict["input_path"]):
        for file in files:
            if file.endswith(".cha"):
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
                            inv_ts = get_inv_slots(all_tran)
                            param_dict["20000" + file.split(".")[0]] = inv_ts
                        except AttributeError:
                            # if no qualified transcript
                            pass
                    elif param_dict["dataset_choice"].lower() == "db":
                        tran = clean_text(all_tran, param_dict)
                        tran_row = {"file": file.split(".")[0],
                                    "text": tran}
                        inv_ts = get_inv_slots(all_tran)
                        param_dict[file.split(".")[0]] = inv_ts
                        task_df = task_df.append(tran_row, ignore_index=True)
                    else:
                        raise ValueError("Dataset is not supported, please double check...")
    task_df.to_csv(param_dict["out_path"], sep="\t", index=False)
    # rewrite param_dict with investigator timestep
    with open("text_process.json", "w") as json_file:
        json.dump(param_dict, json_file)


if __name__ == "__main__":
    parse_dirs()
