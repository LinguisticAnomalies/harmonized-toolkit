"""
This script generates pre-processing text parameters for WLS or DB with user inputs
"""
import csv
import os
import json
import re
import pandas as pd
from util_fun import read_json, return_bool


def clean_text(text_chunk, param_dict):
    """
    replace windows line breaker into \n
    clean the text to keep the participant transcript only, based on users' selection
    return the cleaned text

    :param text_chunk: qualified text chunk selected from original transcript
    :type text_chunk: str
    :return: the cleaned participant transcript
    :rtype: str
    """
    text_chunk = text_chunk.replace("\r\n", " ")
    text_chunk = text_chunk.replace("\n", " ")
    # remake the new line
    text_chunk = re.sub(r"((\*|\%|\@)[A-Za-z]+\:)", r"\n\1", text_chunk)
    all_sents = text_chunk.split("\n")
    tran = ""
    starts = []
    ends = []
    lines = []
    for line in all_sents:
        if re.match(r"\*PAR:", line):
            line = re.sub(r"\*PAR:", "", line)
            pattern = re.findall(r"\d+\_\d+", line)
            start = 0
            end = 0
            if pattern:
                ts_intervals = pattern[0].split("_")
                # convert millisecond to second
                ts_intervals = [int(item)/1000 for item in ts_intervals]
                start = ts_intervals[0]
                end = ts_intervals[1]
            line = line.replace("_", " ")
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
                line = re.sub(r'(\w)\1\1', r'', line)
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
                lines.append(line.strip())
                starts.append(start)
                ends.append(end)
                # period at the end of sentence
                if return_bool(param_dict["eos_period"]):
                    tran += ". "
                # newline at the end of sentence
                if return_bool(param_dict["eos_newline"]):
                    tran += "\n"
    return tran, lines, starts, ends


def parse_dirs():
    """
    extract task specifc transcript from raw dataset,
    pre-processed as users' selections,
    save it to local path
    """
    param_dict = read_json("text_process.json")
    if os.path.exists(param_dict["out_path"]):
        os.remove(param_dict["out_path"])
    out_cha_path = param_dict["out_path"].replace(".tsv", "_cha.csv")
    full_lines = []
    full_indexes = []
    full_starts = []
    full_ends = []
    full_files = []
    full_cha_df = pd.DataFrame()
    with open(param_dict["out_path"], "a") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(["file", "text"])
        for subdir, _, files in os.walk(param_dict["input_path"]):
            for file in files:
                if file.endswith(".cha"):
                    with open(
                        os.path.join(subdir, file),
                        mode="r", errors="ignore") as file_content:
                        all_tran = file_content.read()
                        if param_dict["dataset_choice"].lower() == "wls":
                            file_name = "20000" + file.split(".")[0]
                            try:
                                text = re.search(
                                    r'@Bg:	Activity\n.*?@Eg:	Activity', all_tran, re.DOTALL).group()
                                tran, lines, starts, ends = clean_text(text, param_dict)
                                index = [i for i in range(0, len(starts))]
                                full_files.extend([file_name]*len(starts))
                                full_lines.extend(lines)
                                full_starts.extend(starts)
                                full_ends.extend(ends)
                                full_indexes.extend(index)
                                tsv_writer.writerow([file_name, tran])
                                inv_ts = [[i, j] for i, j in zip(starts, ends)]
                                param_dict[file_name] = inv_ts
                            except AttributeError:
                                # if no qualified transcript
                                pass
                        elif param_dict["dataset_choice"].lower() == "db":
                            file_name = file.split(".")[0]
                            tran, lines, starts, ends = clean_text(all_tran, param_dict)
                            index = [i for i in range(0, len(starts))]
                            full_files.extend([file_name]*len(starts))
                            full_lines.extend(lines)
                            full_starts.extend(starts)
                            full_ends.extend(ends)
                            full_indexes.extend(index)
                            tsv_writer.writerow([file_name, tran])
                            inv_ts = [[i, j] for i, j in zip(starts, ends)]
                            param_dict[file_name] = inv_ts
                        else:
                            raise ValueError("Dataset is not supported, please double check...")
    full_cha_df["file"] = full_files
    full_cha_df["index"] = full_indexes
    full_cha_df["trans"] = full_lines
    full_cha_df["start"] = full_starts
    full_cha_df["end"] = full_ends
    full_cha_df.to_csv(out_cha_path, index=False)
    # rewrite param_dict with investigator timestep
    with open("text_process.json", "w") as json_file:
        json.dump(param_dict, json_file)


if __name__ == "__main__":
    parse_dirs()
