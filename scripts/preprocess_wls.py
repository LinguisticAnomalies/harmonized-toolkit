"""
This script is designed for extracting cookie theft task text from WLS dataset
"""


import os
import re
import string
import pandas as pd


def clean_text(text_chunk):
    """
    clean the text to keep the participant transcript only,
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
            line = re.sub(r'\&\=clears\s+throat', r' ', line)
            # open parentheses e.g, (be)coming
            line = re.sub(r'\((\w+)\)(\w+)', r'\1\2', line)
            # open square brackets eg. [: overflowing] - error replacements
            line = re.sub(r'\s+\w+\s+\[\:\s+([^\]]+)\]', r' \1 ', line)
            # remove disfluencies prefixed with "&"
            line = re.sub(r'\&\w+\s+', r' ', line)
            # remove unitelligible words
            line = re.sub(r'xxx', r' ', line)
            # remove pauses eg. (.) or (..)
            line = re.sub(r'\(\.+\)', r' ', line)
            # remove forward slashes in square brackets
            line = re.sub(r'\[\/+\]', r' ', line)
            # remove noise indicators eg. &=breath
            line = re.sub(r'\&\=\S+\s+', r' ', line)
            # remove turn identifiers
            line = re.sub(r'\*PAR\:', r' ', line)
            # remove star or plus and material
            # inside square brackets indicating an error code
            line = re.sub(r'\[(\*|\+|\%)[^\]]+\]', r' ', line)
            line = re.sub(r'\[(\=\?)[^\]]+\]', r' ', line)
            # finally remove all non alpha characters
            # line = "<s> "+ line + "</s>" # format with utterance start and end symbols
            line = re.sub(r'[^A-Za-z\n \']', '', line)
            line = re.sub(r'\s+', ' ', line)
            # replace multiple spaces with a single space
            line = line.capitalize()
            # if line is not empty
            if line.strip():
                tran += line.strip()
                tran += ". "
    return tran


def parse_dirs(input_path):
    """
    extract task specifc transcript from raw dataset,
    return the cleaned dataframe
    :param input_path: the folder to transcripts
    :type input_path: str
    :return: the cleaned dataframe
    :rtype:pd.DataFrame
    """
    task_df = pd.DataFrame(columns=["file", "text"])
    for subdir, _, files in os.walk(input_path):
        for file in files:
            with open(os.path.join(subdir, file), mode="r", errors="ignore") as file_content:
                all_tran = file_content.read()
                try:
                    text = re.search(r'@Bg:	Activity\n.*?@Eg:	Activity',
                                     all_tran, re.DOTALL).group()
                    tran = clean_text(text)
                    tran_row = {"file": "20000" + file.split(".")[0],
                                "text": tran}
                    task_df = task_df.append(tran_row, ignore_index=True)
                except AttributeError:
                    # if no qualified transcript
                    pass
    return task_df


def add_meta(tran_df):
    """
    add meta data to WLS dataset

    :param tran_df: the 
    :type tran_df: pd.DataFrame
    """
    meta_df = pd.read_csv("../WLS/meta.tsv", sep="\t")
    meta_df.rename(columns={"idtlkbnk":"ids"}, inplace=True)
    meta_df["ids"] = meta_df["ids"].astype(str)
    wls_tran.rename(columns={"file": "ids"}, inplace=True)
    wls_meta = pd.merge(tran_df, meta_df, on="ids")
    # add uid
    wls_meta["uid"] = wls_meta.groupby(["ids"]).ngroup()
    # start from 291, which is the total uid from pitt corpus
    wls_meta["uid"] = wls_meta["uid"] + 292
    # save to local file
    wls_meta.sort_values(by=["uid"], inplace=True)
    wls_meta.to_csv("../wls_full.tsv", sep="\t", index=False)


def extract_ids():
    """
    keep uids and file name as a single file
    """
    try:
        wls_full = pd.read_csv("../wls_full.tsv", sep="\t")
    except FileNotFoundError:
        wls_tran = parse_dirs("../WLS/")
        add_meta(wls_tran)
        wls_full = pd.read_csv("../wls_full.tsv", sep="\t")
    wls_ids = wls_full[["ids", "uid"]]
    wls_ids.to_csv("../wls_ids.tsv", sep="\t", index=False)


def separate_full_data():
    """
    for the full merged dataset, only keep filename, uid and metadata,
    save it as separated dataset
    """
    try:
        wls_full = pd.read_csv("../wls_full.tsv", sep="\t")
    except FileNotFoundError:
        wls_tran = parse_dirs("../WLS/")
        add_meta(wls_tran)
        wls_full = pd.read_csv("../wls_full.tsv", sep="\t")
    # drop transcript column
    wls_full.drop("text", axis=1, inplace=True)
    # change column to more appropriate names
    wls_columns = wls_full.columns.values.tolist()
    new_columns = []
    for item in wls_columns:
        item_piece = re.findall(r"\w+|[^\w\s]", item.lower(), re.UNICODE)
        item_piece = [v for v in item_piece if v not in string.punctuation]
        new_columns.append("_".join(item_piece))
    wls_full.columns = new_columns
    # remove other unwanted columns
    wls_full.drop(wls_full.filter(regex="unnamed").columns, axis=1, inplace=True)
    # fill the NA in the age column
    wls_full["age_2004"] = wls_full["age_2011"] - 7
    wls_full.to_csv("../wls_final_meta.tsv", sep="\t", index=False)


if __name__ == "__main__":
    wls_tran = parse_dirs("../WLS/")
    add_meta(wls_tran)
    separate_full_data()
    extract_ids()
