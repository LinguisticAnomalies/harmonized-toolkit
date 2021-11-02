"""
This script is designed to merge ADReSS and DementiaBank dataset into one unified dataset
"""
import os
import re
import pandas as pd
import numpy as np


def get_db_meta(input_folder):
    """
    extract metadata, including age, gender, diagnosis and mmse from .cha file,
    return tehe meta dataframe

    :param input_folder: the input folder contains .cha files
    :type input_folder: str
    :return: the dataframe with meta data to corresponding .cha files
    :rtype: pd.DataFrame
    """
    meta_df = pd.DataFrame(columns=["file", "age", "gender", "dx", "mmse"])
    # iterate throughout the input folder
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            with open(os.path.join(subdir, file)) as fp:
                line = fp.readline()
                while line:
                    if re.match(r"\@ID:", line):
                        # participant only
                        meta_chunks = line.strip().split("|")
                        if meta_chunks[2] == "PAR":
                            meta_dict = {"file": file.split(".")[0], "gender": meta_chunks[4],
                                         "age": re.sub('[^0-9]','', meta_chunks[3]),
                                         "dx": meta_chunks[5], "mmse": re.sub('[^0-9]','', meta_chunks[8])}
                            meta_df = meta_df.append(meta_dict, ignore_index=True)
                    line = fp.readline()
    return meta_df


def read_txt_to_df(input_folder):
    """
    read cleaned .txt transcritps as dataframe,
    return the dataframe

    :param input_folder: the input folder contains .txt files
    :type input_folder: str
    """
    tran_df = pd.DataFrame(columns=["file", "text"])
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            with open(os.path.join(subdir, file)) as fp:
                tran = fp.readlines()
                tran = [item.strip() for item in tran]
                # remove empty sentence
                tran = [item for item in tran]
                tran = ". ".join(tran)
                tran_row = {"file": file.split(".")[0], "text": tran}
                tran_df = tran_df.append(tran_row, ignore_index=True)
    return tran_df

def merge_db():
    """
    merge DementiaBank transcripts with meta data
    """
    pitt_control = "../Pitt/Control/cookie/"
    pitt_dementia = "../Pitt/Dementia/cookie/"
    meta_control_df = get_db_meta(pitt_control)
    meta_dementia_df = get_db_meta(pitt_dementia)
    pitt_clean_control = "../Pitt-clean/control"
    pitt_clean_dementia = "../Pitt-clean/dementia"
    # read pre-processed .txt files for DementiaBank dataset
    pitt_clean_control_tran = read_txt_to_df(pitt_clean_control)
    pitt_clean_dementia_tran = read_txt_to_df(pitt_clean_dementia)
    pitt_full_control = pd.merge(pitt_clean_control_tran, meta_control_df, on="file")
    pitt_full_dementia = pd.merge(pitt_clean_dementia_tran, meta_dementia_df, on="file")
    pitt_full = pitt_full_control.append(pitt_full_dementia, ignore_index=True)
    print("pitt corpus size: {}".format(pitt_full.shape))
    # read ADReSS meta dataframe
    adr_train = pd.read_csv("../ADReSS-IS2020-data/ids/adress_train_id_maptoDB.csv")
    adr_train = adr_train[["file_x", "label"]]
    adr_train["file_x"] = adr_train["file_x"].str.split(".").str[0]
    adr_train.rename(columns={"file_x": "file"}, inplace=True)
    adr_test = pd.read_csv("../ADReSS-IS2020-data/ids/adress_test_id_maptoDB.csv")
    adr_test = adr_test[["file_x", "label"]]
    adr_test["file_x"] = adr_test["file_x"].str.split(".").str[0]
    adr_test.rename(columns={"file_x": "file"}, inplace=True)
    adr_full = adr_train.append(adr_test, ignore_index=True)
    adr_full = adr_full[["file", "label"]]
    print("ADReSS dataset size: {}".format(adr_full.shape))
    # add indicators to pitt corpus
    adr_files = adr_train["file"].values.tolist() + adr_test["file"].values.tolist()
    pitt_full["inADReSS"] = np.where(pitt_full["file"].isin(adr_files), 1, 0)
    pitt_full["ADReSS_train"] = np.where(pitt_full["file"].isin(adr_train["file"].values.tolist()), 1, 0)
    pitt_full["ADReSS_test"] = np.where(pitt_full["file"].isin(adr_test["file"].values.tolist()), 1, 0)
    # add label to pitt corpus
    pitt_full = pd.merge(pitt_full, adr_full, on="file", how="outer")
    print("merged pitt corpus size: {}".format(pitt_full.shape))
    # add uid to this dataset
    pitt_full["pid"] = [item.split("-")[0] for item in pitt_full["file"].values.tolist()]
    pitt_full["uid"] = pitt_full.groupby(["pid"]).ngroup()
    pitt_full.sort_values(by=["file"], inplace=True)
    pitt_full.to_csv("../pitt_merged.tsv", sep="\t", index=False)
    

if __name__ == "__main__":
    merge_db()
