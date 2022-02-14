'''
This script contains multiple utility functions that are used in the preprocessing and toolkit
'''
import json
import time


def read_json(file_name):
    """
    read the user-generated json file and save it as dict,
    return the dict

    :param file_name: the name of generated customized json file
    :type file_name: str
    """
    with open(file_name, "r") as json_file:
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
    elif value.lower() in ("n", "no", "0", 0):
        return False
    else:
        raise ValueError("Wrong response, please double check...")


def convert_time(input_time):
    """
    convert time in seconds into hh:mm:ss format

    :param input_time: time in seconds
    :type input_time: float
    """
    return time.strftime('%H:%M:%S', time.gmtime(input_time))