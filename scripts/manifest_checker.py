"""
This is a test-run file for manifest format checking.
Please check Reporting and Submitting Manifest section in the README
for more details about the template
"""

import argparse
import json
import sys
from util_fun import read_json



def parse_args():
    """
    Add arguments to this Python script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str,
        help="""The location to your manifest for submission""")
    return parser.parse_args()


def check_file(manifest_loc):
    """
    The test function for manifest template

    :param manifest_loc: the location of submission manifest
    :type manifest_loc: str
    """
    manifest = read_json(manifest_loc)
    # You manifest must contain the following key-value pairs
    must_keys = [
        'pre_process', 'data_uids', 'positive_uids', 'negative_uids', 'training_uids', 'test_uids', 'evaluation']
    try:
        assert all(k in manifest.keys() for k in must_keys)
    except AssertionError:
        sys.stdout.write(
            "Your manifest did not include all required keys.\
                Please double check your manifest before submission\n")
    sys.stdout.write("Congraduations, you manifest passed our checking system.\n")


if __name__ == '__main__':
    args = parse_args()
    check_file(args.manifest)
