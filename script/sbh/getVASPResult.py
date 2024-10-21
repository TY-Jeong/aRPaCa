#!/usr/bin/env python

import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to get VASP results.")
parser.add_argument("result_path", type=str, help="Path to save the result file.")
args = parser.parse_args()
result_path = args.result_path
result_dir = os.path.dirname(result_path)
if result_dir and not os.path.exists(result_dir):
    os.makedirs(result_dir)

#print(f"Result will be saved at: {result_path}")

result = GetSBHResult(result_path)
result.get_vasp_result()