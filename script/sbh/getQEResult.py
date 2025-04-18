#!/usr/bin/env python

import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to get VASP results.")

parser.add_argument(
    "--path",
    "-p",
    type=str,
    default=None,
    help="Path of surface calculation containing the results of complex band structure calculation")
args = parser.parse_args()
result_path = args.path

if result_path is None:
    print("No surface path provided. Use the '--help' option to see usage details if you want.\n")
else:
    print(f"Processing QE results with: {result_path}")

result = GetSBHResult()
result.get_qe_result(result_path)
