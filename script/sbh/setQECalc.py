#!/usr/bin/env python

import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to get VASP results.")

parser.add_argument(
    "--path",
    "-p",
    type=str,
    default=None,
    required=False,
    help="Path of results of surface calculations")

args = parser.parse_args()
surface_path = args.path

if surface_path is None:
    print("No surface path provided. Use the '--help' option to see usage details if you want.\n")
else:
    print(f"Processing QE inputs with: {surface_path}")

result = GetSBHResult()
result.qe_setup( surface_path, scheduler='Auto') 
