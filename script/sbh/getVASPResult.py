#!/usr/bin/env python

import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to get VASP results.")

parser.add_argument(
    "--bulk_path",
    "-b",
    type=str,
    default=None,
    required=False,
    help="Path of results of bulk calculations")

parser.add_argument(
    "--interface_path",
    "-i",
    type=str,
    default=None,
    required=False,
    help="Path of results of interface calculations")



args = parser.parse_args()
bulk_path = args.bulk_path
interface_path = args.interface_path

if bulk_path is None:
    print("No bulk path provided. Use the '--help' option to see usage details if you want.\n")
else:
    print(f"Processing bulk VASP results at: {bulk_path}")

if interface_path is None:
    print("No interface path provided. Use the '--help' option to see usage details if you want.\n")
else:
    print(f"Processing interface VASP results at: {interface_path}")



result = GetSBHResult()
result.get_vasp_result(bulk_path, interface_path)