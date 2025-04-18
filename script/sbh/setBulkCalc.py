#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process bulk POSCAR file.")
parser.add_argument('-p', '--path', type=str, required=False, default=None, help="Path to the POSCAR file.")
args = parser.parse_args()

poscar_path = args.path

if poscar_path is None:
    print("No arguments provided. Use the '--help' option to see usage details if you want.")
else:
	print(f"Bulk structure path: {poscar_path}")

Bulk_calc = BulkSet(poscar_path)
Bulk_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='None', scheduler='Auto')