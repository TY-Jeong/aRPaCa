#!/usr/bin/env python
from arpaca.utils import *
import argparse

parser = argparse.ArgumentParser(description='Process chemical formula input.')
parser.add_argument('-c', '--chemical_formula', type=str, required=False, default=None, help='Chemical formula to process')
args = parser.parse_args()
chem_formula = args.chemical_formula

if chem_formula is None:
    print("No arguments provided. Use the '--help' option to see usage details if you want.")
else:
	print(f"Chemical formula provided: {chem_formula}")

Crystal = GenBulk(chem_formula)
Crystal.view_data()