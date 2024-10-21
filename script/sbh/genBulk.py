#!/usr/bin/env python
from arpaca.utils import *
import argparse

parser = argparse.ArgumentParser(description='Process chemical formula input.')
parser.add_argument('-f', '--formula', type=str, required=True, help='Chemical formula to process')
args = parser.parse_args()
chem_formula = args.formula
print(f"Chemical formula provided: {chem_formula}")

Crystal = GenBulk(chem_formula)
Crystal.view_data()

# RuO2=GenBulk('RuO2')
# RuO2.view_data()
# TiO2=GenBulk('TiO2')
# TiO2.view_data()

