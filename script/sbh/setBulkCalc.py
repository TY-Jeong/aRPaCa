#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process bulk POSCAR file.")
parser.add_argument("poscar_path", type=str, help="Path to the POSCAR file.")
args = parser.parse_args()

poscar_path = args.poscar_path

if not os.path.isfile(poscar_path):
    print(f"Error: The file {poscar_path} does not exist.")
    exit(1)

Bulk_calc = BulkSet(poscar_path)
Bulk_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='ykh', scheduler='pbs')

# #RuO2_calc = BulkSet()
# RuO2_calc = BulkSet('Project/Bulk/RuO2/POSCAR_mp-825')
# RuO2_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='ykh', scheduler='pbs')

# #TiO2_calc = BulkSet()
# TiO2_calc = BulkSet('Project/Bulk/TiO2/POSCAR_mp-2657')
# TiO2_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='ykh', scheduler='pbs')