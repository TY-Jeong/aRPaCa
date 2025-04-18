#!/usr/bin/env python
import argparse
from arpaca.utils import *


parser = argparse.ArgumentParser(description="Script to process the interface POSCAR file.")

parser.add_argument('-p', '--path', type=str, required=False, default=None, help="Path to the interface POSCAR file.")


# 입력된 경로 파싱
args = parser.parse_args()

# 경로 변수 할당
interface_path = args.path

if interface_path is None:
    print("No arguments provided. Use the '--help' option to see usage details if you want.")
else:
    print(f"Processing interface VASP file at: {interface_path}")


interface_calc = InterfaceSet(interface_path, ncore=4)

# interface_calc = InterfaceSet("Project/Interface/interface.vasp",ncore=4)
interface_calc.runfile_setup(runfile_name='run.sh', nodes='4', processors='24', queue_name='None', scheduler='Auto')