#!/usr/bin/env python

import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to get VASP results.")
parser.add_argument("result_path", type=str, help="Path for results.")

args = parser.parse_args()
result_path = args.result_path
result_dir = os.path.dirname(result_path)
if result_dir and not os.path.exists(result_dir):
    os.makedirs(result_dir)

result = GetSBHResult(result_path)

pseudopotentials = {
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"
}
result.qe_setup(pseudopotentials, scheduler='pbs')
