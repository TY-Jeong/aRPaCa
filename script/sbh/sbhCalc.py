#!/usr/bin/env python

import argparse
from arpaca.sbh.sbh import *
from datetime import datetime

parser = argparse.ArgumentParser(description="Script to get VASP results.")
parser.add_argument("result_path", type=str, help="Path for results.")
parser.add_argument("temperature", type=str, help="Absolute temperature.")

args = parser.parse_args()
result_path = args.result_path
temperature = args.temperature

result_dir = os.path.dirname(result_path)
if result_dir and not os.path.exists(result_dir):
    os.makedirs(result_dir)

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

SBH = SBPlot(result_path=result_path, Temperature = float(temperature))
#SBH = SBPlot(result_path='/home/ysj/aRPaCa/script/sbh/Project', Temperature = 300)

end_time = datetime.now()
execution_time = (end_time - start_time).total_seconds()

print(f"\n\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Execution time: {execution_time} seconds")