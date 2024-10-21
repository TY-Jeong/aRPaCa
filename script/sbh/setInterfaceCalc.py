#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process the interface POSCAR file.")
parser.add_argument("interface_path", type=str, help="Path to the interface POSCAR file.")

# 입력된 경로 파싱
args = parser.parse_args()

# 경로 변수 할당
interface_path = args.interface_path

# 파일 존재 여부 확인
if not os.path.isfile(interface_path):
    print(f"Error: The interface VASP file {interface_path} does not exist.")
    exit(1)

# 이후 interface VASP 파일을 처리하는 코드 작성
print(f"Processing interface VASP file at: {interface_path}")


interface_calc = InterfaceSet(interface_path, ncore=4)

# interface_calc = InterfaceSet("Project/Interface/interface.vasp",ncore=4)
interface_calc.runfile_setup(runfile_name='run.sh', nodes='4', processors='24', queue_name='ykh', scheduler='pbs')