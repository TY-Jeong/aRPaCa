#!/bin/sh


# 1. 결정 (Bulk crystalline) 구조 및 VASP 계산셋 생성 

# Input: chemical formula
# Output: POSCAR(bulk crystal)
./genBulk.py
./setBulkCalc.py



# 2. 표면 (Surface) 구조 생성

# Input: CONTCAR (bulk crystal) & Miller indices
# Output: POSCAR (surface)
./genSurface.py



# 3. 계면 (Interface) 구조 및 VASP 계산셋 생성

# Input: POSCAR (surface)
# Output: POSCAR (interface)
./genInterface.py
./setInterfaceCalc.py



# 4. VASP 계산 결과 후처리
./getVASPResult.py
# 5. QE CBS 계산셋 생성
./setQECalc.py
# 6. QE 계산 결과 후처리
./getQEResult.py



# 7. SBH 계산

# Input: Temperature, VASP & QE results
# Output: SB profile, SBH
./sbhCalc.py