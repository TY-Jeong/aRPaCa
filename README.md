# aRPaCa
---
**aRPaCa** (***a****b-initio* **R**RAM **Pa**rameter **Ca**lculator) is a python package for calculating parameters used in RRAM simulation, which contains modules for pre- or post-processing of DFT calculation using VASP. (Note: aRPaCa is written baed on VASP 5.4.4.)

#### Functions in aRPaCa:

1. Tools for **structure generation**. (amorphous, crystalline, and interface)
2. **Diffusion coefficient** calculation using *ab initio* molecular dynamics (AIMD)
3. **Schottky profile** of metla/oxide interface (*will be updated*)
4. **Electric conductivity** of oxide with arbitrary composition (*will be update*)

#### Contents:

---
## Install

*will be updated*

---
## Amorphous module
Amorphous generation is two-step process: (1) initial amorphous structure generation and (2) generating ensembles of amorphous using melt-quench method.
Amorphous module help the users carry out this process conveniently.

For amorphous generation, aRPaCa use [Packmol](https://m3g.github.io/packmol/download.shtml) softwawre.
Please make sure Packmol is installed on your device, and add the path of the packmol excutable into ***{where aRPaCa exist}/RPaCa/data/path.dat***.

Below is an example of *path.dat*
```ruby
POT_PBE=/home/taeyoung/POT/POT_PBE
POT_LDA=/home/taeyoung/POT/POT_LDA
packmol=/home/taeyoung/Downloads/packmol-20.14.3/packmol
```
### 1. Generating initial amorpohus structure
User can generate an initial amorphous structure usin one-line command.
```ruby
from amorphous import *
genAmorphous(density=10.00, chem_formula="Hf34O68", outfile='POSCAR_Hf34O68')
```
This code will generate a cubic structure containing 34 Hf atoms and 68 O atoms, whose density is 10.00 g/cm<SUP>3</SUP>.
It should be note that when the user write **chem_formula**, number 1 cannnot be omitted. For example, if user try to generate an amorphous containing 34 Hf, 68 O and 1 Ag, **chem_formula** should be Hf34O68Ag1 not Hf34O68Ag.

*description for arguments will be updated*

### 2. Generating VASP inputs for MD
User can generate INCAR, KPOINTS, POTCAR, POSCAR files using below command lines.
```ruby
from amorphous import *
genAmorphous(density=10.00, chem_formula="Hf34O68")  # POSCAR will be generated
genInput()  # INCAR, KPOINTS, POTCAR will be generated
```

*description for arguments will be updated*

### 3. Generating NVT ensembles from MD
Below command lines generates NVT ensemble set from XDATCAR file
```ruby
from amorphous import *
xdat2pos('XDATCAR', 1000, 10000, 1000)
```
Last three numbers are *start*, *end* and *step* in *numpy.arange(start, end, step)*

*description for arguments will be updated*

---
## Diffusion coefficient
**Diffusion coefficient (*D*)** of amorphous is calculated based in **Einstein relation**.
Since this is a statistical methods, user should use enough ensembles of amorphous to ensure the reliability.
With aRPaCa, the user can easily integrate the MD trajectory of many ensembles. 
