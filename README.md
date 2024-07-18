# aRPaCa
---
**aRPaCa** (***a****b-initio* **R**RAM **Pa**rameter **Ca**lculator) is a python package for calculating parameters used in RRAM simulation, which contains modules for pre- or post-processing of DFT calculation using VASP. (Note: aRPaCa is written baed on VASP 5.4.4.)

#### <ins>Functions in aRPaCa</ins>:

1. Tools for **structure generation**.
2. **Diffusion coefficient** calculation using *ab initio* molecular dynamics
3. **Schottky profile** of metla/oxide interface ${\textsf{\color{gray} will be update}}$
4. **Electric conductivity** of oxide with arbitrary composition ${\textsf{\color{gray} will be update}}$

#### <ins>Contents</ins>:
* **Installation**

* **Step by step exampe**
  * Structure generation
    * Amorphous generation
    * Crystallin generation ${\textsf{\color{gray} will be update}}$
    * Interface generation ${\textsf{\color{gray} will be update}}$

  * **Parameter calculation**
    * Mass transport paramters
    * Electrical conductivity ${\textsf{\color{gray} will be update}}$
    * Schottky profile ${\textsf{\color{gray} will be update}}$

* **Module details**
  * amorphous
  * einstein
    * einstein.file_manager
    * einstein.einstein
  * ${\textsf{\color{gray} will be update}}$
  * ${\textsf{\color{gray} will be update}}$
  * ${\textsf{\color{gray} will be update}}$
  * ${\textsf{\color{gray} will be update}}$
---
## Install

${\textsf{\color{gray} will be update}}$

For generating amorphous, aRPaCa uses [Packmol](https://m3g.github.io/packmol/download.shtml), the open-source code providing a good cadidate for the initial structure. Please make sure Packmol is installed on your system, and add the path of the packmol excutable into **aRPaCa/data/path.dat**. Below is an example of **path.dat**
```ruby
POT_PBE=/home/taeyoung/POT/POT_PBE
POT_LDA=/home/taeyoung/POT/POT_LDA
packmol=/home/taeyoung/Downloads/packmol-20.14.3/packmol
```
---
## Step by step example
### Structure generation
---
### Amorphous generation
#### Step 1: Generate input files for melt-quench
Using **arpaca.amorphous** module, you can conveniently get input files for amorphous generation
```ruby
from arpaca.amorphous import *

genAmorphous(density=10.00, chem_formula="Hf34O68") # POSCAR will be generated
genInput() # INCAR, KPOINTS, POTCAR will be generated
```
This code will generate a cubic structure containing 34 Hf atoms and 68 O atoms, whose density is 10.00 g/cm<SUP>3</SUP>.
It should be note that when the user write **chem_formula**, number 1 cannnot be omitted. For example, if user try to generate an amorphous containing 34 Hf, 68 O and 1 Ag, **chem_formula** should be Hf34O68Ag1 not Hf34O68Ag.

#### Step 2: Run VASP

#### Step 3: Extract structures from MD trajectory.
The below code will gernerate POSCAR files using the XDATCAR file from previous step.

```ruby
from arpaca.amorphous import *

xdat2pos('XDATCAR', 1000, 10000, 1000) # numpy.arange(1000, 10000+1, 1000)
```
The last three arguments are *start*, *end*, and *step* in numpy.arange(*start*, *end*, *step*). The generated POSCAR files will be saved in **ensmebles** directory.

### Crystallin generation
${\textsf{\color{gray} will be update}}$

### Interface generation
${\textsf{\color{gray} will be update}}$

---
### Parameter calculation
---
### Mass transport paramters
aRPaCa uses the **Einstein relation** to calculate the mass transport parameters of oxygen vacancies in both amorphous and crystalline materials. Due to the statistical nature of this method, large ensembles are needed to ensure reliable calculations. However, the computational burden of DFT calculations limits the size of these ensembles. To address this issue, aRPaCa provides functions that integrate multiple individual MD simulations. For example, if the user runs 10 different MD simulations using amorphous Hf<SUB>34</SUB>O<SUB>68</SUB>, aRPaCa can integrate these 10 simulations to achieve the same effect as a single simulation in a cell of Hf<SUB>340</SUB>O<SUB>680</SUB>.

After the following steps, the user can obtain **diffusion barrier (E<SUB>a</SUB>)** and **pre-exponential of diffusivity (D<SUB>0</SUB>)**.

#### Step 1: Generate MD simulation sets
To calculate **diffusion barrier (E<SUB>a</SUB>)** and **pre-exponential of diffusivity (D<SUB>0</SUB>)**, MD trajectories from various temperatures are required. The user can generate the MD simulation sets using **einstein.file_manager.getMDset module**.

```ruby
from arpaca.einstein.file_manager import *

getMDset(path_poscar='ensembles',
         temp=np.arange(1500, 2000+1, 100))
```
The **path_poscar** refers to directory path containing POSCAR files, which are named in format **POSCAR_{label}**.
The user can get the POSCAR files using **amorphous.xdat2pos** module.
```
ensembles\
    POSCAR_01
    POSCAR_02
    POSCAR_03 
    POSCAR_04
    POSCAR_05
```
By running the code, the directories for MD simulation are generated.
```
1500K\
    01\
        INCAR
        KPOINTS
        POSCAR # corresponding to ensembles/POSCAR_01
        POTCAR
    ⋮
    05\
        INCAR
        KPOINTS
        POSCAR # corresponding to ensembles/POSCAR_05
        POTCAR
1600K\
    01\
        INCAR
        KPOINTS
        POSCAR # corresponding to ensembles/POSCAR_01
        POTCAR
    ⋮
    05\ # omitted
1700K\ # omitted
1800K\ # omitted
1900K\ # omitted
2000K\ # omitted
```
#### Step 2: Run VASP

#### Step 3: Gather MD trajectories
The user can gather the MD trajectories (XDATCAR files) using below code.
```ruby
from arpaca.einstein.file_manager import *

getMDresult()
```


---
## Diffusion coefficient
**Diffusion coefficient (*D*)** of amorphous is calculated based in **Einstein relation**.
Since this is a statistical methods, user should use enough ensembles of amorphous to ensure the reliability.
With aRPaCa, the user can easily integrate the MD trajectory of many ensembles. 
