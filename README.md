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
---
## Structure generation

### Amorphous generation
#### Step 1: Generate input files for melt-quench
Using **arpaca.amorphous** module, you can conveniently get input files for amorphous generation
```ruby
from arpaca.amorphous import *

genAmorphous(density=10.00, chem_formula="Hf34O42") # POSCAR will be generated
genInput() # INCAR, KPOINTS, POTCAR will be generated
```
This code will generate a cubic structure containing 34 Hf atoms and 42 O atoms, with a density of 10.00 g/cm<SUP>3</SUP>. Note that when specifying the chem_formula, the number 1 cannot be omitted. For example, to generate an amorphous structure containing 34 Hf atoms, 42 O atoms, and 1 Ag atom, the **chem_formula** should be written as Hf34O42Ag1, not Hf34O42Ag.

#### Step 2: Run VASP

By running VASP, the user can get NVT ensmbles.

#### Step 3: Extract structures from MD trajectory.
The below code will gernerate POSCAR files using the XDATCAR file from previous step.

```ruby
from arpaca.amorphous import *

xdat2pos('XDATCAR', 1000, 10000, 1000) # numpy.arange(1000, 10000+1, 2000)
```
The last three arguments are *start*, *end*, and *step* in numpy.arange(*start*, *end*, *step*). The generated POSCAR files will be saved in **ensmebles** directory.

### Crystallin generation
${\textsf{\color{gray} will be update}}$

### Interface generation
${\textsf{\color{gray} will be update}}$

---
## Parameter calculation

### Mass transport paramters
aRPaCa uses the **Einstein relation** to calculate the mass transport parameters of oxygen vacancies in both amorphous and crystalline materials. Due to the statistical nature of this method, large ensembles are needed to ensure reliable calculations. However, the computational burden of DFT calculations limits the size of these ensembles. To address this issue, aRPaCa provides functions that integrate multiple individual MD simulations. For example, if the user runs 10 different MD simulations using amorphous Hf<SUB>34</SUB>O<SUB>68</SUB>, aRPaCa can integrate these 10 simulations to achieve the same effect as a single simulation in a cell of Hf<SUB>340</SUB>O<SUB>680</SUB>.

After the following steps, the user can obtain **diffusion barrier (E<SUB>a</SUB>)** and **pre-exponential of diffusivity (D<SUB>0</SUB>)**.

#### Step 1: Generate MD simulation sets
To calculate mass transport parameters, MD trajectories from various temperatures are required. The user can generate the MD simulation sets using **einstein.file_manager.getMDset module**.

```ruby
from arpaca.einstein import file_manager as fm

fm.getMDset(path_poscar='ensembles',
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
    05\ # contents are omitted
1700K\ # contents are omitted
1800K\ # contents are omitted
1900K\ # contents are omitted
2000K\ # contents are omitted
```
#### Step 2: Run VASP

#### Step 3: Gather MD trajectories
The user can gather the MD trajectories (XDATCARs) using below code.
```ruby
from arpaca.einstein import file_manager as fm

fm.getMDresult()
```
The MD trajectoris (XDATCARs) and information of each calculation (OUTCARs) will be collected in **xdatcar** directory.
```
xdatcar\
    xdatcar.1500K\
        OUTCAR
        XDATCAR_01 # corresponding to POSCAR_01
           ⋮
        XDATCAR_05 # corresponding to POSCAR_05
   xdatcar.1600K\ # contents are omitted
   xdatcar.1700K\ # contents are omitted
   xdatcar.1800K\ # contents are omitted
   xdatcar.1900K\ # contents are omitted
   xdatcar.2000K\ # contents are omitted
```
#### Step 4: Calculate mass transprot parameters
``` ruby
from arpaca.einstein import einstein as ein

ein.getDiffusivity(symbol='O',
                   path_xdatcar='./xdatcar',
                   skip=500, # optional; exclude first 500 steps.
                   segment=2, # optional; divide total MD step into two segment. (ex. 15000 step -> two 7500 step)
                   start=1000, # optional; assign fitting range.
                   xyz=False # optional; if True, x, y, and z component of D is calcualted.
                  )
```
After running the code, the user will obtaun two images (or three if '**xyz=True'**'). The first imgage is the MSD (mean squred displacemnt) graph.
According to Einstein relation, MSD and time should have a linear relationship (i.e., $MSD = 6Dt$). Therefore, the user should check that the MSD graphs are sufficiently linear.
Below is an example of the MSD graph.
<p align="center">
<img src="https://github.com/user-attachments/assets/ff7afbc7-6477-49c7-b34d-4be764c80c46" width="400" height="300"/>
</p>

The two black vertical lines indicate the range of linear fitting used to calculate $D$. The fittung range can be adjusted using **start** and **end** arguments. If **end** is not specified, it will be automatically set to the right end. 



The second image is Arrheniys plot. (i.e., $ln D = ln D_{0} - \frac{E_{a}}{k_{B}} \frac{1}{T}$)
<p align="center">
<img src="https://github.com/user-attachments/assets/ec2f42e9-04e2-4b7f-be61-721ff821e2ea" width="400" height="300"/> 
</p>
To ensure the reliability of calculations, the points should be well alighned with the fitting line.


The calculated **diffusion barrier (E<SUB>a</SUB>)** and **pre-exponential of diffusivity (D<SUB>0</SUB>)** values will be written in **D.txt** file.
```
# D.txt
Ea = 0.7442163575033494 eV
D0 = 2.073189097303261e-07 m2/s

T (K)   D (m2/s)
1500    6.170153833929504e-10
1600    1.0062298779273406e-09
1700    1.32240515177761e-09
1800    1.7033480801825758e-09
1900    2.135080855776783e-09
2000    2.7588899714142115e-09
```

#### Appendix : Explore an individal ensemble
Below is an example of a method to explore an individual ensemble.
```ruby
import numpy as np
from arpaca.einstein import einstein as ein

example = ein.EinsteinRelation(xdatcar='../xdatcar/xdatcar.2000K/XDATCAR_01',
                               outcar='../xdatcar/xdatcar.2000K/OUTCAR',
                               skip=0, # optional; exclude first 500 steps.
                               segment=1,# optional; divide total MD step into two segment.
                               verbose=True
                              ) 

D = example.diffcoeff.squeeze()   # [[Dx(Hf), Dy(Hf), Dz(Hf)],
                                  #  [Dx(O),  Dy(O),  Dz(O)]]
D_Hf, D_O = np.average(D, axis=1)
print("D_Hf = %.3e m2/s"%D_Hf) # Diffusivity of Hf
print("D_O  = %.3e m2/s"%D_O) # Diffusivity of O
```
If **verbose=True**, the information of the MD simulation is printed.
```
# 'verbose=True' prints MD information 
reading ../xdatcar/xdatcar.2000K/OUTCAR...
	potim = 2.0
	nblock = 1

reading ../xdatcar/xdatcar.2000K/XDATCAR_01...
	atom type = ['Hf', 'O']
	number of type = 2
	number of atom = [34 42]
	total number of atom = 76
	shape of position = (15000, 76, 3) #(nsw, number of atom, xyz)

	--- MD information ---
	total time = 30.0 ps
	number of segments = 1
	segment length = 15000
	shape of msd = (1, 2, 15000, 3) #(segment, number of type, nsw - skip, xyz)
```
In addition, a MSD graph of each atom is displayed.
<p align="center">
<img src="https://github.com/user-attachments/assets/d567e1ba-331e-48a4-b36e-f4988d133b2c" width="400" height="300"/> 
</p>

---
### Electrical conductivity

${\textsf{\color{gray} will be update}}$


---
### Schottky profile

${\textsf{\color{gray} will be update}}$

---
## Module details
---
## amorphous
### amorphous.genAmorphous (*class*)
```
genAmorphous(density, chem_formula, outfile, clear_dummy)
```
* **Description** : Generating an amorphous with a density of '*density*' and a composition of '*chem_formula*'. POSCAR file named of '*outfile*' is generated.
* **Arguments**
  * density: (*float*) density of amorphous in unit of g/cm<SUP>3</SUP>
  * chem_formula: (*str*) chemical formula of amorphous. **Note:** the number 1 cannot be omitted. (ex. Hf32O42Ag1)
  * outfile: (*str; opt*) name of output file. (Default: 'POSCAR')
  * clear_dummy: (*bool; opt*) whether to remove dummy files. (Default: True)


### amorphous.genInput (*class*)
```
genInput(potcar, nsw, potim, temp, charge, ncore)
```
* **Description** : Generating VASP input files matching POSCAR file
* **Arguments**
  * potcar: (*str; opt*) kind of potcar. 'lda' or 'pbe'. (Default: 'pbe')
  * nsw: (*int; opt*) interation number of ionic step. (Default: 10000)
  * potim: (*int; opt*) time step in MD in unit of fs. (Default: 2)
  * temp: (*float; opt*) temperature in unit of K. (Default: 4000)
  * charge: (*float; opt*) charge state of the cell. (Default: 0)
  * ncore: (*int; opt*) number of core for parallel computing. (Default: 4)


### amorphous.xdat2pos (*class*)
```
xdat2pos(xdatcar, step)	# to generate one POSCAR
```
* **Description** : Generating POSCAR file from XDATCAR file. POSCAR_step{step} will be generated in the present directory.
* **Arguments**
  * xdatcar: (*str*) XDATCAR file
  * step: (*int*) step number in MD simulation

### amorphous.xdat2pos (*class*)
```
xdat2pos(xdatcar, start, end, step)	# to generate one POSCAR
```
* **Description** : Generating POSCAR files from XDATCAR file. POSCAR files will be generated in **./ensembles**. (ex. POSCAR_01, POSCAR_02, ...)
* **Arguments**
  * xdatcar: (*str*) XDATCAR file
  * start, end, step: (*int*) start, end, step in **numpy.arange(start, end+1, step)**
 
## einstein
## einstein.file_manager
### einstein.file_manager.getMDset (*class*)
```
getMDset(path_poscar, temp, label, potcar, nsw, potim, charge, ncore)
```
* **Description** : Generating MD sets using POSCAR files in **path_poscar** directory.
  
**Note:** The format of name of POSCAR files should be **'POSCAR_{label}'**, label can be an arbitrary string.
```
{path_poscar}\	# example of POSCAR files
    POSCAR_01
    POSCAR_02
    POSCAR_03 
    POSCAR_04
    POSCAR_05
```
* **Arguments**
  * path_poscar: (*str*) path of directory containing POSCAR files.
  * temp: (*list*) temperature in unit of K. **Note:** only integer temperature is available
  * label: (*list; opt*)  labels of POSCARs. (Default: *automately generated by reading files in* ***path_poscar***)
  * potcar: (*str; opt*) kind of potcar. 'lda' or 'pbe'. (Default: 'pbe')
  * nsw: (*int; opt*) interation number of ionic step. (Default: 10000)
  * potim: (*int; opt*) time step in MD in unit of fs. (Default: 2)
  * charge: (*float; opt*) charge state of the cell. (Default: 0)
  * ncore: (*int; opt*) number of core for parallel computing. (Default: 4)


### einstein.file_manager.getMDresult (*class*)
```
getMDresult(temp, label, outdir)
```
* **Description** : Collecting MD results into **outdir** directory.
* **Arguments**
  * temp: (*list; opt*) temperature in unit of K. (Default: *automately generated by reading files in present directory*)
  * label: (*list; opt*) labels of POSCARs. (Default: *automately generated by reading files in present directory*)
  * outdir: (*str; opt*) directory where the results will be collected. (Default: './xdatcar')

## einstein.einstein
### einstein.einstein.EinsteinRelation (*class*)
```
EinsteinRelation(xdatcar, outcar, segments, skip, skip2, verbose, getD)
```
* **Description** : Calculate MSD and diffusion coefficient of one set of XDATCAR and OUTCAR.
* **Arguments**
  * xdatcar: (*str*) path of XDATCAR file.
  * outcar: (*str*) path of OUTCAR file.
  * segment: (*int; opt*) number of segment. For example, if '**segment=2**', the total MD steps are divided into two MD sets, effectively doubling the overall ensemble size. (Default: 1)
  * skip: (*int; opt*) number of initial steps to be excluded in analysis. **Note:** Since the initial steps are unstable and contain undesirable factors like ballistic transport, excluding some initial steps helps increase the reliability of calculation. (Default: 0)
  * skip2: (*int; opt*) skip2 specifies number of initail steps to be excluded in each segment. **Note:** In early steps in each segment, MSD might not have linear relation with time since not enough scattering is included. Hence, excluding some early steps can improve the linearity of MSD. (Default: 0)
  * verbose: (*bool; opt*) if True, MD information and MSD of each atom are displayed. (Default: True)
  * getD: (*bool; opt*) if True, diffusion coefficeint is calculated. (Default: True)
* **Attributes**
  * msd: (*numpy array*) MSD of each atom. **Dimension**: (segments, atom species, steps, xyz)
  * diffcoeff: (*numpy array*) diffusion coefficient of each atom. **Dimension**: (segments, atom species, xyz)
  * plot_msd(): (*method*) plot MSD graph.

 
### einstein.einstein.EnsembleEinstein (*class*)
```
EnsembleEinstein(symbol, prefix, labels, segments, skip, start, end)
```
* **Description** : Integrate multiple individual MD simulations, and calculate integrated MSDs and mass transprot parameters. **NOTE:** MD simulations should be obtained with the same conditions. (ex. temperature, number of MD steps, potim)
* **Arguments**
  * symbol: (*str*) target atom species. (ex. 'O')
  * prefix: (*str*) path where the XDATACR files exist. (ex. 'xdatcar/xdatcar.2000K/') **NOTE:** XDATCAR files should be in format of **XDATCAR_{label}**
  * labels: (*list*) labels of XDATCAR files.
  * segments: (*int*) number of segment. For example, if '**segment=2**', the total MD steps are divided into two MD sets, effectively doubling the overall ensemble size.
  * skip: (*int*) number of initial steps to be excluded in analysis. **Note:** Since the initial steps are unstable and contain undesirable factors like ballistic transport, excluding some initial steps helps increase the reliability of calculation.
  * start: (*int; opt*) initial step to be used in linear fitting.
  * end: (*int; opt*) final step to be used in linear fitting.
* **Attributes**
  * msd_xyz: (*numpy array*) MSD of each atom (with x, y, z component). **Dimension**: (steps, xyz)
  * msd: (*numpy array*) MSD of each atom. **Dimension**: (steps, xyz)
  * diffcoeff: (*float*) diffusion coefficient of each atom.
  * diffcoeff_x: (*float*) x component of diffusion coefficient of each atom.
  * diffcoeff_y: (*float*) y component of diffusion coefficient of each atom.
  * diffcoeff_z: (*float*) z component of diffusion coefficient of each atom.
  * plotEnsembleMSD: (*method*) plot MSD graph. **NOTE:** After run the method, *plt.show()* command is required.
  * saveMSD: (*method*) save MSD data in *msd_{temp}.txt*.
