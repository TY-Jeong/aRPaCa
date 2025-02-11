# aRPaCa
---
**aRPaCa** (***a****b-initio* **R**RAM **Pa**rameter **Ca**lculator) is a python package for calculating parameters used in RRAM simulation, which contains modules for pre- or post-processing of DFT calculation using VASP. (Note: aRPaCa is written baed on VASP 5.4.4.)

## Features

* Tools for **structure generation**.
    * Amorphous generation
    * Interface generation
*  **Diffusion coefficient** calculation based on *ab initio* molecular dynamics (AIMD)
    * Effecitve diffusion parameters
    * Diffusion coefficient using Einstein relation
* **Schottky profile** of metal/oxide interface
* **Electric conductivity** of oxide with arbitrary composition

## Contents

* Getting started
* User instruction
  * Structure generation
    * Amorphous generation
    * Interface generation
  * Parameter calculation
    * Mass transport paramters
    * Electrical conductivity
    * Schottky profile

## Getting started

For amorphous generation, aRPaCa uses [Packmol](https://m3g.github.io/packmol/download.shtml), the open-source code providing a plausible random structure. Please make sure Packmol is installed on your system, and add the path of the packmol excutable into `aRPaCa/data/path.dat`. Below is an example of `path.dat`
```ruby
POT_PBE=/home/taeyoung/POT/POT_PBE
POT_LDA=/home/taeyoung/POT/POT_LDA
packmol=/home/taeyoung/Downloads/packmol-20.14.3/packmol
```


## User instruction

To facilitate implementation of **aRPaCa**, scipts files are provided in `aRPaCa/script` directory. The user can also implement **aRPaCa** in Python interpreter with `import arpaca`.


The list of sripts are summarized below:


|Scipt|Explanation|Note|
|-----|-----------|----|
|**genAmorphous.py**|Generate amorphous structure| |
|**einstein.py**|Calculate diffusion coefficient based on Einstein relation| |
|**trajectory.py**|Determine trajectory of vacancy <br> Calculate effective diffusion parameters (except for frequency)|*not opened*<br>(*papar in preparation*)|
|**frequency.py**|Calculate jump attempt frequency |*not opened*<br>(*papar in preparation*)|
|**genBulk.py**|Generate bulk crystal structure| |
|**genSurface.py**|Generate surface structure based on the bulk crystal structure| |
|**genInterface.py**|Generate interface structure based on the surface structure| |
|**getVASPResult.py**|Parse VASP results of bulk and interface calculations| |
|**getQEResult.py**|Parse QE result of complex band structure calculation| |
|**sbhCalc.py**|Calculate the Schottky barrier profile (Schottky barrier height) using a self-consistent procedure| |


### **Amorpohus generation**
---
Simple example:

```ruby
python genAmorphous.py -c Hf32O64 -d 10 # -c {chemical_formula} -d {density}
```
Here, the term following `-c` represents the chemical formula of an amorphous which the user want to obtain. It should be noted that the number of 1 cannot be ommited (*i.e.*, Hf32O64Ag1 rather than Hf34O64Ag). the second term following `-d` the density of the amorphous in unit of g/cm<SUP>3</SUP>. By executing the script, `POSCAR_{chemical_formula}` file will be generated. The user can also use the `-h` option for detailed description. 


### **Interface generation**
---
Simple example:

```ruby
python genBulk.py -c TiO2 # -c {chemical formula}
```
Here, the term following `-c` represents the chemical formula of an amorphous which the user want to obtain. By executing the  script, `POSCAR_{mp_id}` file will be generated. 
```ruby
python genSurface.py -p CONTCAR -m 110 -t 20 -v 10 
# -p {path to the relaxed bulk crystal structure file} 
# -m {miller index as three integers} 
# -t {minimum thickness of slab in Å}
# -v {vacuum thickness in Å}
```
Here, the term following `-p` represents the path to the relaxed bulk crystal structure file that the user wants to use. The second term following `-m` represents the Miller index of the surface plane. The third term following `-t` represents the minimum thickness of the slab in Å. The fourth term following `-v` represents the vacuum thickness in Å. By executing the script, `{chemical_formula}_slab-{miller_index}.vasp` file will be generated.
```ruby
python genInterface.py -s POSCAR_substrate -f POSCAR_film 
# -s {path to the surface structure file as a substrate}
# -f {path to the surface structure file as a film}
```
Here, the term following `-s` represents the path to the surface structure file that the user wants to use as a substrate. The second term following `-f` represents the path to the surface structure file that the user wants to use as a film. By executing the script, `interface.vasp` file will be generated.

The user can also use the `-h` option for detailed descriptions. If no terms are provided, a GUI will be launched to allow the user to input the necessary parameters.

### **Mass transport parameter**
---
### Preparations
---
***Note**: this codes are in preparation, hence this portion will be opend after publication*


For the mass transprot parameter calculation, the user need to conduct AIMD simulation (NVT ensemble) in several temperatures in advance. Then, the results should be arranged as follows:

``` ruby
# example of ensemble directory
traj\
    traj.1500K\
        OUTCAR
        XDATCAR_01, FORCE_01
           ⋮
        XDATCAR_05, FORCE_05
   traj.1600K\ # contents are omitted
   traj.1700K\ # contents are omitted
```
Here, five AIMD simulations, labeled as 01, 02, ..., 05, were performed at each temperature. The required number of simulation is system-dependent, but should be large enough to ensure the reliability. FORCE files contains force vectors at each iteration and can be obtained using the `extractForce.py` script.

### Einstein relation
---
Simple example:
```ruby
python einstein.py O 50 # {symbol of moving atom}, {x-range of msd plot}
```
Please use `-h` options, for other available options. If you used this script, please also cite [here](https://doi.org/10.1016/j.cpc.2022.108599).


(*will be updated after publication*)

### Effective diffusion parameter
---
simple example:
```ruby
python trajectory.py O 0.1 # {symbol of moving atom}, {time interval}
python frequency.py # to obtain effective frequency and effective coordination number
```
Please use `-h` options, for other available options.

(*will be updated after publication*)


### **Electronic Conductivity**
---
|Scipt|Explanation|Note|
|-----|-----------|----|
|**setKGcalc.py**|Create calculation set for Kubo-Greenwood conductivity calculation| |
|**setKGstep.py**|Do works between each step of Kubo-Greenwood conductivity calculation| |
|**condOutputAnalyze.py**|Analyze the conductivity calculation results| |

simple example:
After your MD simulation, there should be the `ensembles` directory.
`python setKGcalc.py` to prepare for KG calculation.

Do the following calculations under each temperature/snapshot directory.

0. (VASP) structure relax
1. (under 1_wave directory) (VASP) get WAVECAR
2. (under 2_nabla directory) (GreeKuP-recompiled VASP) get optic property
3. (under 3_cond directory) (GreeKuP) get electronic conductivity
after each step, `python setKGstep.py` to prepare for next calcuation step.

After the electronic conductity is calculated, `python condOutputAnalyze.py` to print statistics.
