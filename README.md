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


Here, the usages of the scipt files are summarized:


|Scipt|Explanation|Note|
|-----|-----------|----|
|**genAmorphous.py**|Generate amorphous structure| |
|**einstein.py**|Calculate diffusion coefficient based on Einstein relation| |
|**trajectory.py**|Determine trajectory of vacancy <br> Calculate effective diffusion parameters (except for frequency)|Crystalline only|
|**frequency.py**|Calculate jump attempt frequency |**trajectory.py** should be executed first|


### Amorpohus generation


