#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process substrate and film surface POSCAR files.")
parser.add_argument("-s", "--substrate", type=str, required=True, help="Path of the surface POSCAR of the substrate.")
parser.add_argument("-f", "--film", type=str, required=True, help="Path of the surface POSCAR of the film surface.")

args = parser.parse_args()

substrate = args.substrate
film = args.film

if not os.path.isfile(substrate):
    print(f"Error: The substrate surface POSCAR file {substrate} does not exist.")
    exit(1)

if not os.path.isfile(film):
    print(f"Error: The film surface POSCAR file {film} does not exist.")
    exit(1)

print(f"Processing substrate surface POSCAR file: {substrate}")
print(f"Processing film surface POSCAR file: {film}")

interface = GenInterface(substrate, film)
#interface = GenInterface("Project/Surface/RuO2_slab-110.vasp","Project/Surface/TiO2_slab-110.vasp")
interface.auto_lattice_matching()
# # interface.manual_lattice_matching([2,2],[2,2])
#interface.interface_maker(spacing=0.75,vacuum=0)
interface.film_xy_shift_direct([0.5,0])
interface.interface_maker(spacing=0.75,vacuum=0)
# # interface.shift_xy()