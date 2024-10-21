#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process substrate and film surface POSCAR files.")
parser.add_argument("substrate_surface", type=str, help="Path of the surface POSCAR of the substrate.")
parser.add_argument("film_surface", type=str, help="Path of the surface POSCAR of the film surface.")

args = parser.parse_args()

substrate_surface = args.substrate_surface
film_surface = args.film_surface

if not os.path.isfile(substrate_surface):
    print(f"Error: The substrate surface POSCAR file {substrate_surface} does not exist.")
    exit(1)

if not os.path.isfile(film_surface):
    print(f"Error: The film surface POSCAR file {film_surface} does not exist.")
    exit(1)

print(f"Processing substrate surface POSCAR file at: {substrate_surface}")
print(f"Processing film surface POSCAR file at: {film_surface}")

interface = GenInterface(substrate_surface, film_surface)
#interface = GenInterface("Project/Surface/RuO2_slab-110.vasp","Project/Surface/TiO2_slab-110.vasp")
interface.auto_lattice_matching()
# # interface.manual_lattice_matching([2,2],[2,2])
interface.interface_maker(spacing=0.75,vacuum=0)
interface.film_xy_shift_direct([0.5,0])
# #interface.interface_maker(spacing=0.75,vacuum=0)
# # interface.shift_xy()