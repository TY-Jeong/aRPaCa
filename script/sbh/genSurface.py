#!/usr/bin/env python
import argparse
from arpaca.utils import *
import time

parser = argparse.ArgumentParser(description="Script to make surface POSCAR file.")
parser.add_argument("contcar_path", type=str, help="Path to the bulk CONTCAR file.")
args = parser.parse_args()
contcar_path = args.contcar_path

if not os.path.isfile(contcar_path):
    print(f"Error: The file {contcar_path} does not exist.")
    exit(1)


# RuO2_surface=GenSurface(layer_num = 9, vacuum=15)
# RuO2_surface.view_structure()

#RuO2_surface=GenSurface("Project/Bulk/RuO2/Relax/CONTCAR", (1,1,0), layer_num = 9, vacuum=15)
# RuO2_surface=GenSurface(structure_file="Project/Bulk/RuO2/Relax/CONTCAR", layer_num = 9, vacuum=15)
# RuO2_surface.view_structure()
# RuO2_surface.slice_slab_cartesian()
# #RuO2_surface.slice_slab_cartesian(20.1, 35.44)
# RuO2_surface.view_structure()

# #TiO2_surface=GenSurface('Project/Bulk/TiO2/Relax/CONTCAR', (1,1,0), layer_num = 9, vacuum=15)
surface=GenSurface(contcar_path, layer_num = 9, vacuum=15)
surface.slice_slab_cartesian(20.2, 42.6)
#TiO2_surface.xy_shift_direct(0.5,0.5)
surface.view_structure()

time.sleep(4)
surface.cbs_surface_maker(layer_num = 2)
surface.view_cbs_structure()