from arpaca.generate_input import *


interface = GenInterface("Project/Surface/RuO2_slab-110.vasp","Project/Surface/TiO2_slab-110.vasp")
interface.auto_lattice_matching()
# # interface.manual_lattice_matching([2,2],[2,2])
interface.interface_maker(spacing=0.75,vacuum=0)
interface.film_xy_shift_direct([0.5,0])
# #interface.interface_maker(spacing=0.75,vacuum=0)
# # interface.shift_xy()