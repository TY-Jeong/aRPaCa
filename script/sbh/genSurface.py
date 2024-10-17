from arpaca.generate_input import *


RuO2_surface=GenSurface((1,1,0),"Project/Bulk/RuO2/Relax/CONTCAR", layer_num = 9, vacuum=15)
RuO2_surface.slice_slab_cartesian(20.1, 35.44)

TiO2_surface=GenSurface((1,1,0),'Project/Bulk/TiO2/Relax/CONTCAR',layer_num = 9, vacuum=15)
TiO2_surface.slice_slab_cartesian(20.2, 42.6)
TiO2_surface.xy_shift_direct([0.5,0.5])
TiO2_surface.cbs_surface_maker(layer_num = 2)