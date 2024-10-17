from arpaca.file_manager import *

RuO2_calc = BulkSet('Project/Bulk/RuO2/POSCAR_mp-825')
RuO2_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='ykh', scheduler='pbs')
TiO2_calc = BulkSet('Project/Bulk/TiO2/POSCAR_mp-2657',ncore=4)
TiO2_calc.semi_setup()
TiO2_calc.runfile_setup(runfile_name='run.sh', nodes='2', processors='24', queue_name='ykh', scheduler='pbs')