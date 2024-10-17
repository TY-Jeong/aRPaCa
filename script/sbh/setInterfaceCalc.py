from arpaca.file_manager import *

interface_calc = InterfaceSet("Project/Interface/interface.vasp",ncore=4)
interface_calc.runfile_setup(runfile_name='run.sh', nodes='4', processors='24', queue_name='ykh', scheduler='pbs')