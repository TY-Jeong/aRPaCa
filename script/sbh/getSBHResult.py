from arpaca.file_manager import *

result = GetSBHResult('Project')
result.get_vasp_result()



pseudopotentials = {
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"
}
result.qe_setup(pseudopotentials, scheduler='pbs')


result.get_qe_result()
