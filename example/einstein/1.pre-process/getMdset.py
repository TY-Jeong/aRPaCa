import sys
import numpy as np
from arpaca.einstein.file_manager import *
#sys.path.append('/home/taeyoung/github/aRPaCa/arpaca/einstein')
#from file_manager import *

getMDset(path_poscar='ensembles',
         temp=np.arange(1200, 1900+1, 100))
