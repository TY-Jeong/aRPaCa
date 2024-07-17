import sys
sys.path.append('/home/taeyoung/github/aRPaCa/arpaca/einstein')
from einstein import *

getDiffusivity(symbol='O',
               label=[format(i+1, '02') for i in range(10)],
               temp=np.arange(1200, 2000+1, 100),
               segment=1,
               skip=500,
               start=1000,
               end=14500,
               xdatcar='./xdatcar',
               xyz=True)
