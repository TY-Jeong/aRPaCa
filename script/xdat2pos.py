# Argument : {XDATCAR} {step_init} {step_final} {step_interval}
import sys
from arpaca.utils import *

xdatcar = sys.argv[1]
start, end, step = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
xdat2pos(xdatcar, start, end, step)
