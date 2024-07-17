import os
import numpy as np
import subprocess

command = "grep LOOP+ OUTCAR | awk '{print $7}' > time.dat"
os.system(command)

dummy = np.loadtxt('time.dat')
time_step = np.average(dummy)
os.remove('time.dat')

step = None
with open('stdout.dat', 'r') as f:
        for line in f:
                if 'T=' in line:
                        step = line.split()[0]
NSW = None
with open('INCAR', 'r') as f:
        for line in f:
                if 'NSW' in line:
                        NSW = line.split('=')[1].split()[0]
                        break
time_left = time_step * (int(NSW) - int(step))
H = int(time_left/3600)
M = int((time_left - 3600*H)/60)

print(f"process : {step}/{NSW}")
print(f"speed   : {time_step} s/step")
print(f"time remaining : {H} h {M} m")

