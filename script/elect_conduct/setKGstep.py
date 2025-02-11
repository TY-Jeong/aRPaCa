import arpaca.elect_conduct.calculator as aec
import numpy as np

step = int(input('your previous step done [0=relax/1=wave/2=nabla/3=cond]: '))
# 0 = after relax
# 1 = after wave
# 2 = after nabla
# 3 = after cond

if step<4:
    tweak = input('do you want to set list of temperatures or snapshots? [Y/n] ')
    if tweak == 'Y':
        trange = input('list of temperatures: ') ## e.g. ['300K', '400K']
        srange = input('list of snapshots: ') ## e.g. [10000, 10500, 11000]
        kg_step = aec.KGCalcStep(step, temp_list=trange, snap_list=srange)
    elif tweak == 'n':
        kg_step = aec.KGCalcStep(step)
    else:
        print('Invalid input')
else:
    print('Invalid input')


