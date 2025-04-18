#!/bin/sh
#PBS -o sys_mesg.log -N SBH
#PBS -j oe
#PBS -l nodes=4:ppn=24
#PBS -q ykh

NUMBER=`wc -l < $PBS_NODEFILE`
cd $PBS_O_WORKDIR

mpirun -np $NUMBER /home/ysj/vasp.5.4.4/bin/vasp_std > stdout.dat
