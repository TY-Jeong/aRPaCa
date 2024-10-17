#!/bin/sh
#PBS -o sys_mesg.log -N SBH
#PBS -j oe
#PBS -l nodes=1:ppn=32
#PBS -q Null

NUMBER=`wc -l < $PBS_NODEFILE`
cd $PBS_O_WORKDIR

mpirun -np $NUMBER /home/ysj/program/qe-6.6/bin/pw.x < qe.in > stdout.dat
