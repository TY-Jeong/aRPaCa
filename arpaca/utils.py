import os
import sys
import shutil
import copy
import numpy as np
import warnings
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
import tkinter as tk
from tkinter import font, ttk, filedialog, messagebox
from ase.io import read, write
from ase.build import surface, make_supercell
from ase import Atoms
from ase.visualize import view
from ase.units import Bohr, Angstrom
from pymatgen.core import Element, Composition
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.pwscf import PWInput
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import interp1d
try:
    from mp_api.client import MPRester
except:
    print("mp_api module is not found: interface generation is not avail")
    
warnings.filterwarnings("ignore", module='pymatgen')


'''
VASP Input generator
    genInput
    genAmorphous
    
'''
class genInput:
    def __init__(self,
                 potcar = 'pbe',
                 nsw = 10000,
                 potim = 2,
                 temp = 4000,
                 charge = 0,
                 ncore = 4):
        """
        Arg 1: (str) kind of potcar: lda or pbe (default:pbe)\n
        Arg 2: (int) iteration number. (default:10000)\n
        Arg 3: (int) time step in fs. (default: 2)\n
        Arg 4: (float) temperature in K (default: 4000)\n
        Arg 5: (float) charge state. (ex. 2 for +2) (default: 0)\n
        Arg 6: (int) number of core. (default: 4)
        """
        self.potcar = potcar
        self.nsw = nsw
        self.potim = potim
        self.temp = temp
        self.charge = charge
        self.ncore = ncore

        self.prefix_data = '../data/'

        # write KPOINTS
        self.write_kpoints()
        
        # write POTCAR
        self.pot_recommend = {}
        self.read_pot_recommended()

        self.prefix_pot = None
        self.read_pot_path()

        self.atom_name = None
        self.atom_num = None
        self.zval =[]
        self.enmax = []
        self.write_potcar()

        # write INCAR
        self.atom_num = np.array(self.atom_num)
        self.zval = np.array(self.zval)
        self.nelect = np.sum(self.atom_num*self.zval)-self.charge

        self.write_incar()


    def read_pot_recommended(self):
        recommended_potcar_path = os.path.join(
            os.path.dirname(__file__), self.prefix_data+'recommended_potcar.dat')
        if not os.path.isfile(recommended_potcar_path):
            print('recommended_potcar.dat is not found.\n \
                  please visit https://github.com/TY-Jeong/arpaca')
            sys.exit(0)     

        with open(recommended_potcar_path, 'r') as file:
            lines = [line.strip() for line in file]

        for line in lines:
            name, recommend = line.split('\t')
            self.pot_recommend[name] = recommend
    
    
    def read_pot_path(self):
        path_dat_path = os.path.join(os.path.dirname(__file__), self.prefix_data+'path.dat')
        if not os.path.isfile(path_dat_path):
            print('path.dat is not found.\n')
            sys.exit(0)

        with open(path_dat_path, 'r') as file:
            lines = [line.strip() for line in file]

        self.prefix_pot = lines[0].split('=')[1] if self.potcar == 'pbe' else lines[1].split('=')[1]
        

    def write_kpoints(self):
        with open('./KPOINTS','w') as f:
            f.write("kpoints\n")
            f.write("0\n")
            f.write("G\n")
            f.write("1 1 1\n")
            f.write("0 0 0")
            
            
    def write_potcar(self):
        if not os.path.isfile('POSCAR'):
            print('no POSCAR file exists.')
            sys.exit(0)
        
        with open('POSCAR', 'r') as file:
            lines = [line.strip() for line in file]
        self.atom_name = lines[5].split()
        self.atom_num = np.array(lines[6].split(), dtype=int)

        with open('./POTCAR','w') as f:
            for name in self.atom_name:
                pot = os.path.join(self.prefix_pot, self.pot_recommend[name], 'POTCAR')
                with open(pot, 'r') as pot_file:
                    pot_lines = [line for line in pot_file]
                for line in pot_lines:
                    f.write(line)
                    if 'ZVAL' in line:
                        self.zval += [float(line.split()[5])]
                    if 'ENMAX' in line:
                        self.enmax += [float(line.split()[2].split(';')[0])]


    def write_incar(self):
        system = ''
        for name, num in zip(self.atom_name, self.atom_num):
            system += name+str(num)

        with open("./INCAR",'w') as f:
            f.write(f"SYSTEM = {system}\n")
            f.write(f"NCORE = {self.ncore}\n")
            f.write('\n')
            f.write('# electronic degrees\n')
            f.write(f"ENCUT = {np.max(self.enmax)}\n")
            f.write("LREAL = Auto\n")
            f.write("PREC  = Normal\n")
            f.write("EDIFF = 1E-5\n")
            f.write("ISMEAR = 0\n")
            f.write("SIGMA = 0.01\n")
            f.write("ALGO = Fast\n")
            f.write("MAXMIX = 40\n")
            f.write("ISYM = 0\n")
            f.write("NELMIN = 4\n")
            f.write('\n')
            f.write('# molecular dynamics\n')
            f.write('IBRION = 0\n')
            f.write(f"NSW = {self.nsw}\n")
            f.write(f"POTIM = {self.potim}\n")
            f.write("NWRITE = 0\n")
            f.write("NBLOCK = 1\n")
            f.write("LCHARG = .FALSE.\n")
            f.write("LWAVE = .FALSE.\n")
            f.write(f"TEBEG = {self.temp}\n")
            f.write(f"TEEND = {self.temp}\n")
            f.write('\n')
            f.write('# canonical (Nose-Hoover) thermostat\n')
            f.write("MDALGO = 2\n")
            f.write("SMASS =  0\n")
            f.write("ISIF = 2\n")
            f.write('\n')
            f.write('# charge state\n')
            f.write("NELECT = %.3f"%self.nelect)



class genAmorphous:
    def __init__(self, 
                 density, 
                 chem_formula,
                 outfile = 'POSCAR',
                 clear_dummy = True):
        """
        Arg 1: (float) density in g/cm3. (ex. 10.00)\n
        Arg 2: (str) chemical forumula. (ex. Hf32O64Ag1)\n
        Arg 3: (str) name of POSCAR. (default:POSCAR)\n
        Arg 4: (bool) remove dummy files. (default:True)
        """
        self.density = density
        self.chem_formula = chem_formula

        self.prefix_data = '../data/'

        self.atomMass = {}
        self.read_atomic_mass()

        self.atom_name = []
        self.atom_num = []
        self.read_chem_formula()

        # check atom name
        for name in self.atom_name:
            if not(name in self.atomMass.keys()):
                print(f"{name} is unknown atom.")
                sys.exit(0)

        self.alat = None
        self.alat_pbc = None
        self.get_dimension_cubic()

        self.write_pdb()
        self.write_packmol_input()

        # run packmol
        try:
            path_dat_path = os.path.join(os.path.dirname(__file__),
                                         self.prefix_data,'path.dat')
            with open(path_dat_path, 'r') as file:
                lines = [line.strip() for line in file]
            packmol = lines[2].split('=')[1]
            command = f"{packmol} < packmol.inp > packmol.out"
            os.system(command)
            #subprocess.run(command, shell=True, check=True)
        except Exception as e:
             print(f"Error running packmol: {e}")
             sys.exit(1)

        # write POSCAR
        self.covert_pdb2poscar(outfile)

        if clear_dummy:
            self.clear_dummy()


    def read_atomic_mass(self):
        atomic_mass_table_path = os.path.join(os.path.dirname(__file__),
                                              self.prefix_data,'atomic_mass_table.dat')
        if not os.path.isfile(atomic_mass_table_path):
            print('atomic_mass_table.dat is not found.\n \
                  please visit https://github.com/TY-Jeong/arpaca')
            sys.exit(0)

        with open(atomic_mass_table_path,'r') as file:
            lines = [line.strip() for line in file]

        for line in lines:
            name, mass = line.split('\t')
            self.atomMass[name] = float(mass)


    def read_chem_formula(self):
        check, word = 'alpha', ''
        for letter in self.chem_formula:
            if letter.isalpha():
                if check == 'alpha':
                    word += letter
                elif check == 'num':
                    self.atom_num += [int(word)]
                    check = 'alpha'
                    word = letter
            else:
                if check == 'alpha':
                    self.atom_name += [word]
                    check = 'num'
                    word = letter
                elif check == 'num':
                    word += letter
        self.atom_num += [int(word)]
        self.atom_num = np.array(self.atom_num)


    def get_dimension_cubic(self):
        u  =  1.660539e-24 # g
        mass = [self.atomMass[name]*u for name in self.atom_name]
        mass = np.array(mass)
        m = np.sum(mass * self.atom_num)
        self.alat = (m/self.density)**(1/3)*1e8  # Å
        self.alat_pbc = self.alat*0.98 - 2


    def write_pdb(self):
        for name in self.atom_name:
            with open(f"./{name}.pdb",'w') as f:
                f.write(f"COMPND    {name}_atom\n")
                f.write(f"COMPND   1Created by aRPaCa\n")
                f.write("HETATM    1 %-2s           1       5.000   5.000   5.000\n"%(name))
                f.write("END")


    def write_packmol_input(self):
        with open("./packmol.inp",'w') as f:
            f.write("tolerance 2.0\n")
            f.write("filetype pdb\n")
            f.write("output packmol_output.pdb\n\n")

            for name, num in zip(self.atom_name,self.atom_num):
                f.write(f"structure {name}.pdb\n")
                f.write(f"  number {num}\n")
                f.write("  inside cube 0. 0. 0. %.6f \n"%self.alat_pbc)
                f.write("end structure\n\n")


    def covert_pdb2poscar(self, 
                         outfile):
        with open ('packmol_output.pdb','r') as file:
            lines = [line.strip() for line in file]
        num_tot = np.sum(self.atom_num)
        shift = (self.alat-self.alat_pbc)/2
        with open(outfile, 'w') as f:
            f.write("Generated by aRPaCa\n")
            f.write("1.0\n")
            f.write("   %.6f      0.000000      0.000000\n"%self.alat)
            f.write("   0.000000      %.6f      0.000000\n"%self.alat)
            f.write("   0.000000      0.000000      %.6f\n"%self.alat)
            for name in self.atom_name:
                f.write("   "+name)
            f.write("\n")
            for num in self.atom_num:
                f.write("   "+str(num))
            f.write("\n")
            f.write("Cartesian\n")
            for line in lines[5:5+num_tot]:
                coord = np.array(line.split()[-3:], dtype=float)
                coord += np.array([shift, shift, shift])
                f.write("    %.6f    %.6f   %.6f\n"%(coord[0], coord[1], coord[2]))
    
    
    def clear_dummy(self):
        for name in self.atom_name:
            os.remove(f"{name}.pdb")
        os.remove('packmol.inp')
        os.remove('packmol.out')
        os.remove('packmol_output.pdb')
            
            
            
'''
Utilities for mass_transport module
    getMDset
    getMDresult
    xdat2pos
    extract_force
'''         
class getMDset:
    def __init__(self,
                 path_poscar,
                 temp,
                 label=None,
                 potcar='pbe',
                 nsw=10000,
                 potim=2,
                 charge=0,
                 ncore=4):
        """
        Arg 1: (str) path_poscar; path of directory containing poscar files
        Arg 2: (list) temp; temperature in K 
        Arg 3: (list; opt) label; labels of poscar. poscar format should be POSCAR_{label}
        """
        self.path_poscar = path_poscar
        self.label = label
        self.temp = temp
        self.potcar = potcar
        self.nsw = nsw
        self.potim = potim
        self.charge = charge
        self.ncore = ncore

        if self.label is None:
            self.label = []
            path_now = os.getcwd()
            for name in os.listdir(self.path_poscar):
                if len(name.split('_')) == 2:
                    poscar, label = name.split('_')
                    if poscar=='POSCAR':
                        self.label += [label]
        
        self.foldername=[]
        self.make_input_set()


    def make_input_set(self):
        path_now = os.getcwd()
        for t in self.temp:
            outer_folder = f"{t}K"
            for l in self.label:
                # make folder
                path_dir = os.path.join(outer_folder, f"{l}")
                self.foldername += [path_dir]
                os.makedirs(path_dir, exist_ok=True)
                
                # copy poscar
                from_pos_path = os.path.join(self.path_poscar, f"POSCAR_{l}")
                to_pos_path = os.path.join(path_dir, 'POSCAR')
                shutil.copyfile(from_pos_path, to_pos_path)
                
                # make input files
                os.chdir(path_dir)
                _ = genInput(potcar=self.potcar,
                             nsw=self.nsw,
                             potim=self.potim,
                             temp=t,
                             charge=self.charge,
                             ncore=self.ncore)
                os.chdir(path_now)                



class getMDresult:
    def __init__(self,
                 temp=None,
                 label=None,
                 outdir='xdatcar'):
        """
        Arg 1: (list; opt) temp;
        Arg 2: (list; opt) list;
        """
        
        self.temp = temp
        self.label = label
        self.outdir = outdir

        # folders where MD was conducted
        self.foldername=[]
        if self.temp is None and self.label is None:
            self.auto_search()
        else:
            for t in self.temp:
                for l in self.label:
                    foldername = f"{t}K/{l}"
                    self.foldername += [foldername]
        
        # copy XDATCAR files
        if os.path.isdir(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
            
        check_temp = []   
        for path in self.foldername:
            temp, label = path.split('/')
            path_save = os.path.join(self.outdir, f"xdatcar.{temp}")
            os.makedirs(path_save, exist_ok=True)
            
            # copy xdatcar
            from_xdat_path = os.path.join(path, 'XDATCAR')
            to_xdat_path = os.path.join(path_save, f'XDATCAR_{label}')
            if not os.path.isfile(from_xdat_path):
                print(f"no XDATCAR in {path}.")
            else:
                shutil.copyfile(from_xdat_path, to_xdat_path)    

            # copy outcar
            if not temp in check_temp:
                from_out_path = os.path.join(path, 'OUTCAR')
                to_out_path = os.path.join(path_save, 'OUTCAR')
                if not os.path.isfile(from_out_path):
                    print(f"no OUTCAR in {path}.")
                else:
                    check_temp += [temp]
                    shutil.copyfile(from_out_path, to_out_path)


    def auto_search(self):
        path_now = os.getcwd()
        for name_out in os.listdir(path_now):
            if os.path.isdir(name_out) and name_out[-1]=='K':
                outer_folder = name_out
                os.chdir(name_out)
                for name_in in os.listdir(os.getcwd()):
                    if os.path.isdir(name_in):
                        inner_folder = name_in
                        foldername = os.path.join(outer_folder, inner_folder)
                        self.foldername += [foldername]
                os.chdir(path_now)   
            

class xdat2pos:
    def __init__(self, 
                 xdatcar, 
                 *args):
        """
        Arg 1: (str) XDATCAR file\n
        Arg 2: (case 1) step (for one POSCAR)\n
            (case 2) start, end, interval (for multiple POSCAR)
        """
        if os.path.isfile(xdatcar):
            self.xdatcar = xdatcar
        else:
            print(f"{xdatcar} is not found.")
            sys.exit(0)

        # read xdatcar
        with open(xdatcar, 'r') as file:
            self.lines = [line for line in file]
        self.num_atoms = np.array(self.lines[6].split(), dtype=int).sum()

        if len(args) == 1:
            step = int(args[0])
            self.write_poscar(step, label=f"step{step}")
        elif len(args) == 3:
            start, end, interval = int(args[0]), int(args[1]), int(args[2]) 
            steps = np.arange(start, end+1, interval)
            if not os.path.isdir('ensembles'):
                os.mkdir('ensembles')
            for i, step in enumerate(steps):
                self.write_poscar(step, label=format(i+1,'02'), prefix='ensembles/')
        else:
            print("check your arguments.\nexpected number of argumnets is 2 or 4.")
            sys.exit(0)
    
    
    def write_poscar(self, step, label, prefix='./'):
        check = False
        file_out = f"POSCAR_{label}"
        with open(prefix+file_out, 'w') as f:
            f.write(f"step {step}\n")
            for num, line in enumerate(self.lines):
                if num > 0 and num < 6:
                    f.write(line)
                elif num == 6:
                    f.write(line)
                    f.write("Direct\n")
                else:
                    if "Direct" in line and " "+str(step)+"\n" in line:
                        for line in self.lines[num+1:num+1+self.num_atoms]:
                            f.write(line)
                        print(f"{file_out} was generated.")
                        check = True
        if not check:
            print("step is out of range.")
            os.remove(prefix+file_out)     
            
            
            
def extract_force(file_in, 
                  file_out='FORCE'):
    """
    extract force profile from vasprun.xml
    """
    # read vasprun.xml
    with open(file_in, 'r') as f:
        lines = [s.strip() for s in f]
    
    # system info
    nsw, num_atoms = None, None
    for line in lines:
        if 'NSW' in line:
            nsw = int(line.split('>')[-2].split('<')[0])
        if '<atoms>' in line:
            num_atoms = int(line.split()[1])
        if nsw and num_atoms:
            break
    
    # save forces
    step = 0
    forces = np.zeros((nsw, num_atoms, 3))
    for i, line in enumerate(lines):
        if 'forces' in line:
            force = [list(map(float, s.split()[1:4])) for s in lines[i+1:i+1+num_atoms]]
            force = np.array(force)
            forces[step] = force
            step += 1

    # write out_file
    with open(file_out, 'w') as f:
        for i, force in enumerate(forces, start=1):
            f.write(f"Iteration {i}\n")
            for fx, fy, fz in force:
                fx = str(fx).rjust(12)
                fy = str(fy).rjust(12)
                fz = str(fz).rjust(12)
                f.write(f"{fx} {fy} {fz}\n")       
            
# SBH part
class BasicTool:
    def __init__(self):
        self.prefix_data = '../data'
        self.path_dat_path = os.path.join(os.path.dirname(__file__), self.prefix_data, 'path.dat')
        if not os.path.isfile(self.path_dat_path):
            print('Error: path.dat is not found.\n')
            sys.exit(0)

    def read_path_file(self, target):
        data = {}
        with open(self.path_dat_path, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith("#"): 
                    key, value = line.split('=', 1) 
                    data[key.strip()] = value.strip()
        return data.get(target)

    def set_project_directory(self, default_dir=None):
        if default_dir == None:
            default_dir = ''
        else:
            default_dir_txt = " (default: %s)"%default_dir
        self.project_directory = input("Enter project directory name%s: "%default_dir_txt).strip()
        if self.project_directory == '':
            self.project_directory = default_dir

        self.make_directory(self.project_directory)
        self.project_directory = os.path.abspath(self.project_directory)

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

    def check_scheduler(self):
        try:
            output = subprocess.getoutput("qsub --version")
            if "not found" not in output:
                self.scheduler = 'pbs'
                return
        except Exception as e:
            print(f"Error checking PBS version: {e}")

        try:
            output = subprocess.getoutput("sbatch --version")
            if "not found" not in output:
                self.scheduler = 'slurm'
                return
        except Exception as e:
            print(f"Error checking SLURM version: {e}")

        self.scheduler = 'bash'

    def make_runfile(self, run_dir, runfile_name, nodes, processors, queue_name, scheduler):
        def submit(event=None):
            self.run_dir = run_dir_entry.get()
            self.runfile_name = runfile_entry.get()
            self.nodes = nodes_entry.get()
            self.processors = processors_entry.get()
            self.queue_name = queue_entry.get()
            self.scheduler = scheduler_entry.get()

            print("=== Runfile Saved ===")
            print(f"Run Directory: {self.run_dir}")
            print(f"Runfile Name: {self.runfile_name}")
            print(f"Nodes: {self.nodes}")
            print(f"Processors: {self.processors}")
            print(f"Queue Name: {self.queue_name}")
            print(f"Scheduler: {self.scheduler}\n")

            top.destroy()

        top = tk.Tk()
        top.title("Configure Runfile in %s"%run_dir)

        def labeled_entry(master, label, default):
            frame = ttk.Frame(master)
            frame.pack(padx=10, pady=5, fill='x')
            ttk.Label(frame, text=label, width=20).pack(side='left')
            entry = ttk.Entry(frame, width=100)
            entry.insert(0, default if default is not None else "")
            entry.pack(side='left', fill='x', expand=True)
            return entry

        run_dir_entry = labeled_entry(top, "Run Directory:", run_dir)
        runfile_entry = labeled_entry(top, "Runfile Name:", runfile_name)
        nodes_entry = labeled_entry(top, "Nodes:", nodes)
        processors_entry = labeled_entry(top, "Processors:", processors)
        queue_entry = labeled_entry(top, "Queue Name:", queue_name)
        scheduler_entry = labeled_entry(top, "Scheduler:", scheduler)

        submit_button = ttk.Button(top, text="Submit", command=submit)
        submit_button.pack(pady=10)

        top.bind('<Return>', submit)

        top.mainloop()

        if scheduler.strip().lower() == "auto":
            self.check_scheduler()
        elif (scheduler=='pbs') or (scheduler=='slurm') or (scheduler=='bash') :
            self.scheduler = scheduler
        else:
            print('Error: Invalid scheduler type !!!')
            raise TypeError

        with open(os.path.join(self.run_dir, self.runfile_name),'w') as runfile:
            runfile.write("#!/bin/sh\n")
            if self.scheduler == 'pbs':
                runfile.write(f"#PBS -o sys_mesg.log -N SBH\n")
                runfile.write("#PBS -j oe\n")
                runfile.write(f"#PBS -l nodes={self.nodes}:ppn={self.processors}\n")
                runfile.write(f"#PBS -q {self.queue_name}\n")
                runfile.write("\n")
                runfile.write("NUMBER=`wc -l < $PBS_NODEFILE`\n")
                runfile.write("cd $PBS_O_WORKDIR\n")
                runfile.write("\nmpirun -np $NUMBER %s > stdout.dat\n"%(self.exe_path))
            elif self.scheduler == 'slurm':
                runfile.write("#SBATCH --output=sys_mesg.log\n")
                runfile.write("#SBATCH --job-name=SBH\n")
                runfile.write(f"#SBATCH --ntasks={self.processors}\n")
                runfile.write(f"#SBATCH --nodes={self.nodes}\n")
                runfile.write(f"#SBATCH --partition={self.queue_name}\n")
                runfile.write("\n")
                runfile.write("cd $SLURM_SUBMIT_DIR\n")
                runfile.write("\nmpirun -np $number %s > stdout.dat\n"%(self.exe_path))
            else:
                runfile.write("module load your_module\n")
                runfile.write("\n%s\n"%(self.exe_path))

    def find_matching_path(self, target_dir, required_paths):
        not_found_paths = []
        for required_path in required_paths:
            path = os.path.join(target_dir, required_path)
            if not os.path.exists(path):
                    not_found_paths.append(required_path)
        return not_found_paths

    def find_subdirs(self, target_name):
        matches = []
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            for d in dirs:
                if d == target_name:
                    matches.append(os.path.join(root,d))
        return matches


    def path_checker(self, parent_dir, required_paths):
        not_found_paths = self.find_matching_path(parent_dir, required_paths)
        if len(not_found_paths) == 0:
            return os.path.abspath(parent_dir)
            
        min_not_found = len(not_found_paths)
        min_not_found_paths = ' & '.join(not_found_paths)
    
        for dirpath, dirnames, filenames in os.walk(parent_dir):
            for dirname in dirnames:
                subdir_path = os.path.join(dirpath, dirname)
                not_found_paths = self.find_matching_path(subdir_path, required_paths)
                if len(not_found_paths) == 0:
                    return os.path.abspath(subdir_path)
                else:
                    if len(not_found_paths) < min_not_found:
                        min_not_found = len(not_found_paths)
                        min_not_found_paths = ' & '.join(not_found_paths)
    
        raise FileNotFoundError("Can't find required files or directories, \"%s\" in \"%s\""%(min_not_found_paths, parent_dir))

    def open_filedialog(self, title=None):
        root = tk.Tk()
        root.withdraw() 

        if title is None:
            selected_path = filedialog.askopenfilename()
        elif isinstance(title, str):
            selected_path = filedialog.askopenfilename(title=title)

        if not selected_path:
            messagebox.showwarning("No Selection", "No file was selected!")
            return None

        if os.path.exists(selected_path):
            confirm = messagebox.askyesno(
                "Confirm Path",
                f"Processing with selected path:\n\n{selected_path}\n\nIs this correct?"
            )
            if confirm:
                root.destroy()
                return selected_path
            else:
                messagebox.showinfo("Try Again", "Please select the correct path.")
                return None
        else:
            messagebox.showerror("Invalid Path", "The selected path does not exist.")
            return None

class VASPInput(BasicTool):
    def __init__(self, potcar='pbe', charge = 0):
        super().__init__()
        self.potcar = potcar
        self.charge = charge
        self.pot_recommend = {}
        self.recommended_potcar_path = os.path.join(
            os.path.dirname(__file__), self.prefix_data, 'recommended_potcar.dat')

        if not os.path.isfile(self.recommended_potcar_path):
            print('recommended_potcar.dat is not found.\n \
                  please visit https://github.com/TY-Jeong/arpaca')
            sys.exit(0)


    def make_poscar(self,filename, ase_structure):
        positions = ase_structure.get_positions()
        elements = ase_structure.get_chemical_symbols()
        sorted_indices = np.argsort(elements)
        self.sorted_structure = ase_structure[sorted_indices]
        write(filename, self.sorted_structure, format='vasp')

    def make_incar(self, filename, incar_params):
        self.atom_num = np.array(self.atom_num)
        self.zval = np.array(self.zval)
        self.nelect = np.sum(self.atom_num*self.zval)-self.charge
        incar = Incar(incar_params)
        incar.write_file(filename)

    def make_potcar(self, filename):
        self.read_path_potcar()
        self.read_recommended_potcar()
        elements = self.sorted_structure.get_chemical_symbols()
        element_counts = Counter(elements)
        self.atom_name = list(element_counts.keys())
        self.atom_num = list(element_counts.values())
        self.zval = []
        self.enmax = []

        with open(filename,'w') as f:
            for name in self.atom_name:
                pot = os.path.join(self.prefix_pot, self.pot_recommend[name], 'POTCAR')
                with open(pot, 'r') as pot_file:
                    pot_lines = [line for line in pot_file]
                for line in pot_lines:
                    f.write(line)
                    if 'ZVAL' in line:
                        self.zval += [float(line.split()[5])]
                    if 'ENMAX' in line:
                        self.enmax += [float(line.split()[2].split(';')[0])]

    def make_kpoints(self, filename, structure, density=[20,20,20]):
        self.kpoints = Kpoints.automatic_density_by_lengths(structure, length_densities=density)
        shift = [0.0,0.0,0.0]
        with open(filename, 'w') as f:
            f.write(f"{self.kpoints.comment}\n")
            f.write(f"{self.kpoints.num_kpts}\n")
            f.write(f"{self.kpoints.style.name}\n")
            for kpt in self.kpoints.kpts:
                f.write(f"{' '.join(map(str, kpt))}\n")
            f.write(f"{' '.join(map(str, shift))}\n")

    def read_recommended_potcar(self):
        with open(self.recommended_potcar_path, 'r') as file:
            lines = [line.strip() for line in file]

        for line in lines:
            name, recommend = line.split('\t')
            self.pot_recommend[name] = recommend

    def read_path_potcar(self):
        if self.potcar == 'pbe':
            self.prefix_pot = self.read_path_file('POT_PBE')
        else:
            self.prefix_pot = self.read_path_file('POT_LDA')


    def read_path_vasp(self):
        self.exe_path = self.read_path_file('vasp')

class QE_Input(BasicTool):
    def __init__(self):
        super().__init__()

    def read_path_pw(self):
        self.exe_path = self.read_path_file('qe_pw') + ' < qe.in'

    def read_path_pwcond(self):
        self.exe_path = self.read_path_file('qe_pwcond') + ' < qe.in'

    def read_path_qe_pseudo(self):
        self.prefix_qe_pseudo = self.read_path_file('qe_pseudo')

class GenBulk(BasicTool):
    def __init__(self,chemsys_formula_mpids = None):
        super().__init__()
        self.API_KEY = self.read_path_file('mp_api')
        root_directory = os.getcwd()
        self.iter = False
        self.set_project_directory('Bulk')

        if chemsys_formula_mpids == None:
            def on_submit(event=None):
                formula = entry.get()
                if formula:
                    confirm = messagebox.askyesno("Confirm Input", f"You entered: {formula}\nIs this correct?")
                    if confirm:
                        root.quit()
                        root.result = formula
                    else:
                        entry.delete(0, tk.END)
                        messagebox.showinfo("Try Again", "Please enter the chemical formula again.")
                else:
                    messagebox.showwarning("Input Error", "Please enter a chemical formula.")
        
            root = tk.Tk()
            root.title("Chemical Formula Input")
            root.geometry("300x120")
        
            tk.Label(root, text="Enter the chemical formula (e.g., TiO2):").pack(pady=10)
            entry = tk.Entry(root, width=20)
            entry.pack(pady=5)
            entry.bind("<Return>", on_submit)
        
            tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
        
            root.result = None  
            root.mainloop()
            root.destroy()
            
            chemsys_formula_mpids = root.result
            
        try:
            int(chemsys_formula_mpids)
            chemsys_formula_mpids = int(chemsys_formula_mpids)
            chemsys_formula_mpids = 'mp-%s'%chemsys_formula_mpids
            print('\n\n############### Initialize GenBulk to make bulk POSCAR for %s ###############\n'%chemsys_formula_mpids)
            self.mpid_poscar_write(chemsys_formula_mpids)
        except ValueError:
            print('\n\n############### Initialize GenBulk to make bulk POSCAR for %s ###############\n'%chemsys_formula_mpids)
            self.formula_poscar_write(chemsys_formula_mpids)

    def formula_poscar_write(self,chemsys_formula):
        with MPRester(self.API_KEY, mute_progress_bars=True) as mpr:
            print(' @ Searching MP IDs for %s'%chemsys_formula)
            material_ids = mpr.get_material_ids(chemsys_formula)
            for id_ in material_ids:
                self.mpid_poscar_write(str(id_))
            print("\n * Structure data log is written in %s"%self.project_directory+'/%s_MP_data.txt *\n'%self.formula)


    def mpid_poscar_write(self,mpid):
        with MPRester(self.API_KEY, mute_progress_bars=True) as mpr:
            print('\n * Searching material with %s'%mpid)
            entries = mpr.get_entries(mpid)
            formula = entries[0].composition.formula
            self.formula = Composition(formula).reduced_formula
            self.structure_directory = self.project_directory + '/%s'%self.formula

            self.make_directory(self.structure_directory)

            material = mpr.get_structure_by_material_id(mpid, conventional_unit_cell = True)
            space_group = material.get_space_group_info()
            space_group_symbol = space_group[0]
            space_group_number = space_group[1]

            if not self.iter:
                with open(self.project_directory+'/%s_MP_data.txt'%self.formula,'w') as MP_data:
                    MP_data.write('MP_id       Space_group_symbol  Space_group_number\n')
                    MP_data.write('-'*50+'\n')
                    MP_data.write('%-10s  %-18s  %-18s\n'%(mpid,space_group_symbol,space_group_number))

            if self.iter:
                with open(self.project_directory+'/%s_MP_data.txt'%self.formula,'a') as MP_data:
                    MP_data.write('%-10s  %-18s  %-18s\n'%(mpid,space_group_symbol,space_group_number))

            print('  -. Writing "%s/POSCAR_%s"'%(self.structure_directory, mpid))
            material.to(fmt='poscar', filename='%s/POSCAR_%s'%(self.structure_directory, mpid))
            self.iter = True

    def set_bulk(self,file_name_mpid):
        def find_files_with_keyword(self,current_dir, keyword):
            matched_files = []
            try:
                for root, dirs, files in os.walk(current_dir):
                    for file in files:
                        if keyword in file:
                            file_path = os.path.join(root, file)
                            matched_files.append(file_path)
                if len(matched_files) > 1:
                    print('Check your file_name or mpid')
                    raise ValueError
                #print('\nSetting bulk structure with %s\n'%(file_path))
                bulk = read(matched_files[0])
                return bulk

            except:
                print('Check your file_name or mpid')
                raise FileNotFoundError

        try:
            file_name_mpid = int(file_name_mpid)
            self.bulk = find_files_with_keyword(self, self.project_directory, 'POSCAR_mp-%s'%file_name_mpid)
        
        except:
            if os.path.isfile(file_name_mpid):
                self.bulk = read(file_name_mpid)

            else:
                print('Check your file_name or mpid')
                raise FileNotFoundError

    def view_structure(self, file_name_mpid):
        self.set_bulk(file_name_mpid)
        viewer = view(self.bulk)

    def view_data(self):
        def on_item_click(event):
            item = tree.selection()[0]
            mp_id = tree.item(item, "values")[0]
            print("Showing selected MP_id: %s"%mp_id)
            self.view_structure(mp_id.replace("mp-",""))
        file_path = self.project_directory+'/%s_MP_data.txt'%self.formula
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]
            data = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    mp_id = parts[0]
                    space_group_symbol = parts[1]
                    space_group_number = int(parts[2])
                    data.append((mp_id, space_group_symbol, space_group_number))
        root = tk.Tk()
        root.title(file_path)
        root.geometry("700x400")
        frame = tk.Frame(root)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        y_scrollbar = tk.Scrollbar(frame, orient="vertical")
        y_scrollbar.grid(row=0, column=1, sticky="ns", rowspan=1)
        tree = ttk.Treeview(frame, columns=("MP_id", "Space_group_symbol", "Space_group_number"), show="headings", yscrollcommand=y_scrollbar.set)
        tree.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        tree.heading("MP_id", text="MP_id")
        tree.heading("Space_group_symbol", text="Space_group_symbol")
        tree.heading("Space_group_number", text="Space_group_number")
        tree.column("MP_id", width=200)
        tree.column("Space_group_symbol", width=200)
        tree.column("Space_group_number", width=200)
        y_scrollbar.config(command=tree.yview)
        for item in data:
            tree.insert("", "end", values=item)
        tree.bind("<Double-1>", on_item_click)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        root.mainloop()
        
class GenSurface(VASPInput):
    def __init__(self, structure_file=None, miller_index=None, min_thickness=20, vacuum=10):
        self.set_project_directory('Surface')
        self.check_object_type(structure_file)
        formula = self.structure.get_chemical_formula()
        formula = Composition(formula)
        self.formula = formula.reduced_formula
        self.get_miller_index(miller_index)
        self.get_thickness(min_thickness, vacuum)
        print('\n\n############### Initialize GenSurface to make slab POSCAR for %s ###############\n'%self.formula)
        self.slab_maker()

    def check_object_type(self, obj):
        if obj == None:
            self.structure = read(self.open_filedialog('Select relaxed bulk structure file (e.g., CONTCAR)'))
        elif isinstance(obj, str):
            try:
                self.structure = read(obj)
            except:
                print('Error: Check your POSCAR file !!')
                raise FileNotFoundError
        else:
            raise TypeError

    def get_miller_index(self, miller_index):
        if miller_index != None:
            self.miller_index = miller_index
        root = tk.Tk()
        root.title("Miller Indices of surface")
        root.geometry("400x120")

        label_prompt = tk.Label(root, text="Please enter the Miller Indices of %s (e.g., (100))"%self.formula)
        label_prompt.pack(pady=10)

        frame_input = tk.Frame(root)
        frame_input.pack(pady=10)

        tk.Label(frame_input, text="h:").grid(row=0, column=0, padx=5)
        entry_h = tk.Entry(frame_input, width=5)
        entry_h.grid(row=0, column=1, padx=5)

        tk.Label(frame_input, text="k:").grid(row=0, column=2, padx=5)
        entry_k = tk.Entry(frame_input, width=5)
        entry_k.grid(row=0, column=3, padx=5)

        tk.Label(frame_input, text="l:").grid(row=0, column=4, padx=5)
        entry_l = tk.Entry(frame_input, width=5)
        entry_l.grid(row=0, column=5, padx=5)

        def on_submit(event=None):
            h, k, l = entry_h.get(), entry_k.get(), entry_l.get()
            if h and k and l:
                miller_index = f"({h}{k}{l})"
                self.miller_index = (int(h), int(k), int(l))
                root.destroy()
            else:
                label_result.config(text="Please enter all values.")

        button_submit = tk.Button(root, text="Submit", command=on_submit)
        button_submit.pack(pady=10)
        root.bind("<Return>", on_submit)
        label_result = tk.Label(root)
        label_result.pack(pady=10)
        root.mainloop()

    def get_thickness(self, min_thickness, vacuum_thickness):
        self.vacuum = vacuum_thickness
        self.slab_min_thickness = min_thickness
        def on_submit(event=None):
            try:
                slab_thickness = float(entry_slab.get())
                vacuum_thickness = float(entry_vacuum.get())
                
                if slab_thickness <= 0 or vacuum_thickness < 0:
                    raise ValueError("Thickness must be positive, and vacuum cannot be negative.")

                confirm = messagebox.askyesno("Confirm Input", 
                                              f"Slab Thickness: {slab_thickness} Å\nVacuum Thickness: {vacuum_thickness} Å\nIs this correct?")
                
                if confirm:
                    self.slab_min_thickness = slab_thickness
                    self.vacuum = vacuum_thickness
                    root.destroy()
                else:
                    messagebox.showinfo("Try Again", "Please enter the values again.")
                    entry_slab.delete(0, tk.END)
                    entry_vacuum.delete(0, tk.END)

            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}")
                entry_slab.delete(0, tk.END)
                entry_vacuum.delete(0, tk.END)

        root = tk.Tk()
        root.title("Slab and Vacuum Input")
        root.geometry("300x150")

        tk.Label(root, text="Minimum Slab Thickness (Å):").pack(pady=5)
        entry_slab = tk.Entry(root)
        entry_slab.pack(pady=5)
        
        if self.slab_min_thickness is not None:
            entry_slab.insert(0, str(self.slab_min_thickness))

        tk.Label(root, text="Vacuum Thickness (Å):").pack(pady=5)
        entry_vacuum = tk.Entry(root)
        entry_vacuum.pack(pady=5)

        if self.vacuum is not None:
            entry_vacuum.insert(0, str(self.vacuum))

        tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
        root.bind("<Return>", on_submit)
        root.mainloop()

    def calculate_thickness(self):
        self.z_coords = self.slab.get_positions()[:, 2]  
        max_z = np.max(self.z_coords)
        min_z = np.min(self.z_coords)
        self.thickness = max_z - min_z

    def semiconductor_check(self):
        def on_button_click(event=None):
            if self.var.get() == 1:
                self.is_semi = True
            else:
                self.is_semi = False
            root.destroy()
        root = tk.Tk()
        self.var = tk.IntVar()
        root.title("Semiconductor check")
        label = tk.Label(root, text="%s: Semiconductor or not?"%self.formula, font=("Arial", 14))
        label.pack(pady=20)
        yes_check = tk.Radiobutton(root, text="Yes (Semiconductor)", variable=self.var, value=1, font=("Arial", 12))
        yes_check.pack(pady=10)
        no_check = tk.Radiobutton(root, text="No (Metal)", variable=self.var, value=0, font=("Arial", 12))
        no_check.pack(pady=10)
        btn = tk.Button(root, text="Submit", font=("Arial", 12), command=on_button_click)
        btn.pack(pady=20)
        root.bind("<Return>", on_button_click)
        root.mainloop()

    def cbs_surface_maker(self):
        self.semiconductor_check()
        def on_submit(event=None):
            try:
                layer_num = int(entry.get())
                if layer_num <= 0:
                    raise ValueError("The number of layers must be a positive integer.")
                confirm = messagebox.askyesno(
                    "Confirm Input", f"You entered: {layer_num}\nIs this correct?"
                )
                if confirm:
                    root.result = layer_num
                    root.quit()
                else:
                    entry.delete(0, tk.END)
                    messagebox.showinfo("Try Again", "Please enter the  minimal number of surface layer again.")
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid positive integer.")
        if self.is_semi:
            root = tk.Tk()
            root.title("Minimal Surface Layer Input")
            root.geometry("400x120")
            tk.Label(root, text="Enter the minimal number of surface layers:").pack(pady=10)
            entry = tk.Entry(root, width=20)
            entry.pack(pady=5)
            entry.bind("<Return>", on_submit)
            tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
            root.mainloop()
            layer_num = root.result
            root.destroy()
            self.cbs_slab = surface(self.structure, self.miller_index, layers=layer_num, periodic=True) # layers=self.layer_num, 
            self.make_poscar('%s/cbs_surface.vasp'%(self.project_directory), self.cbs_slab)
            print(' &&&&& Generating POSCAR for cbs calculation at "%s/cbs_surface.vasp" &&&&&'%(self.project_directory))
            self.view_cbs_structure()

    def view_cbs_structure(self):
        if hasattr(self, 'cbs_slab'):
            view(self.cbs_slab, block=True)

    def slab_maker(self):
        stop_condition = False
        self.layer_num = 1
        while not stop_condition:
            self.layer_num += 1
            self.slab = surface(self.structure, self.miller_index, layers=self.layer_num, vacuum=self.vacuum, periodic=True)
            self.calculate_thickness()
            if self.thickness > self.slab_min_thickness:
                stop_condition = True

        print(' * Generating slab POSCAR for %s at "%s/%s_slab-%s.vasp"'%(self.formula, self.project_directory , self.formula, ''.join(map(str, self.miller_index))))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.project_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. %s %s slab thickness is %.2f Å\n'%(self.formula, ''.join(map(str, self.miller_index)),self.thickness))
        self.view_structure()

    def slice_slab_direct(self, z_min=None, z_max=None):
        if z_min is not None and z_max is not None:
            self.z_min = z_min
            self.z_max = z_max
        elif (z_min is None) or (z_max is None):
            root.destroy()
            root = tk.Tk()
            root.title("Slab Slicing Range (Direct Coordinates)")
            root.geometry("800x120")

            label_prompt = tk.Label(root, text="Please enter the slicing range of the slab in Direct coordinates (e.g., Min: 0.2, Max: 0.35)")
            label_prompt.pack(pady=10)

            frame_input = tk.Frame(root)
            frame_input.pack(pady=10)

            tk.Label(frame_input, text="Min (Direct):").grid(row=0, column=0, padx=5)
            entry_min = tk.Entry(frame_input, width=10)
            entry_min.grid(row=0, column=1, padx=5)

            tk.Label(frame_input, text="Max (Direct):").grid(row=0, column=2, padx=5)
            entry_max = tk.Entry(frame_input, width=10)
            entry_max.grid(row=0, column=3, padx=5)

            def on_submit(event=None):
                min_value, max_value = entry_min.get(), entry_max.get()
                if min_value and max_value:
                    try:
                        self.z_min = float(min_value)
                        self.z_max = float(max_value)
                        if self.z_min < self.z_max:  
                            root.destroy() 
                        else:
                            label_result.config(text="Min value must be less than Max value.")
                    except ValueError:
                        label_result.config(text="Please enter valid numerical values.")
                else:
                    label_result.config(text="Please enter both values.")
            
            button_submit = tk.Button(root, text="Submit", command=on_submit)
            button_submit.pack(pady=10)
            root.bind("<Return>", on_submit)
            label_result = tk.Label(root)
            label_result.pack(pady=10)
            root.mainloop()

        if self.z_min == None or self.z_max == None:
            return

        self.z_coords_direct = self.slab.get_scaled_positions()[:, 2]  
        mask = (self.z_coords_direct >= self.z_min) & (self.z_coords_direct <= self.z_max)
        self.slab = self.slab[mask]
        self.calculate_thickness()
        print(' * Slicing %s slab from z-coordinate from %.2f to %.2f'%(self.formula, self.z_min, self.z_max))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.project_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. Sliced %s slab thickness is %.2f Å\n'%(self.formula,self.thickness))
        self.view_structure()

    def slice_slab_cartesian(self, z_min=None, z_max=None):
        self.z_min = z_min
        self.z_max = z_max
        if (self.z_min == None) or (self.z_min == None):
            root = tk.Tk()
            root.title("Slab Slicing Range")
            root.geometry("600x120")

            label_prompt = tk.Label(root, text="Please enter the slicing range of the slab in Å (e.g., Min: 20.1 Å, Max: 35.44 Å)")
            label_prompt.pack(pady=10)

            frame_input = tk.Frame(root)
            frame_input.pack(pady=10)

            tk.Label(frame_input, text="Min (Å):").grid(row=0, column=0, padx=5)
            entry_min = tk.Entry(frame_input, width=10)
            entry_min.grid(row=0, column=1, padx=5)

            tk.Label(frame_input, text="Max (Å):").grid(row=0, column=2, padx=5)
            entry_max = tk.Entry(frame_input, width=10)
            entry_max.grid(row=0, column=3, padx=5)

            def on_submit(event=None):
                min_value, max_value = entry_min.get(), entry_max.get()
                if min_value and max_value:
                    try:
                        self.z_min = float(min_value)
                        self.z_max = float(max_value)
                        if self.z_min < self.z_max:  
                            root.destroy() 
                        else:
                            label_result.config(text="Min value must be less than Max value.")
                    except ValueError:
                        label_result.config(text="Please enter valid numerical values.")
                else:
                    label_result.config(text="Please enter both values.")
            button_submit = tk.Button(root, text="Submit", command=on_submit)
            button_submit.pack(pady=10)
            root.bind("<Return>", on_submit)
            label_result = tk.Label(root)
            label_result.pack(pady=10)
            root.mainloop()

        if self.z_min == None or self.z_max == None:
            return

        self.z_coords = self.slab.get_positions()[:, 2]
        mask = (self.z_coords >= self.z_min) & (self.z_coords <= self.z_max)
        self.slab = self.slab[mask]
        self.calculate_thickness()
        print(' * Slicing %s slab from %.2fÅ to %.2fÅ'%(self.formula, self.z_min, self.z_max))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.project_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. Slice %s slab thickness is %.2f Å\n'%(self.formula,self.thickness))
        self.view_structure()

    def xy_shift_direct(self, x, y):
        positions_direct = self.slab.get_scaled_positions()
        positions_direct += [x, y, 0]
        self.slab.set_scaled_positions(positions_direct)
        print(' * Shifting %s slab by (%f %f 0)\n'%(self.formula, x, y))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.project_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)

    def xy_shift_cartesian(self, x, y):
        self.slab.translate([x, y ,0])
        print(' * Shifting %s slab by (%f %f 0)\n'%(self.formula, x, y))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.project_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)

    def view_structure(self):
        view(self.slab, block=True)

class GenInterface(VASPInput):
    def __init__(self, substrate_surface=None, film_surface=None):
        """
        substrate_surface: str(slab POSCAR filename)
        film_surface: str(slab POSCAR filename)
        """
        self.shift = False
        self.set_project_directory('Interface')
        self.substrate = self.check_object_type(substrate_surface,'substrate')
        self.substrate_formula = self.formula
        self.film = self.check_object_type(film_surface, 'film')
        self.film_formula = self.formula
        print('\n\n############### Initialize GenInterface to make interface POSCAR for %s/%s ###############\n'%(self.substrate_formula,self.film_formula))
        self.editing_directory = self.project_directory + '/Edit'
        self.make_directory(self.editing_directory)
        self.lattice_adjustment()


    def check_object_type(self, obj, stack):
        if obj == None:
            slab = read(self.open_filedialog('Select surface structure file of %s'%stack))
        elif isinstance(obj, GenSurface):
            slab = obj.slab
        elif isinstance(obj, str):
            try:
                slab = read(obj)
            except:
                print('Error: Check your POSCAR file !!')
                raise FileNotFoundError
        else:
            raise TypeError
        formula = slab.get_chemical_formula()
        formula = Composition(formula)
        self.formula = formula.reduced_formula
        return slab

    def interface_maker(self, spacing = None, vacuum = None):
        def align_substrate_z0(self):
            min_z = np.min(self.substrate_supercell.positions[:, 2]) 
            self.substrate_supercell.translate([0, 0, -min_z])
        
        def align_film_on_substrate(self):
            max_z_substrate = np.max(self.substrate_supercell.positions[:, 2])  
            min_z_film = np.min(self.film_lattice_matched.positions[:, 2]) 
            self.film_lattice_matched.translate([0, 0, max_z_substrate - min_z_film + self.spacing]) 

        def create_interface(self):
            combined_positions = np.vstack([self.substrate_supercell.positions, self.film_lattice_matched.positions])
            max_z = np.max(combined_positions[:, 2])
            combined_symbols = self.substrate_supercell.get_chemical_symbols() + self.film_lattice_matched.get_chemical_symbols()
            combined_cell = self.substrate_supercell.get_cell() 
            if self.vacuum < self.spacing:
                self.vacuum = self.spacing
            combined_cell = self.lattice_z_transform(combined_cell, [0, 0, self.vacuum + max_z])
            combined_pbc = self.substrate_supercell.get_pbc()
            self.interface = Atoms(symbols=combined_symbols, positions=combined_positions, cell=combined_cell, pbc=combined_pbc)
            print(' * Generating interface POSCAR at "%s/interface.vasp"\n'%(self.project_directory))
            self.make_poscar("%s/interface.vasp"%self.project_directory, self.interface)
            if self.shift == True:
                print(' * Generating interface POSCAR at "%s/shifted_interface.vasp"\n'%(self.project_directory))
                self.make_poscar("%s/shifted_interface.vasp"%self.project_directory, self.interface)
                self.shift = False
            view(self.interface, block=True)


        if vacuum is not None and spacing is not None:
            self.vacuum = vacuum
            self.spacing = spacing

        else:
            def on_submit():
                try:
                    if vacuum is None:
                        self.vacuum = float(vacuum_entry.get())
                    else:
                        self.vacuum = vacuum
                    if spacing is None:
                        self.spacing = float(spacing_entry.get())
                    else:
                        self.spacing = spacing
                    root.destroy()
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers.")
    
            root = tk.Tk()
            root.title("Creates an interface by specifying the vacuum and spacing")
    
            if vacuum is None:
                tk.Label(root, text="Vacuum Thickness (Å):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
                vacuum_entry = tk.Entry(root)
                vacuum_entry.grid(row=0, column=1, padx=10, pady=5)
                tk.Label(root, text="Length of vacuum region in interface structure.").grid(
                row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    
            if spacing is None:
                tk.Label(root, text="Interface Spacing (Å):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
                spacing_entry = tk.Entry(root)
                spacing_entry.grid(row=2, column=1, padx=10, pady=5)
                tk.Label(root, text="Spacing between adjacent layers between film and substrate in the structure.").grid(
                    row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    
            submit_button = tk.Button(root, text="Submit", command=on_submit)
            last_row = 3 if spacing is None else 1
            submit_button.grid(row=last_row + 1, column=0, columnspan=2, pady=10)
    
            root.bind('<Return>', lambda event: on_submit())
            root.mainloop()

        align_substrate_z0(self)
        align_film_on_substrate(self)
        create_interface(self)
        return

    def film_shift(self, parent=None):
        def on_confirm(event=None):
            try:
                x = float(x_entry.get())
                y = float(y_entry.get())
                self.x_value = x
                self.y_value = y
    
                shift_type = self.shift_type.get()
                if shift_type == "cartesian":
                    self.film_xy_shift_cartesian(x, y)
                elif shift_type == "direct":
                    self.film_xy_shift_direct(x, y)
                else:
                    raise ValueError("Invalid shift type selected.")
    
                film_shift_window.destroy()
                if self.interface_maker:
                    try:
                        self.interface_maker(self.spacing, self.vacuum)
                    except AttributeError:
                        self.interface_maker()
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}")

        self.shift = True
    
        film_shift_window = tk.Toplevel(parent)
        film_shift_window.title("Adjust Film XY Position")
    
        self.shift_type = tk.StringVar(value="cartesian")
        shift_type_frame = tk.Frame(film_shift_window)
        shift_type_frame.pack(pady=10)
    
        label = tk.Label(shift_type_frame, text="Choose a shift type:")
        label.grid(row=0, column=0, columnspan=2, pady=5)
    
        cartesian_radio = tk.Radiobutton(
            shift_type_frame, text="Cartesian Shift (in Å, absolute distance):",
            variable=self.shift_type, value="cartesian"
        )
        cartesian_radio.grid(row=1, column=0, padx=10, sticky="w")
    
        direct_radio = tk.Radiobutton(
            shift_type_frame, text="Direct Shift (fractional, 0 to 1):",
            variable=self.shift_type, value="direct"
        )
        direct_radio.grid(row=1, column=1, padx=10, sticky="w")
    
        shift_type_description = tk.Label(
            shift_type_frame,
            text=(
                "Cartesian: Specify shifts as absolute distances in Å.\n"
                "Direct: Specify shifts as fractional values (0 to 1) relative to lattice vectors."
            ),
            fg="gray", wraplength=300, justify="left"
        )
        shift_type_description.grid(row=2, column=0, columnspan=2, pady=5)
    
        input_frame = tk.Frame(film_shift_window)
        input_frame.pack(pady=10)
    
        x_label = tk.Label(input_frame, text="Enter X value:")
        x_label.grid(row=0, column=0, padx=5, pady=5)
        x_entry = tk.Entry(input_frame)
        x_entry.grid(row=0, column=1, padx=5, pady=5)
    
        y_label = tk.Label(input_frame, text="Enter Y value:")
        y_label.grid(row=1, column=0, padx=5, pady=5)
        y_entry = tk.Entry(input_frame)
        y_entry.grid(row=1, column=1, padx=5, pady=5)
    
        confirm_button = tk.Button(film_shift_window, text="Confirm", command=on_confirm)
        confirm_button.pack(pady=10)
        film_shift_window.bind('<Return>', on_confirm)


    def film_xy_shift_direct(self, x, y):
        print(' * Shifting interface by (%f %f 0)'%(x, y))
        try:
            positions_direct = self.film_lattice_matched.get_scaled_positions()
            positions_direct += [x, y, 0]
            self.film_lattice_matched.set_scaled_positions(positions_direct)
        except:
            positions_direct = self.film
            positions_direct += [x, y, 0]
            self.film.set_scaled_positions(positions_direct)


    def film_xy_shift_cartesian(self, x, y):
        print(' * Shifting interface by %f Å, %f Å, 0.000000 Å'%(x, y))
        try:
            positions_direct = self.film_lattice_matched.translate([x, y ,0])
        except:
            positions_direct = self.film.translate([x, y ,0])


    def apply_strain(self, parent=None):
        def on_confirm(event=None):
            try:
                x_strain = float(x_entry.get())
                y_strain = float(y_entry.get())
                self.x_strain = x_strain
                self.y_strain = y_strain
                lattice_params = self.interface.cell
                self.x_strain = 0.01 * self.x_strain
                self.y_strain = 0.01 * self.y_strain
                lattice_params[0, 0] *= (1 + self.x_strain)
                lattice_params[1, 1] *= (1 + self.y_strain)
                self.strained_interface = self.interface
                self.strained_interface.set_cell(lattice_params, scale_atoms=True)
                self.make_poscar("%s/strained_interface.vasp"%self.project_directory, self.strained_interface)
                strain_window.destroy()
                view(self.strained_interface, block=True)
                print(f"Applying strain: X = {x_strain}%, Y = {y_strain}%")
                
                
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}")
    
        strain_window = tk.Toplevel(parent)
        strain_window.title("Apply Strain")
    
        instruction_frame = tk.Frame(strain_window)
        instruction_frame.pack(pady=10)
    
        instruction_label = tk.Label(
            instruction_frame, 
            text="Enter the strain values to apply (as percentages):",
            fg="black"
        )
        instruction_label.pack()
    
        input_frame = tk.Frame(strain_window)
        input_frame.pack(pady=10)
    
        x_label = tk.Label(input_frame, text="X Strain (%):")
        x_label.grid(row=0, column=0, padx=5, pady=5)
        x_entry = tk.Entry(input_frame)
        x_entry.grid(row=0, column=1, padx=5, pady=5)
    
        y_label = tk.Label(input_frame, text="Y Strain (%):")
        y_label.grid(row=1, column=0, padx=5, pady=5)
        y_entry = tk.Entry(input_frame)
        y_entry.grid(row=1, column=1, padx=5, pady=5)
    
        confirm_button = tk.Button(strain_window, text="Confirm", command=on_confirm)
        confirm_button.pack(pady=10)
        strain_window.bind('<Return>', on_confirm)
    
        strain_window.update_idletasks()
        width = strain_window.winfo_width()
        height = strain_window.winfo_height()
        x = (strain_window.winfo_screenwidth() // 2) - (width // 2)
        y = (strain_window.winfo_screenheight() // 2) - (height // 2)
        strain_window.geometry(f"{width}x{height}+{x}+{y}")



    def edit_tool(self):
        def on_select(choice):
            if choice == "Film Shift":
                self.film_shift(root)
            elif choice == "Apply Strain":
                self.apply_strain(root)

        # Initialize the main Tkinter window
        root = tk.Tk()
        root.title("Interface edit tool")

        root.geometry("300x200")

        tk.Label(root, text="Edit Tool", font=("Arial", 14)).pack(pady=10)
        tk.Label(root, text="Choose a tool:").pack(pady=5)

        film_shift_button = tk.Button(root, text="Adjust Film XY Position", command=lambda: on_select("Film Shift"))
        film_shift_button.pack(pady=5)

        apply_strain_button = tk.Button(root, text="Apply Strain", command=lambda: on_select("Apply Strain"))
        apply_strain_button.pack(pady=5)

        exit_button = tk.Button(root, text="Exit", command=root.destroy)
        exit_button.pack(pady=5)

        # Start the Tkinter main loop
        root.mainloop()

    def lattice_adjustment(self):
        root = tk.Tk()
        root.title("Lattice Adjustment")

        label = tk.Label(root, text="Choose a lattice matching option:")
        label.pack(pady=10)

        option_var = tk.StringVar()
        option_var.set('')

        manual_radio = tk.Radiobutton(
            root, text="Manual Lattice Matching", variable=option_var, value="manual")
        manual_radio.pack(anchor="w", padx=20)

        auto_radio = tk.Radiobutton(
            root, text="Auto Lattice Matching", variable=option_var, value="auto")
        auto_radio.pack(anchor="w", padx=20)

        error_label = tk.Label(root, text="", fg="red")
        error_label.pack()

        def on_confirm(event=None):
            selected_option = option_var.get()
            print(f"Selected option: {selected_option}")

            if selected_option:
                self.option = selected_option
                root.destroy()
            else:
                error_label.config(text="No option selected. Please select an option.")

        confirm_button = tk.Button(root, text="Confirm", command=on_confirm)
        confirm_button.pack(pady=10)
        root.bind("<Return>", on_confirm)

        root.mainloop()

        if self.option == "manual":
            self.manual_lattice_matching()
        elif self.option == "auto":
            self.auto_lattice_matching()
        else:
            print("No valid option selected.")

    def auto_lattice_matching(self):
        print(' * Automatically matching lattice\n')
        self.minimize_lattice_mismatch()
        self.lattice_matching()

    def manual_lattice_matching(self):
        print(' * Manually matching lattice')
        def transform_matrix(xy):
            x, y = xy
            return [[x, 0, 0],  
                    [0, y, 0],  
                    [0, 0, 1]]
        def on_confirm():
            try:
                sub_x = int(substrate_x_entry.get())
                sub_y = int(substrate_y_entry.get())
                film_x = int(film_x_entry.get())
                film_y = int(film_y_entry.get())

                self.substrate_scaling_matrix = transform_matrix([sub_x, sub_y])
                self.film_scaling_matrix = transform_matrix([film_x, film_y])

                root.destroy()

            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid integer values.")

        root = tk.Tk()
        root.title("Manual Lattice Matching")

        instructions = tk.Label(root, text="Enter x and y values for scaling matrices:")
        instructions.grid(row=0, column=0, columnspan=2, pady=10)

        substrate_label = tk.Label(root, text="Substrate Scaling Matrix:")
        substrate_label.grid(row=1, column=0, pady=5, sticky="e")

        substrate_x_entry = tk.Entry(root, width=5)
        substrate_x_entry.grid(row=1, column=1, pady=5, sticky="w")
        substrate_x_entry.insert(0, "1")

        substrate_y_entry = tk.Entry(root, width=5)
        substrate_y_entry.grid(row=1, column=2, pady=5, sticky="w")
        substrate_y_entry.insert(0, "1")

        # Film scaling matrix inputs
        film_label = tk.Label(root, text="Film Scaling Matrix:")
        film_label.grid(row=2, column=0, pady=5, sticky="e")

        film_x_entry = tk.Entry(root, width=5)
        film_x_entry.grid(row=2, column=1, pady=5, sticky="w")
        film_x_entry.insert(0, "1")

        film_y_entry = tk.Entry(root, width=5)
        film_y_entry.grid(row=2, column=2, pady=5, sticky="w")
        film_y_entry.insert(0, "1")

        # Confirm button
        confirm_button = tk.Button(root, text="Confirm", command=on_confirm)
        confirm_button.grid(row=3, column=0, columnspan=3, pady=10)

        # Run the GUI
        root.mainloop()

        # Print the scaling matrices after the GUI closes
        print(f"Substrate scaling matrix: {self.substrate_scaling_matrix}")
        print(f"Film scaling matrix: {self.film_scaling_matrix}")

        self.substrate_scaling_matrix = transform_matrix(substrate_scaling_matrix)
        self.film_scaling_matrix = transform_matrix(film_scaling_matrix)
        self.lattice_matching()

    def minimize_lattice_mismatch(self):
        print(' * Minimizing lattice mismatch')
        substrate_lattice = np.array(self.substrate.get_cell())
        film_lattice = np.array(self.film.get_cell())

        def find_best_multiples(a1, a2):
            best_k1 = None
            best_k2 = None
            min_diff = float('inf') 
            should_break = False

            for k1 in range(1, 6):  
                for k2 in range(1, 6):  
                    diff = abs(k1 * a1 - k2 * a2)

                    if diff/(a2*k1) < 0.05:
                        min_diff = diff
                        best_k1 = k1
                        best_k2 = k2
                        should_break = True
                        break

                    if diff < min_diff:
                        min_diff = diff
                        best_k1 = k1
                        best_k2 = k2

                if should_break:
                    break
            return best_k1, best_k2, min_diff/(a2*best_k1)


        x_substrate = substrate_lattice[0][0]
        y_substrate = substrate_lattice[1][1]
        x_film = film_lattice[0][0]
        y_film = film_lattice[1][1]

        # without lotation
        x_multiple = find_best_multiples(x_substrate, x_film)
        y_multiple = find_best_multiples(y_substrate, y_film)
        areal_mismatch = (x_multiple[2]+1)*(y_multiple[2]+1)

        # with 90 degree lotation
        x_multiple_ = find_best_multiples(y_substrate, x_film)
        y_multiple_ = find_best_multiples(x_substrate, y_film)
        areal_mismatch_ = (x_multiple_[2]+1)*(y_multiple_[2]+1)

        if areal_mismatch <= areal_mismatch_:
            self.areal_mismatch = areal_mismatch
            self.x_multiple = x_multiple
            self.y_multiple = y_multiple

        else:
            self.areal_mismatch = areal_mismatch_
            self.x_multiple = x_multiple_
            self.y_multiple = y_multiple_
            print('You should rotate one of your system to minimize lattice mismatch')

        print('  -. The lattice mismatch is minimal with %i x %i for substrate'%(self.x_multiple[1],self.y_multiple[1]))
        print('     The lattice mismatch is minimal with %i x %i for film'%(self.x_multiple[0],self.y_multiple[0]))

        self.substrate_scaling_matrix = [[self.x_multiple[1], 0, 0],  
                               [0, self.y_multiple[1], 0],  
                               [0, 0, 1]]

        self.film_scaling_matrix = [[self.x_multiple[0], 0, 0],  
                               [0, self.y_multiple[0], 0],  
                               [0, 0, 1]]

    def lattice_matching(self):
        self.substrate_supercell = make_supercell(self.substrate, self.substrate_scaling_matrix)
        self.film_supercell = make_supercell(self.film, self.film_scaling_matrix)
        substrate_lattice = np.array(self.substrate_supercell.get_cell())
        film_lattice = np.array(self.film_supercell.get_cell())

        x_mismatch = abs(substrate_lattice[0][0]-film_lattice[0][0])/film_lattice[0][0]
        y_mismatch = abs(substrate_lattice[1][1]-film_lattice[1][1])/film_lattice[1][1]

        print('  -. Lattice mismatch along x-axis:: %.2f%%'%(100*x_mismatch))
        print('     Lattice mismatch along y-axis:: %.2f%%\n'%(100*y_mismatch))

        cell = self.lattice_z_transform(substrate_lattice ,self.film_supercell.get_cell()[2])

        self.film_lattice_matched = Atoms(symbols=self.film_supercell.get_chemical_symbols(),
            scaled_positions=self.film_supercell.get_scaled_positions(),
            cell = cell,
            pbc= True)

        self.make_poscar("%s/Supercell_%s_POSCAR.vasp"%(self.editing_directory, self.substrate_formula), self.substrate_supercell)
        self.make_poscar("%s/Lattice_matched_supercell_%s_POSCAR.vasp"%(self.editing_directory, self.film_formula), self.film_lattice_matched)

    def lattice_z_transform(self, original_lattice, z_latt_to_transform):
        return np.array([original_lattice[0], original_lattice[1], z_latt_to_transform])

class BulkSet(VASPInput):
    def __init__(self, bulk = None, potcar = 'pbe', ncore = 4, charge = 0):
        super().__init__(potcar, charge)
        self.check_object_type(bulk)
        self.incar_params = {
            'ENCUT': 600,
            'NSW': 300,
            'ISIF': 3,
            'IBRION': 2,
            'PREC': 'Accurate',
            'ALGO': 'Normal',
            'EDIFF': 1e-8,
            'EDIFFG': -0.001,
            'ISMEAR': 0,
            'SIGMA': 0.01,
            'LCHARG': False,
            'LWAVE': False,
            'LREAL': False,
            'LORBIT': 11,
            'NCORE': ncore,
            }

        print('\n\n############### Initialize BulkSet to make calculation sets for %s ###############\n'%self.formula)
        self.relax_directory = self.working_directory+'/Relax'
        self.chg_directory = self.working_directory+'/Chg'
        self.dos_directory = self.working_directory+'/DOS'
        self.dielec_directory = self.working_directory+'/Dielec'
        self.corrected_directory = self.working_directory+'/Correction'

        self.relax_setup()
        self.semiconductor_check()

    def semi_setup(self):
        print('\n&&&&& Configure additional calculation set for semiconductor %s &&&&&\n'%self.formula)
        self.chg_setup()
        self.dos_setup()
        self.dielectric_setup()
        self.corrected_setup()
        self.is_semi = True

    def check_object_type(self, obj):
        self.set_project_directory('Bulk')

        if obj == None:
            obj = self.open_filedialog('Select bulk structure file (e.g., POSCAR)')
            try:
                self.structure = read(obj)
            except Exception as e:
                print("An error occurred while opening the file:", e)
                sys.exit(1)

        elif isinstance(obj, str):
                try:
                    self.structure = read(obj)
                except:
                    print('Check your bulk POSCAR file path')
                    raise FileNotFoundError
        else:
            raise TypeError

        self.formula = Composition(self.structure.get_chemical_formula()).reduced_formula
        self.working_directory = self.project_directory + '/%s'%self.formula
        self.make_directory(self.working_directory)

    def relax_setup(self, custom_incar_params=None):
        print(' * Making relaxation calculation set for %s in "%s"\n'%(self.formula, self.relax_directory))
        incar_params_dict = copy.deepcopy(self.incar_params)
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.relax_directory)
        self.make_poscar(self.relax_directory + '/POSCAR', self.structure)
        self.make_potcar(self.relax_directory + '/POTCAR')
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.relax_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.relax_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [20,20,20])
        return

    def chg_setup(self, custom_incar_params=None):
        print(' * Making CHGCAR calculation set for %s in "%s"\n'%(self.formula, self.chg_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.chg_directory)
        if not os.path.islink(self.chg_directory+'/POSCAR'):
            os.symlink('../Relax/CONTCAR', self.chg_directory+'/POSCAR')
        self.make_potcar(self.chg_directory + '/POTCAR')
        incar_params_dict = copy.deepcopy(self.incar_params)
        incar_params_dict['LCHARG'] = True
        incar_params_dict['IBRION'] = -1
        incar_params_dict['NSW'] = 0
        del incar_params_dict['EDIFFG'], incar_params_dict['ISIF']
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.chg_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.chg_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [20,20,20])

    def dos_setup(self, custom_incar_params=None):
        print(' * Making DOS calculation set for %s in "%s" \n'%(self.formula, self.dos_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.dos_directory)
        if not os.path.islink(self.dos_directory+'/POSCAR'):
            os.symlink('../Relax/CONTCAR', self.dos_directory+'/POSCAR')
        if not os.path.islink(self.dos_directory+'/CHGCAR'):
            os.symlink('../Chg/CHGCAR', self.dos_directory+'/CHGCAR')
        self.make_potcar(self.dos_directory + '/POTCAR')
        incar_params_dict = copy.deepcopy(self.incar_params)
        incar_params_dict['NEDOS'] = 2001
        incar_params_dict['ICHARG'] = 11
        incar_params_dict['IBRION'] = -1
        incar_params_dict['NSW'] = 0
        incar_params_dict['EMIN'] = -20
        incar_params_dict['EMAX'] = 20
        incar_params_dict['ISMEAR'] = -5
        del incar_params_dict['EDIFFG'], incar_params_dict['ISIF']
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.dos_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.dos_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [40,40,40])

    def dielectric_setup(self, custom_incar_params=None):
        print(' * Making dielectric calculation set for %s in "%s" \n'%(self.formula, self.dielec_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")

        self.incar_params_dict = copy.deepcopy(self.incar_params)
        self.incar_params_dict['IBRION'] = 8
        self.incar_params_dict['LEPSILON'] = True
        del self.incar_params_dict['NSW'], self.incar_params_dict['NCORE'], self.incar_params_dict['ISIF'], self.incar_params_dict['LORBIT'], self.incar_params_dict['EDIFFG']
        self.incar_params_dict['ENCUT'] = np.max(self.enmax)

        def pseudo_setup(self, target_directory):
            self.make_directory(target_directory)
            if not os.path.islink(target_directory+'/POSCAR'):
                os.symlink('/Relax/CONTCAR', target_directory+'/POSCAR')
            self.make_potcar(target_directory + '/POTCAR')
            self.make_incar(target_directory + '/INCAR', self.incar_params_dict)
            self.make_kpoints(target_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [40,40,40])

        self.make_directory(self.dielec_directory)
        if self.potcar == 'pbe':
            self.dielec_pbe_directory = self.dielec_directory + '/PBE'
            pseudo_setup(self,self.dielec_pbe_directory)
            self.potcar = 'lda'
            self.dielec_lda_directory = self.dielec_directory + '/LDA'
            self.make_directory(self.dielec_lda_directory)
            original_relax_directory = self.relax_directory
            self.relax_directory = self.dielec_lda_directory + '/Relax'
            self.relax_setup()
            pseudo_setup(self,self.dielec_lda_directory)
            self.relax_directory = original_relax_directory
            self.potcar = 'pbe'

        elif self.potcar == 'lda':
            self.dielec_lda_directory = self.dielec_directory + '/LDA'
            pseudo_setup(self,self.dielec_lda_directory)
            self.potcar = 'pbe'
            self.dielec_pbe_directory = self.dielec_directory + '/PBE'
            self.make_directory(self.dielec_pbe_directory)
            original_relax_directory = self.relax_directory
            self.relax_directory = self.dielec_pbe_directory + '/Relax'
            self.relax_setup()
            pseudo_setup(self,self.dielec_pbe_directory)
            self.relax_directory = original_relax_directory
            self.potcar = 'lda'         

    def corrected_setup(self, custom_incar_params=None):
        print(' * Making bandgap correction calculation set for %s in "%s"\n'%(self.formula, self.corrected_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.corrected_directory)
        if not os.path.islink(self.corrected_directory+'/POSCAR'):
            if not os.path.exists(self.corrected_directory+'/POSCAR'):
                os.symlink('../Relax/CONTCAR', self.corrected_directory+'/POSCAR')
        self.make_potcar(self.corrected_directory + '/POTCAR')
        incar_params_dict = copy.deepcopy(self.incar_params)
        incar_params_dict['LCHARG'] = True
        incar_params_dict['IBRION'] = -1
        incar_params_dict['NSW'] = 0
        incar_params_dict['ALGO'] = 'All'
        incar_params_dict['TIME'] = 0.5
        incar_params_dict['LHFCALC'] = True
        incar_params_dict['HFSCREEN'] = 0.2
        incar_params_dict['AEXX'] = 0.25
        del incar_params_dict['EDIFFG']
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.corrected_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.corrected_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [20,20,20])

    def semiconductor_check(self):
        def on_button_click(event=None):
            if self.var.get() == 1:
                self.semi_setup()
            else:
                self.is_semi = False
            root.destroy()
        root = tk.Tk()
        self.var = tk.IntVar()
        root.title("Semiconductor check")
        label = tk.Label(root, text="%s: Semiconductor or not?"%self.formula, font=("Arial", 14))
        label.pack(pady=20)
        yes_check = tk.Radiobutton(root, text="Yes (Semiconductor)", variable=self.var, value=1, font=("Arial", 12))
        yes_check.pack(pady=10)
        no_check = tk.Radiobutton(root, text="No (Metal)", variable=self.var, value=0, font=("Arial", 12))
        no_check.pack(pady=10)
        btn = tk.Button(root, text="Submit", font=("Arial", 12), command=on_button_click)
        btn.pack(pady=20)
        root.bind("<Return>", on_button_click)
        root.mainloop()


    def runfile_setup(self,runfile_name='run.sh', nodes='2', processors='16', queue_name='Null', scheduler="Auto"):
        self.read_path_vasp()
        for dirpath, dirnames, filenames in os.walk(self.working_directory):
            if 'POSCAR' in filenames:
                self.make_runfile(dirpath,runfile_name, nodes, processors, queue_name, scheduler=scheduler)

class SurfaceSet(QE_Input):
    def __init__(self, surface):
        super().__init__()
        print('\n\n############### Initialize SurfaceSet to make calculation sets for complex band structure ###############\n')
        self.check_object_type(surface)
        self.QE_directory = os.path.join(self.working_directory,'QE')
        self.make_directory(self.QE_directory)
        self.make_directory(os.path.join(self.QE_directory,'pseudos'))

    def check_object_type(self, obj):
        if isinstance(obj, str):
                try:
                    self.structure = read(obj)
                    self.working_directory = obj.replace('/cbs_surface.vasp','')
                    self.make_directory(self.working_directory)
                except:
                    print('Check your surface POSCAR file path')
                    raise FileNotFoundError
        else:
            raise TypeError

    def write_pseudo(self, elements):

        def select_pseudopotentials(elements, pseudo_dir):

            def browse_file(el, entry):
                initial_dir = pseudo_dir
                filetypes = [("UPF files", f"{el}.*.UPF")]
                filename = filedialog.askopenfilename(
                    title=f"Select pseudopotential file for {el}",
                    initialdir=initial_dir,
                    filetypes=filetypes
                )
                if filename:
                    entry.delete(0, tk.END)
                    entry.insert(0, filename)
        
            root = tk.Tk()
            root.title("Select Pseudopotential Files")
            root.geometry("600x200")
        
            def on_close():
                messagebox.showwarning("No selection", "No pseudopotential selected. The program will now exit.")
                root.destroy()
                sys.exit(1)
        
            root.protocol("WM_DELETE_WINDOW", on_close)
        
            entries = {}
            selected = {}
        
            for idx, el in enumerate(elements):
                tk.Label(root, text=f"{el}:").grid(row=idx, column=0, padx=10, pady=5, sticky="e")
                entry = tk.Entry(root, width=50)
                entry.grid(row=idx, column=1, padx=5, pady=5)
                entries[el] = entry
                tk.Button(root, text="Browse", command=lambda e=el, ent=entry: browse_file(e, ent)).grid(row=idx, column=2, padx=5)
        
            def submit():
                for el, ent in entries.items():
                    val = ent.get().strip()
                    if val == "":
                        messagebox.showwarning("Incomplete", "All pseudopotential files must be selected. Exiting.")
                        root.destroy()
                        sys.exit(1)
                    selected[el] = val
                root.destroy()

            submit_btn = tk.Button(root, text="Submit", command=submit)
            submit_btn.grid(row=len(elements), column=1, pady=15)
            root.bind("<Return>", lambda event: submit())
            root.mainloop()
            return selected

        self.read_path_qe_pseudo()
        self.selected_paths = select_pseudopotentials(elements, self.prefix_qe_pseudo)
        for el, src_path in self.selected_paths.items():
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"File not found: {src_path}")
            dst_path = os.path.join(os.path.join(self.QE_directory, 'pseudos'), os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
        self.pseudopotentials = {el: os.path.basename(path) for el, path in self.selected_paths.items()}

    def scf_setup(self, pseudopotentials=None, custom_option=None):
        self.formula = Composition(self.structure.get_chemical_formula()).reduced_formula
        elements = list(set(self.structure.get_chemical_symbols()))
        self.write_pseudo(elements)
        self.QE_SCF_directory = self.QE_directory + '/SCF'
        print(' * Making Quantum espresso SCF calculation set for %s in "%s" \n'%(self.formula, os.path.abspath(self.QE_SCF_directory)))
        self.make_directory(self.QE_SCF_directory)
        self.structure = AseAtomsAdaptor.get_structure(self.structure)
        self.num_total_electron = 0
        for element in self.structure.composition:
            atomic_number = Element(element).Z
            self.num_total_electron += atomic_number * self.structure.composition[element]
        self.num_band = self.num_total_electron//2

        self.kpoints = Kpoints.automatic_density_by_lengths(self.structure, length_densities=[20,20,20])
        kpoints_grid = self.kpoints.kpts[0]
        qe_input = PWInput(
            structure=self.structure,
            control={"calculation": "scf", "etot_conv_thr": 1.0e-8,
            "prefix": "scf" , "pseudo_dir": "../pseudos", "verbosity": "high"},
            system={"ecutwfc": 80, "ecutrho": 500, "nat": self.structure.num_sites, "ntyp": len(self.structure.symbol_set), "nbnd": self.num_band, "nosym": True},
            electrons={"conv_thr": 1.0e-8, "mixing_beta": 0.4},
            kpoints_grid=kpoints_grid,
            pseudo = self.pseudopotentials
            )
        qe_input.write_file(self.QE_SCF_directory+'/qe.in')

    def cbs_setup(self, bandgap, kmesh, denergy = 0.02):
        self.QE_CBS_directory = self.QE_directory + '/CBS'
        print(' * Making Quantum espresso CBS calculation set for %s in "%s" \n'%(self.formula, os.path.abspath(self.QE_CBS_directory)))
        self.make_directory(self.QE_CBS_directory)
        denergy = denergy
        max_energy = round(bandgap*1.1, 1)
        min_energy = round(bandgap,1) - max_energy
        energy_range = max_energy-min_energy
        pwcond_str = " &inputcond\n" \
        "   prefixl = scf\n" \
        "   band_file = cbs\n" \
        "   ikind = 0\n" \
        "   energy0 = %f\n" \
        "   denergy = -%f\n" \
        "   ewind = 4.0\n" \
        "   epsproj = 1.d-8\n" \
        "   nz1 = 11\n" \
        " /\n"%(max_energy, denergy)
        pwcond_str += kmesh
        pwcond_str += '    %i'%(energy_range/denergy)
        
        with open(self.QE_CBS_directory+"/qe.in","w") as pwcond_input:
            pwcond_input.write(pwcond_str)

        if not os.path.islink(self.QE_CBS_directory+'/scf.save'):
            if not os.path.exists(self.QE_CBS_directory+'/scf.save'):
                os.symlink('../SCF/scf.save', self.QE_CBS_directory+'/scf.save')

    def runfile_setup(self,runfile_name='run.sh', nodes='1', processors='32', queue_name='Null', scheduler="Auto"):
        self.read_path_pw()
        self.make_runfile(self.QE_SCF_directory, runfile_name, nodes, processors, queue_name, scheduler)
        self.read_path_pwcond()
        self.make_runfile(self.QE_CBS_directory, runfile_name, nodes, processors, queue_name, scheduler)

class InterfaceSet(VASPInput):
    def __init__(self, interface,  potcar = 'pbe', ncore = 4 ,charge = 0):
        super().__init__(potcar, charge)
        self.check_object_type(interface)
        self.incar_params = {
            'ENCUT': 600,
            'NSW': 1000,
            'ISIF': 2,
            'IBRION': 2,
            'PREC': 'Accurate',
            'ALGO': 'Normal',
            'EDIFF': 1e-6,
            'EDIFFG': -0.02,
            'ISMEAR': 0,
            'SIGMA': 0.01,
            'LCHARG': False,
            'LWAVE': False,
            'LREAL': 'Auto',
            'LORBIT': 11,
            'NCORE': ncore,
            'AMIN': 0.01,
            'BMIX': 3
            }
        print('\n\n############### Initialize InterfaceSet to make calculation sets for interface ###############\n')
        self.relax_setup()
        self.chg_setup()
        self.dos_setup()

    def check_object_type(self, obj):
        self.set_project_directory("Interface")
        if obj == None:
            obj = self.open_filedialog('Select interface structure file (e.g., interface.vasp)')
            try:
                self.structure = read(obj)
            except Exception as e:
                print("An error occurred while opening the file:", e)
                sys.exit(1)

        elif isinstance(obj, str):
                try:
                    self.structure = read(obj)
                except:
                    print('Check your interface POSCAR file path')
                    raise FileNotFoundError
        else:
            raise TypeError

    def relax_setup(self, custom_incar_params=None):
        self.relax_directory = self.project_directory+'/Relax'
        print(' * Making relaxation calculation set for interface in "%s"\n'%(self.relax_directory))
        incar_params_dict = copy.deepcopy(self.incar_params)
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.relax_directory)
        self.make_poscar(self.relax_directory + '/POSCAR', self.structure)
        self.make_potcar(self.relax_directory + '/POTCAR')
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.relax_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.relax_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [20,20,1])
        return

    def chg_setup(self, custom_incar_params=None):
        self.chg_directory = self.project_directory+'/Chg'
        print(' * Making CHGCAR calculation set for interface in "%s" \n'%(self.chg_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.chg_directory)
        if not os.path.islink(self.chg_directory+'/POSCAR'):
            os.symlink('../Relax/CONTCAR', self.chg_directory+'/POSCAR')
        self.make_potcar(self.chg_directory + '/POTCAR')
        incar_params_dict = copy.deepcopy(self.incar_params)
        incar_params_dict['LCHARG'] = True
        incar_params_dict['IBRION'] = -1
        incar_params_dict['NSW'] = 0
        del incar_params_dict['EDIFFG'], incar_params_dict['ISIF']
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.chg_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.chg_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [20,20,1])
        return

    def dos_setup(self, custom_incar_params=None):
        self.dos_directory = self.project_directory+'/DOS'
        print(' * Making DOS calculation set for interface in "%s" \n'%(self.dos_directory))
        if custom_incar_params != None:
            if isinstance(custom_incar_params, dict):
                incar_params.update(custom_incar_params)
            else:
                print("Custom INCAR parameters have to be entered with dictionary")
        self.make_directory(self.dos_directory)
        if not os.path.islink(self.dos_directory+'/POSCAR'):
            os.symlink('../Relax/CONTCAR', self.dos_directory+'/POSCAR')
        if not os.path.islink(self.dos_directory+'/CHGCAR'):
            os.symlink('../Chg/CHGCAR', self.dos_directory+'/CHGCAR')
        self.make_potcar(self.dos_directory + '/POTCAR')
        incar_params_dict = copy.deepcopy(self.incar_params)
        incar_params_dict['NEDOS'] = 2001
        incar_params_dict['ICHARG'] = 11
        incar_params_dict['IBRION'] = -1
        incar_params_dict['NSW'] = 0
        incar_params_dict['EMIN'] = -20
        incar_params_dict['EMAX'] = 20
        incar_params_dict['ISMEAR'] = -5
        del incar_params_dict['EDIFFG'], incar_params_dict['ISIF']
        incar_params_dict['ENCUT'] = np.max(self.enmax)
        self.make_incar(self.dos_directory + '/INCAR', incar_params_dict)
        self.make_kpoints(self.dos_directory + '/KPOINTS', AseAtomsAdaptor.get_structure(self.structure), [40,40,1])
        return

    def runfile_setup(self,runfile_name='run.sh', nodes='4', processors='16', queue_name='Null', scheduler="Auto"):
        self.read_path_vasp()
        for dirpath, dirnames, filenames in os.walk(self.project_directory):
            if 'POSCAR' in filenames:
                self.make_runfile(dirpath, runfile_name, nodes, processors, queue_name, scheduler=scheduler)

class GetSBHResult(BasicTool):
    def __init__(self):
        super().__init__()
        self.set_project_directory("Result")
        self.result_path_name = os.path.basename(self.project_directory)

    def get_vasp_result(self, bulk_path = None, interface_path = None):
        self.bulk_path = self.check_object_type(bulk_path,'Bulk', ['DOS', 'Correction', 'Dielec', 'Chg', 'Relax'])
        self.interface_path = self.check_object_type(interface_path, 'Interface', ['DOS', 'Chg', 'Relax'])
        self.path_setting()
        print('\n\n############### Retrieving bulk calculation results ###############')
        self.get_bulk_dos(self.bulk_dos_path)
        self.get_bulk_corrected(self.bulk_corrected_path)
        self.scissor_bulk_dos()
        self.get_bulk_dielec(self.bulk_dielec_path)
        print('\n\n############### Retrieving interface calculation results ###############')
        self.get_interface_dos(self.interface_dos_path)
        self.get_semi_1st_layer_dos()#15.9,18.5)
        self.get_metal_1st_layer_dos()#12.7,15.3)
        self.get_semi_next_layer_dos(self.semi_1st_layer_min_z, 
            self.semi_1st_layer_max_z, self.semi_1st_layer_thickness)
        semi_2nd_layer_position = (self.semi_layer_min_z+self.semi_layer_max_z)/2
        self.semi_2nd_layer_depth = min([abs(semi_2nd_layer_position-self.metal_1st_layer_max_z), 
            abs(semi_2nd_layer_position-self.metal_1st_layer_min_z)])
        self.get_kp_resolved_dos(self.interface_dos_path)
        self.get_semi_next_layer_dos(self.semi_layer_min_z, 
            self.semi_layer_max_z, self.semi_layer_thickness)
        self.scissor_total_1st_layer_dos()
        self.scissor_kp_dos()
        self.write_vasp_result()
        return

    def qe_setup(self, surface_path = None, scheduler = None):     
        self.result_path = self.check_object_type(None, self.result_path_name, ['DOStot.dat', 'k_mesh.dat', 'dos_bulk.dat', 'kpdos_int_2.dat', 'vasp_result.dat'])
        self.surface_path = self.check_object_type(surface_path, 'Surface', [])
        self.surface_calc_set = SurfaceSet(self.surface_path+'/cbs_surface.vasp')
        self.surface_calc_set.scf_setup()
        self.original_bandgap = np.genfromtxt(self.result_path + '/vasp_result.dat', comments='!', usecols=0)[0]
        with open(self.result_path+'/k_mesh.dat','r') as kmesh_dat:
            kmesh = kmesh_dat.read() 
        self.surface_calc_set.cbs_setup(self.original_bandgap, kmesh)
        self.surface_calc_set.runfile_setup(scheduler = scheduler)

    def get_qe_result(self, surface_path = None):
        print('\n\n############### Retrieving QE calculation results ###############')
        self.result_path = self.check_object_type(None, self.result_path_name, ['DOStot.dat', 'k_mesh.dat', 'dos_bulk.dat', 'kpdos_int_2.dat', 'vasp_result.dat'])
        self.surface_path = self.check_object_type(surface_path, 'Surface', [])
        self.get_qe_scf()
        self.get_qe_cbs()
        self.scissor_cbs()
        self.write_qe_result()
        return

    def check_object_type(self, obj = None, name = None, required_dirs = None):
        def choose_path(options, name):
            selected = {"value": None}
            def on_ok():
                selected["value"] = var.get()
                window.destroy()

            def on_cancel():
                window.destroy()

            window = tk.Tk()
            window.title("Select Path for %s:"%name)
            window.geometry("480x300")
            tk.Label(window, text="Choose one of the following directories: ", pady=10).pack()
            var = tk.StringVar(value="")
            for option in options:
                tk.Radiobutton(window, text=option, variable=var, value=option).pack(anchor="w", padx=20)

            tk.Radiobutton(window, text="Skip and choose manually", variable=var, value="__NONE__").pack(anchor="w", padx=20, pady=(10, 0))
            btn_frame = tk.Frame(window)
            btn_frame.pack(pady=15)
            
            tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=10)
            tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side=tk.RIGHT, padx=10)
            window.bind("<Return>", lambda event: on_ok())
            window.mainloop()
            return selected["value"]

        def select_directory(name):
            root = tk.Tk()
            root.withdraw()  
            selected_dir = filedialog.askdirectory(title="Select a directory for results of %s calculation"%name)
            if not selected_dir:
                messagebox.showwarning(parent=root, title="No Directory Selected",
                    message="No directory was selected. The program will now exit.")
                root.destroy()
                sys.exit(1)
            confirm = messagebox.askyesno(parent=root, title="Confirm Directory",
                message=f"Use the selected directory?\n\n{selected_dir}")

            if confirm:
                root.destroy()
                return selected_dir
            else:
                messagebox.showinfo(parent=root, title="Directory Not Confirmed",
                    message="The selected directory was not confirmed. The program will now exit.")
                root.destroy()
                sys.exit(1)

        if isinstance(obj, str):
            target_path = obj
            return target_path
        elif obj == None:
            try:
                dir_list = self.find_subdirs(name)
                target_path_list = []
                for dir_ in dir_list:
                    target_path_list.append(self.path_checker(dir_, required_dirs))
                target_path = choose_path(target_path_list, name)
                if target_path and target_path != "__NONE__":
                    return target_path   
                return select_directory(name)
            except SystemExit:
                raise    
            except:
                raise FileNotFoundError(f"Can't find %s directory"%name)
        else:
            raise TypeError

    def path_setting(self):
        self.bulk_dos_path = self.bulk_path + '/DOS'
        self.bulk_corrected_path = self.bulk_path + '/Correction'
        self.bulk_dielec_path = self.bulk_path + '/Dielec'
        self.interface_dos_path = self.interface_path +'/DOS'
        self.result_path = self.project_directory

    def get_bulk_dos(self, bulk_dos_path):
        print("\n * Obtaining bulk DOS result ")
        vasprun = Vasprun(bulk_dos_path+'/vasprun.xml')
        band_structure = vasprun.get_band_structure()
        self.original_bandgap = band_structure.get_band_gap()['energy']
        print("  -. Underestimated bandgap: %f"%self.original_bandgap)
        dos = vasprun.complete_dos
        energies = dos.energies - dos.efermi
        tdos = dos.densities[Spin.up]
        self.raw_bulk_dos = np.column_stack((energies, tdos))

    def get_bulk_corrected(self, bulk_corrected_path):
        print("\n * Obtaining bulk bandgap correction result ")
        vasprun = Vasprun(bulk_corrected_path+'/vasprun.xml')
        self.bulk_volume = vasprun.final_structure.volume
        band_structure = vasprun.get_band_structure()
        self.corrected_bandgap = band_structure.get_band_gap()['energy']
        print("  -. Corrected bandgap: %f\n"%self.corrected_bandgap)

    def scissor_bulk_dos(self):
        print(" * Extending the bandgap from %f to %f using a scissor operator\n"%(self.original_bandgap, self.corrected_bandgap))
        self.scissored_bulk_dos = copy.deepcopy(self.raw_bulk_dos)
        for i in range(len(self.raw_bulk_dos)):
            energy = self.raw_bulk_dos[i][0]
            if 0 <= energy <= self.original_bandgap:
                scissored_energy = energy * self.corrected_bandgap / self.original_bandgap
                self.scissored_bulk_dos[i][0] = scissored_energy
            elif energy > self.original_bandgap:
                self.scissored_bulk_dos[i][0] = self.scissored_bulk_dos[i-1][0] + self.raw_bulk_dos[i][0] - self.raw_bulk_dos[i-1][0]

    def get_bulk_dielec(self, bulk_dielec_path):
        print(" * Obtaining bulk dielectric constant result ")
        def get_dielec(target_directory):
            vasprun = Vasprun(target_directory+'/vasprun.xml')
            electronic_dielec = vasprun.epsilon_static
            electronic_diagonal = np.diagonal(electronic_dielec)
            electronic_mean = np.mean(electronic_diagonal)
            ionic_dielec = vasprun.epsilon_ionic
            ionic_diagonal = np.diagonal(ionic_dielec)
            ionic_mean = np.mean(ionic_diagonal)
            er = electronic_mean + ionic_mean
            return er

        def choose_method(pbe_value, lda_value):
            selected = {"value": None}
            def on_ok():
                    selected["value"] = var.get()
                    window.destroy()
            
            def on_cancel():
                window.destroy()
                sys.exit("Program exited: dielectric method not selected.")
        
            window = tk.Tk()
            window.title("Select Dielectric Constant Method")
            window.geometry("460x180")
            window.protocol("WM_DELETE_WINDOW", on_cancel)  # [X] 버튼 누를 때 종료
        
            tk.Label(window, text="Choose a method for dielectric constant:", pady=15).pack()
        
            var = tk.StringVar(value="pbe")
        
            frame = tk.Frame(window)
            frame.pack()
        
            # PBE row
            row1 = tk.Frame(frame)
            row1.pack(anchor="w", padx=30, pady=5)
            tk.Radiobutton(row1, variable=var, value="pbe").pack(side="left")
            tk.Label(row1, text=f"PBE: {pbe_value:.6f}").pack(side="left", padx=10)
        
            # LDA row
            row2 = tk.Frame(frame)
            row2.pack(anchor="w", padx=30, pady=5)
            tk.Radiobutton(row2, variable=var, value="lda").pack(side="left")
            tk.Label(row2, text=f"LDA: {lda_value:.6f}").pack(side="left", padx=10)
        
            btn_frame = tk.Frame(window)
            btn_frame.pack(pady=20)
        
            tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=10)
            tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="right", padx=10)
        
            window.bind("<Return>", lambda event: on_ok())  # ⏎ = OK
            window.mainloop()
        
            return selected["value"]

        pbe_er = get_dielec(bulk_dielec_path+'/PBE')
        lda_er = get_dielec(bulk_dielec_path+'/LDA')
        choice = choose_method(pbe_er, lda_er)

        if choice == 'pbe':
            print("  -. PBE, dielectric constant: %f" % pbe_er)
            self.er = pbe_er
        elif choice == 'lda':
            print("  -. LDA, dielectric constant: %f" % lda_er)
            self.er = lda_er
        else:
            print('\nError: Invalid input provided!! Please type LDA or PBE!! \n')
            raise TypeError


    def get_interface_dos(self, interface_dos_path):
        print("\n * Obtaining interface DOS results ")
        vasprun = Vasprun(interface_dos_path+'/vasprun.xml')
        self.total_interface_dos = vasprun.complete_dos
        self.total_interface_structure = vasprun.final_structure
        self.interface_volume = self.total_interface_structure.volume
        self.z_length = self.total_interface_structure.lattice.matrix[2][2]
        self.sites = self.total_interface_structure.sites
        self.kpts = vasprun.actual_kpoints
        self.kpts_weight = vasprun.actual_kpoints_weights
        self.kpts_weight = np.round(np.array(self.kpts_weight)*100).astype(int)
        self.kmesh = np.column_stack((np.array(self.kpts)[:, :2],self.kpts_weight))


    def get_z_bounds(self, name):
        result = {"min_z": None, "max_z": None}
    
        def on_submit():
            try:
                min_val = float(entry_min.get())
                max_val = float(entry_max.get())
    
                if min_val > max_val:
                    messagebox.showerror("Invalid Input", "Minimum z must be less than or equal to maximum z.")
                    return
    
                result["min_z"] = min_val
                result["max_z"] = max_val
                root.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
    
        def on_cancel():
            root.destroy()
    
        root = tk.Tk()
        root.title("Enter z-range to extract atoms from the 1st layer of %s interface"%name)
        root.geometry("320x160")
    
        tk.Label(root, text="Minimum z value (Å):").pack(pady=(10, 0))
        entry_min = tk.Entry(root)
        entry_min.pack()
    
        tk.Label(root, text="Maximum z value (Å):").pack(pady=(10, 0))
        entry_max = tk.Entry(root)
        entry_max.pack()
    
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
    
        tk.Button(btn_frame, text="OK", width=10, command=on_submit).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side=tk.RIGHT, padx=10)
    
        root.mainloop()
    
        return result["min_z"], result["max_z"]

    def get_semi_1st_layer_dos(self):
        print("\n  -. Extracting atoms list & deriving DOS of semiconductor 1st layer from given input ")
        self.layer_order = 1
        self.material_type = 'semiconductor'
        view(AseAtomsAdaptor.get_atoms(self.total_interface_structure))
        min_z, max_z = self.get_z_bounds(self.material_type)
        self.layer_dos_extraction(min_z, max_z)
        self.semi_1st_layer_dos = self.layer_dos
        self.semi_1st_layer_min_z = self.min_z_layer
        self.semi_1st_layer_max_z = self.max_z_layer
        self.semi_1st_layer_thickness = self.layer_thickness
        return

    def get_semi_next_layer_dos(self, min_z_previous_layer, max_z_previous_layer, layer_thickness):
        self.layer_order += 1
        if self.semi_1st_layer_max_z < self.metal_1st_layer_min_z:
            self.z_length_inter = self.metal_1st_layer_max_z - self.semi_1st_layer_min_z
            closest_z = -np.inf
            for i, site in enumerate(self.sites):
                coord = site.coords
                if coord[2] < min_z_previous_layer:
                    if coord[2] > closest_z:
                        closest_z = coord[2]
                        closest_index = i
            self.semi_layer_max_z = closest_z
            target_semi_layer = self.semi_layer_max_z - layer_thickness

        else:
            self.z_length_inter = self.semi_1st_layer_max_z - self.metal_1st_layer_min_z
            closest_z = np.inf
            for i, site in enumerate(self.sites):
                coord = site.coords
                if coord[2] > max_z_previous_layer:
                    if coord[2] < closest_z:
                        closest_z = coord[2]
                        closest_index = i

            self.semi_layer_min_z = closest_z
            target_semi_layer = self.semi_layer_min_z + layer_thickness

        diff = np.inf
        for i,site in enumerate(self.sites):
            coord = site.coords
            if abs(coord[2] - target_semi_layer) < diff:
                diff = abs(coord[2] - target_semi_layer)
                closest_z = coord[2]

        if self.semi_1st_layer_max_z < self.metal_1st_layer_min_z:
            self.semi_layer_min_z = closest_z
        else:
            self.semi_layer_max_z = closest_z

        self.semi_layer_thickness = self.semi_layer_max_z-self.semi_layer_min_z
        self.layer_dos_extraction(self.semi_layer_min_z, self.semi_layer_max_z)
        self.search_interface_vbm()
            
    def search_interface_vbm(self):
        energy_range = np.linspace(-self.original_bandgap, self.original_bandgap, int((2*self.original_bandgap) / 0.01) + 1)
        bandgap_boundary = int((self.original_bandgap) / 0.01)
        DOS_minimum = np.inf
        for i in range(len(energy_range)):
            if energy_range[i]>0:
                break
            lowerbound = energy_range[i]
            upperbound = lowerbound+self.original_bandgap
            DOS_sum = np.sum(self.layer_dos[(self.layer_dos[:, 0] >= lowerbound) & (self.layer_dos[:, 0] <= upperbound), 1])
            if DOS_sum < DOS_minimum:
                DOS_minimum = DOS_sum
                self.vbm = lowerbound
                self.cbm = upperbound

    def scissor_total_1st_layer_dos(self):
        self.semi_1st_layer_dos[:, 0] -= self.vbm
        self.metal_1st_layer_dos[:, 0] -= self.vbm
        self.scissored_semi_1st_layer_dos = copy.deepcopy(self.semi_1st_layer_dos)
        for i in range(len(self.semi_1st_layer_dos)):
            energy = self.semi_1st_layer_dos[i][0]
            if 0 <= energy <= self.original_bandgap:
                scissored_energy = energy * self.corrected_bandgap / self.original_bandgap
                self.scissored_semi_1st_layer_dos[i][0] = scissored_energy
            elif energy > self.original_bandgap:
                self.scissored_semi_1st_layer_dos[i][0] = self.scissored_semi_1st_layer_dos[i-1][0] + self.semi_1st_layer_dos[i][0] - self.semi_1st_layer_dos[i-1][0]
        
        SC_E = self.scissored_semi_1st_layer_dos[:, 0]
        SC_DOS = self.scissored_semi_1st_layer_dos[:, 1]

        M_E = self.metal_1st_layer_dos[:, 0]
        M_DOS = self.metal_1st_layer_dos[:, 1]
        interp_SC = interp1d(SC_E, SC_DOS, kind='cubic', fill_value='extrapolate')
        interp_M = interp1d(M_E, M_DOS, kind='cubic', fill_value='extrapolate')
        new_E = np.linspace(min(SC_E[0],M_E[0]), max(SC_E[-1],M_E[-1]), len(M_E))
        SC_interp_DOS = interp_SC(new_E)
        SC_interp_DOS[SC_interp_DOS < 0] = 0
        M_interp_DOS = interp_M(new_E)
        M_interp_DOS[M_interp_DOS < 0] = 0
        self.total_1st_layer_dos = np.column_stack((new_E, SC_interp_DOS, M_interp_DOS))
        self.efermi = self.original_bandgap + self.vbm

    def get_metal_1st_layer_dos(self):
        print("\n  -. Extracting atoms list & deriving DOS of metal 1st layer from given input ")
        self.material_type = 'metal'
        #view(AseAtomsAdaptor.get_atoms(self.total_interface_structure))
        min_z, max_z = self.get_z_bounds(self.material_type)
        self.layer_dos_extraction(min_z, max_z)
        self.metal_1st_layer_dos = self.layer_dos
        self.metal_1st_layer_min_z = self.min_z_layer
        self.metal_1st_layer_max_z = self.max_z_layer
        self.metal_1st_layer_thickness = self.layer_thickness
        return

    def layer_dos_extraction(self, minimum, maximum):
        self.index_list = []
        if self.layer_order == 1:
            print('     Species  index    x_coord     y_coord     z_coord')
        min_z_layer= None
        max_z_layer= None
        for i, site in enumerate(self.sites):
            element = site.species_string
            coord = site.coords

            if minimum<=float(coord[2])<=maximum:
                if min_z_layer:
                    if min_z_layer > coord[2]:
                        min_z_layer = coord[2]
                else:
                    min_z_layer = coord[2]
                if max_z_layer:
                    if max_z_layer < coord[2]:
                        max_z_layer = coord[2]
                else:
                    max_z_layer = coord[2]

                coord_str = ['%-10.5f'%(item) for item in coord]
                if self.layer_order == 1:
                    print('     % -7s  %-6d  '%(element, i+1), '  '.join(coord_str))
                self.index_list.append(i)

        self.min_z_layer = min_z_layer
        self.max_z_layer = max_z_layer
        self.layer_thickness = self.max_z_layer - self.min_z_layer

        if self.layer_order == 1:
            print('  --> %d atoms comprising the %s 1st layer'%(len(self.index_list), self.material_type))
        total_ldos_list = []
        for index in self.index_list:
            site = self.total_interface_dos.structure[index]
            ldos = self.total_interface_dos.get_site_dos(site)
            energies = ldos.energies - ldos.efermi
            ldos = ldos.densities[Spin.up]
            total_ldos_list.append(ldos)
        
        total_ldos = np.sum(total_ldos_list, axis=0)
        self.layer_dos = np.column_stack((energies, total_ldos))

    def get_kp_resolved_dos(self, interface_dos_path):
        print("\n  -. Extracting k-point resolved DOS of semicondcutor 2nd layer ")
        vasprun = Vasprun(interface_dos_path+'/vasprun.xml')
        vaspkit_path = self.read_path_file('vaspkit')
        self.kp_dos = []
        for kpt in range(len(self.kpts)):
            process = subprocess.Popen([vaspkit_path], cwd=interface_dos_path,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            commands = [
            '119\n',
            '1\n',
            '-20 20 2001 0.01\n',
            '3\n',
            '%f %f\n'%(self.semi_layer_min_z, self.semi_layer_max_z),
            '2\n',
            '%i\n'%(kpt+1)]
            for command in commands:
                process.stdin.write(command)
                process.stdin.flush()
            
            try:
                output, error = process.communicate()
                data = np.loadtxt(interface_dos_path+'/PDOS_K%i.dat'%(kpt+1), comments='#')
            except:
                print("\nError: DOSCAR, EIGENVAL, PROCAR files needed for vaspkit !!")
                raise SystemExit()

            os.remove(interface_dos_path+'/PDOS_K%i.dat'%(kpt+1))
            os.remove(interface_dos_path+'/FERMI_ENERGY')
            os.remove(interface_dos_path+'/SELECTED_ATOMS_LIST')
            energy = data[:, 0]
            dos = data[:, 10]
            self.kp_dos.append(dos)

        self.kp_energy = energy
        self.kp_layer_order = self.layer_order

    def scissor_kp_dos(self):
        self.kp_energy -= self.vbm
        self.scissored_kp_dos = {}
        for kp in range(len(self.kp_dos)):
            self.scissored_kp_dos[kp] = []
            for energy, density in zip(self.kp_energy, self.kp_dos[kp]):
                if 0 <= energy <= self.original_bandgap:
                    energy = energy * self.corrected_bandgap / self.original_bandgap
                elif energy > self.original_bandgap:
                    energy = (energy-self.original_bandgap) + self.corrected_bandgap
                self.scissored_kp_dos[kp].append(np.array([energy, density]))
    
    def write_vasp_result(self):
        print('\n\n >>> Writing VASP results in "%s" \n'%os.path.abspath(self.result_path))
        np.savetxt("%s/dos_bulk.dat"%self.result_path, self.scissored_bulk_dos, fmt='%.4f', delimiter=' ', comments='', header='%i'%len(self.scissored_bulk_dos))
        np.savetxt("%s/DOStot.dat"%self.result_path, self.total_1st_layer_dos,  fmt='%.4f', delimiter=' ', comments='', header='%i'%len(self.total_1st_layer_dos))
        np.savetxt("%s/k_mesh.dat"%self.result_path, self.kmesh, fmt='%.14f %.14f %i', delimiter=' ', comments='', header='%i'%len(self.kpts))

        with open("%s/kpdos_int_%i.dat"%(self.result_path, self.kp_layer_order), "w") as kpdos_file:
            kpdos_file.write("%i %i\n"%(len(self.kp_dos), len(self.kp_energy)))
            for kp in range(len(self.scissored_kp_dos)):
                kpdos_file.write("  %i\n"%(kp+1))
                np.savetxt(kpdos_file, self.scissored_kp_dos[kp], fmt='    %.3f %.4f', delimiter='  ')

        input_string = "%-21.4f   ! underestimated bandgap [eV] of the bulk semiconductor\n" \
        "%-21.4f   ! corrected bandgap [eV] of the bulk semiconductor\n" \
        "%-21.4f   ! er, dielectric constant of the bulk semiconductor\n"\
        "%-21.4f   ! V_DSC [Å], volume of bulk cell\n" \
        "%-21.4f   ! V_D0 [Å], volume of interface cell\n" \
        "%-21.4f   ! Lz [Å], length of interface cell along z direction\n" \
        "%-21.4f   ! Lz_int [Å], thickness of interfacial region along z direction\n" \
        "%-21.4f   ! z2 [Å], depth of 2nd layer of semiconductor from metal\n" \
        "%-21.4f   ! (CBM - Fermi level) [eV] of the interface\n" %(self.original_bandgap, self.corrected_bandgap, self.er, self.bulk_volume, self.interface_volume, self.z_length, self.z_length_inter, self.semi_2nd_layer_depth, self.efermi)
        with open("%s/vasp_result.dat"%self.result_path,"w") as input_dat:
            input_dat.write(input_string)

    def get_qe_scf(self):
        print("\n * Obtaining QE SCF results ")
        tree = ET.parse(self.surface_path+'/QE/SCF/scf.xml')
        root= tree.getroot()

        self.real_lattice = []
        self.reciprocal_lattice = []
        for child_1 in root:
            if child_1.tag == 'output':
                for child_2 in child_1:
                    if child_2.tag == 'atomic_structure':
                        self.alat = float(child_2.attrib['alat'])
                        for child_3 in child_2:
                            if child_3.tag == 'cell':
                                for vec in child_3:
                                    self.real_lattice.append(np.array(list(map(float, vec.text.split()))))
                    if child_2.tag == 'basis_set':
                        for child_3 in child_2:
                            if child_3.tag == 'reciprocal_lattice':
                                for vec in child_3:
                                    self.reciprocal_lattice.append(np.array(list(map(float, vec.text.split()))))

        self.alat = self.alat * Bohr / Angstrom
        self.reciprocal_param = 2 * np.pi/self.alat
        self.real_vector = np.array(self.real_lattice) * Bohr / Angstrom
        self.cz = self.real_vector[2][2]
        self.reciprocal_vector = np.array(self.reciprocal_lattice)
        self.cbs_volume = abs(np.dot(self.real_vector[0], np.cross(self.real_vector[1], self.real_vector[2])))

    def get_qe_cbs(self):
        print("\n * Obtaining QE CBS results ")
        self.cbs_data = {}
        current_kpoint = None
        current_energy = None
        min_abs_imk = None
        with open(self.surface_path+'/QE/CBS/cbs.im') as cbs_file:
            lines = cbs_file.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('# k-point'):
                    current_kpoint = line.split()[-1]
                    self.cbs_data[current_kpoint]=[]
                elif current_kpoint is not None and line:
                    values = list(map(float, line.split()))
                    imk = values[0]

                    #if imk > 0:
                    #   abs_imk = imk - self.reciprocal_param/2
                    if imk < 0:
                        abs_imk = abs(imk)  

                    if values[1] != current_energy:
                        current_energy = values[1]
                        min_abs_imk = abs_imk
                        self.cbs_data[current_kpoint] = [np.array([current_energy, min_abs_imk])] + self.cbs_data[current_kpoint]
                    else:
                        if abs_imk < min_abs_imk:
                            min_abs_imk = abs_imk
                            original_imk = values[0]
                            self.cbs_data[current_kpoint][0] = np.array([current_energy, min_abs_imk])

        self.num_kpt = len(self.cbs_data.keys())
        self.num_energy = len(self.cbs_data[list(self.cbs_data.keys())[0]])

    def scissor_cbs(self):
        self.original_bandgap = np.genfromtxt(self.result_path + '/vasp_result.dat', comments='!', usecols=0)[0]
        self.corrected_bandgap = np.genfromtxt(self.result_path + '/vasp_result.dat', comments='!', usecols=0)[1]
        self.scissored_cbs_data = copy.deepcopy(self.cbs_data)
        for kp in self.cbs_data:
            for i in range(len(self.cbs_data[kp])):
                energy = self.cbs_data[kp][i][0]
                if 0 <= energy <= self.original_bandgap:
                    self.scissored_cbs_data[kp][i][0] = energy * self.corrected_bandgap / self.original_bandgap
                elif energy > self.original_bandgap:
                    self.scissored_cbs_data[kp][i][0] = (energy-self.original_bandgap) + self.corrected_bandgap

    def write_qe_result(self):
        print('\n\n >>> Writing QE results in "%s" \n'%os.path.abspath(self.result_path))
        if os.path.exists("%s/vasp_result.dat"%self.result_path):
            b1 = ' '.join(format(x, '.4f') for x in self.reciprocal_vector[0])
            b2 = ' '.join(format(x, '.4f') for x in self.reciprocal_vector[1])
            b3 = ' '.join(format(x, '.4f') for x in self.reciprocal_vector[2])
            
            input_string = "%-21.4f   ! alat, the length [Å] of the cell along the z direction for CBS calc.\n" \
            "%s    ! b1, reciprocal vector for CBS calc.\n" \
            "%s    ! b2, reciprocal vector CBS calc.\n" \
            "%s    ! b3, reciprocal vector for CBS calc\n" \
            "%-21.4f   ! cz, the length [Å] of the cell along the z direction for CBS calc.\n"  %(self.alat, b1, b2, b3, self.cz)
            with open("%s/qe_result.dat"%self.result_path,"w") as input_dat:
                input_dat.write(input_string)
        else:
            print("Please parse the VASP result first!")

        with open("%s/cbs.dat"%self.result_path, "w") as cbs_file:
            cbs_file.write("%i %i\n"%(self.num_kpt, self.num_energy))
            for kp in self.scissored_cbs_data:
                header = "   %s\n"%kp
                cbs_file.write(header)
                np.savetxt(cbs_file, self.scissored_cbs_data[kp], fmt='    %.6f %.6f', delimiter='  ')