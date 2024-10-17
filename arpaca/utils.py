import os
import sys
import shutil
import numpy as np
# from .amorphous import genInput

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
                  file_out='force.dat'):
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
            
            
            
            
            




                

