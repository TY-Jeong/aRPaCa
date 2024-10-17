import sys
import subprocess
import os
import warnings
import numpy as np
from collections import Counter
from ase.io import read, write
from ase.build import surface, make_supercell, stack
from ase import Atoms
from ase.visualize import view
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore", module='pymatgen')

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

    def make_runfile(self, runfile_name, nodes, processors, queue_name, scheduler=None):
        if scheduler == None:
            self.check_scheduler()
        elif (scheduler=='pbs') or (scheduler=='slurm') or (scheduler=='bash') :
            self.scheduler = scheduler
        else:
            print('Error: Invalid scheduler type !!!')
            raise TypeError

        with open(runfile_name,'w') as runfile:
            runfile.write("#!/bin/sh\n")
            if self.scheduler == 'pbs':
                runfile.write(f"#PBS -o sys_mesg.log -N SBH\n")
                runfile.write("#PBS -j oe\n")
                runfile.write(f"#PBS -l nodes={nodes}:ppn={processors}\n")
                runfile.write(f"#PBS -q {queue_name}\n")
                runfile.write("\n")
                runfile.write("NUMBER=`wc -l < $PBS_NODEFILE`\n")
                runfile.write("cd $PBS_O_WORKDIR\n")
                runfile.write("\nmpirun -np $NUMBER %s > stdout.dat\n"%(self.exe_path))
            elif self.scheduler == 'slurm':
                runfile.write("#SBATCH --output=sys_mesg.log\n")
                runfile.write("#SBATCH --job-name=SBH\n")
                runfile.write(f"#SBATCH --ntasks={processors}\n")
                runfile.write(f"#SBATCH --nodes={nodes}\n")
                runfile.write(f"#SBATCH --partition={queue_name}\n")
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




class VASP_Input(BasicTool):
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


class GenBulk(BasicTool):
    def __init__(self,chemsys_formula_mpids = None):
        super().__init__()
        self.API_KEY = self.read_path_file('mp_api')
        self.root_directory = os.getcwd()
        self.iter = False
        self.working_directory = self.root_directory + '/Project/Bulk'

        if chemsys_formula_mpids != None:
            self.make_directory('Project')
            self.make_directory(self.working_directory)
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

    def mpid_poscar_write(self,mpid):
        with MPRester(self.API_KEY, mute_progress_bars=True) as mpr:
            print('\n * Searching material with %s'%mpid)
            entries = mpr.get_entries(mpid)
            formula = entries[0].composition.formula
            formula = Composition(formula).reduced_formula
            self.structure_directory = self.working_directory + '/%s'%formula

            self.make_directory(self.structure_directory)

            material = mpr.get_structure_by_material_id(mpid, conventional_unit_cell = True)
            space_group = material.get_space_group_info()
            space_group_symbol = space_group[0]
            space_group_number = space_group[1]

            if not self.iter:
                with open(self.working_directory+'/%s_MP_data.txt'%formula,'w') as MP_data:
                    MP_data.write('MP_id       Space_group_symbol  Space_group_number\n')
                    MP_data.write('-'*50+'\n')
                    MP_data.write('%-10s  %-18s  %-18s\n'%(mpid,space_group_symbol,space_group_number))

            if self.iter:
                with open(self.working_directory+'/%s_MP_data.txt'%formula,'a') as MP_data:
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
                print('\nSetting bulk structure with %s\n'%(file_path))
                bulk = read(matched_files[0])
                return bulk

            except:
                print('Check your file_name or mpid')
                raise FileNotFoundError

        try:
            file_name_mpid = int(file_name_mpid)
            self.bulk = find_files_with_keyword(self, self.working_directory, 'POSCAR_mp-%s'%file_name_mpid)
        
        except:
            if os.path.isfile(file_name_mpid):
                self.bulk = read(file_name_mpid)

            else:
                print('Check your file_name or mpid')
                raise FileNotFoundError
        
class GenSurface(VASP_Input):
    def __init__(self, miller_index, infile, layer_num=5, vacuum=15):
        # self.check_object_type(structure)
        self.structure = read(infile)
        self.infile = infile
        self.miller_index = miller_index
        self.layer_num = layer_num
        self.vacuum = vacuum
        formula = self.structure.get_chemical_formula()
        formula = Composition(formula)
        self.formula = formula.reduced_formula
        print('\n\n############### Initialize GenSurface to make slab POSCAR for %s ###############\n'%self.formula)
        self.root_directory = os.getcwd()
        self.make_directory('Project')
        self.project_directory = self.root_directory + '/Project'
        self.make_directory('Project/Surface')
        self.working_directory = self.project_directory + '/Surface'
        self.slab_maker()

    # def check_object_type(self, obj):
    #     # Check GenSurface class
    #     if isinstance(obj, GenB):
    #         slab = obj.slab
    #     # Check slab POSCAR file name
    #     elif isinstance(obj, str):
    #         try:
    #             slab = read(obj)
    #         except:
    #             print('Check your POSCAR file')
    #             raise FileNotFoundError
    #     else:
    #         raise TypeError

    def calculate_thickness(self):
        self.z_coords = self.slab.get_positions()[:, 2]  
        max_z = np.max(self.z_coords)
        min_z = np.min(self.z_coords)
        self.thickness = max_z - min_z


    def cbs_surface_maker(self, layer_num):
        self.cbs_slab = surface(self.structure, self.miller_index, layers=layer_num, periodic=True) # layers=self.layer_num, 
        self.make_poscar('%s/cbs_surface.vasp'%(self.working_directory), self.cbs_slab)
        print(' * Generating slab POSCAR for cbs calculation in "%s/cbs_surface.vasp"'%(self.working_directory))


    def slab_maker(self):
        self.slab = surface(self.structure, self.miller_index, layers=self.layer_num, vacuum=self.vacuum, periodic=True)
        self.calculate_thickness()
        print(' * Generating slab POSCAR for %s in "%s/%s_slab-%s.vasp"'%(self.formula, self.working_directory , self.formula, ''.join(map(str, self.miller_index))))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.working_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. %s %s slab thickness is %.2f Å\n'%(self.formula, ''.join(map(str, self.miller_index)),self.thickness))

    def slice_slab_direct(self, z_min, z_max):
        print(os.get_cwd())
        self.z_coords_direct = self.slab.get_scaled_positions()[:, 2]  
        mask = (self.z_coords_direct >= z_min) & (self.z_coords_direct <= z_max)
        self.slab = self.slab[mask]
        self.calculate_thickness()
        print(' * Slicing %s slab from z-coordinate from %.2f to %.2f'%(self.formula, z_min, z_max))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.working_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. Sliced %s slab thickness is %.2f Å\n'%(self.formula,self.thickness))

    def slice_slab_cartesian(self, z_min, z_max):
        self.z_coords = self.slab.get_positions()[:, 2]
        mask = (self.z_coords >= z_min) & (self.z_coords <= z_max)
        self.slab = self.slab[mask]
        self.calculate_thickness()
        print(' * Slicing %s slab from %.2fÅ to %.2fÅ'%(self.formula, z_min, z_max))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.working_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)
        print('  -. Slice %s slab thickness is %.2f Å\n'%(self.formula,self.thickness))

    def xy_shift_direct(self, xy):
        x, y = xy
        positions_direct = self.slab.get_scaled_positions()
        positions_direct += [x, y, 0]
        self.slab.set_scaled_positions(positions_direct)
        print(' * Shifting %s slab by (%f %f 0)\n'%(self.formula, x, y))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.working_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)

    def xy_shift_cartesian(self, xy):
        x, y = xy
        self.slab.translate([x, y ,0])
        print(' * Shifting %s slab by (%f %f 0)\n'%(self.formula, x, y))
        self.make_poscar('%s/%s_slab-%s.vasp'%(self.working_directory,self.formula, ''.join(map(str, self.miller_index))), self.slab)

class GenInterface(VASP_Input):
    def __init__(self, substrate_surface, film_surface):
        """
        substrate_surface: str(slab POSCAR filename) | GenSurface
        film_surface: str(slab POSCAR filename) | GenSurface
        """
        self.substrate = self.check_object_type(substrate_surface)
        self.substrate_formula = self.formula
        self.film = self.check_object_type(film_surface)
        self.film_formula = self.formula
        print('\n\n############### Initialize GenInterface to make interface POSCAR for %s/%s ###############\n'%(self.substrate_formula,self.film_formula))
        
        self.root_directory = os.getcwd()
        self.project_directory = self.root_directory + '/Project'
        self.make_directory(self.project_directory)

        self.working_directory = self.project_directory + '/Interface'
        self.make_directory(self.working_directory)

        self.editing_directory = self.working_directory + '/Edit'
        self.make_directory(self.editing_directory)


    def check_object_type(self, obj):
        if isinstance(obj, GenSurface):
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

    def interface_maker(self, spacing = 2, vacuum = 10):
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
            if self.vacuum < spacing:
                self.vacuum = spacing
            combined_cell = self.lattice_z_transform(combined_cell, [0, 0, self.vacuum + max_z])
            combined_pbc = self.substrate_supercell.get_pbc()
            self.interface = Atoms(symbols=combined_symbols, positions=combined_positions, cell=combined_cell, pbc=combined_pbc)
            self.make_poscar("%s/interface.vasp"%self.working_directory, self.interface)
            view(self.interface)

        self.spacing = spacing
        self.vacuum = vacuum
        align_substrate_z0(self)
        align_film_on_substrate(self)
        print(' * Generating interface POSCAR in "%s/interface.vasp"\n'%(self.working_directory))
        create_interface(self)
        return

    def film_xy_shift_direct(self, xy):
        x, y = xy
        print(' * Shifting interface by (%f %f 0)'%(x, y))
        try:
            positions_direct = self.film_lattice_matched.get_scaled_positions()
            positions_direct += [x, y, 0]
            self.film_lattice_matched.set_scaled_positions(positions_direct)
        except:
            positions_direct = self.film
            positions_direct += [x, y, 0]
            self.film.set_scaled_positions(positions_direct)


    def film_xy_shift_direct(self, xy):
        x, y = xy
        try:
            positions_direct = self.film_lattice_matched.get_scaled_positions()
            positions_direct += [x, y, 0]
            self.film_lattice_matched.set_scaled_positions(positions_direct)
        except:
            positions_direct = self.film
            positions_direct += [x, y, 0]
            self.film.set_scaled_positions(positions_direct)

    def auto_lattice_matching(self):
        print(' * Automatically matching lattice\n')
        self.minimize_lattice_mismatch()
        self.lattice_matching()

    def manual_lattice_matching(self,substrate_scaling_matrix=[1, 1], film_scaling_matrix=[1, 1]):
        print(' * Manually matching lattice')
        def transform_matrix(xy):
            x, y = xy
            return [[x, 0, 0],  
                    [0, y, 0],  
                    [0, 0, 1]]

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
