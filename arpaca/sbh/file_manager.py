import os
import copy
import numpy as np
import warnings
import subprocess
import xml.etree.ElementTree as ET

import pprint

from ase.io import read
from ase.units import Bohr, Angstrom
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.pwscf import PWInput
from pymatgen.core import Composition
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor

from scipy.interpolate import interp1d

from sbh.generate_input import *

warnings.filterwarnings("ignore", module='pymatgen')

class BulkSet(VASP_Input):
	def __init__(self, bulk, potcar = 'pbe', ncore = 4, charge = 0):
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
		self.is_semi = False

	def semi_setup(self):
		print('\n&&&&& Configure additional calculation set for semiconductor %s &&&&&\n'%self.formula)
		self.chg_setup()
		self.dos_setup()
		self.dielectric_setup()
		self.corrected_setup()
		self.is_semi = True

	def check_object_type(self, obj):
		if isinstance(obj, GenBulk):
			self.structure = obj.bulk
			self.root_directory = obj.working_directory
		elif isinstance(obj, str):
			    try:
			        self.structure = read(obj)
			        self.project_directory = os.getcwd() + '/Project'
			        self.make_directory(self.project_directory)
			        self.root_directory = self.project_directory + '/Bulk'
			        self.make_directory(self.root_directory)
			    except:
			        print('Check your bulk POSCAR file path')
			        raise FileNotFoundError
		else:
			raise TypeError

		self.formula = Composition(self.structure.get_chemical_formula()).reduced_formula
		self.working_directory = self.root_directory + '/%s'%self.formula
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

	def runfile_setup(self,runfile_name='run.sh', nodes='2', processors='16', queue_name='Null', scheduler=None):
		self.read_path_vasp()

		for dirpath, dirnames, filenames in os.walk(self.working_directory):
			if 'POSCAR' in filenames:
				self.make_runfile(dirpath+'/'+runfile_name, nodes, processors, queue_name, scheduler=scheduler)

class SurfaceSet(QE_Input):
	def __init__(self, surface):
		super().__init__()
		print('\n\n############### Initialize SurfaceSet to make calculation sets for complex band structure ###############\n')
		self.check_object_type(surface)
		self.QE_directory = self.working_directory+'/QE'
		self.make_directory(self.QE_directory)
		self.make_directory(self.QE_directory+'/pseudos')
		return

	def check_object_type(self, obj):
		if isinstance(obj, GenSurface):
			self.structure = obj.cbs_slab
			self.working_directory = obj.working_directory
		elif isinstance(obj, str):
			    try:
			        self.structure = read(obj)
			        self.working_directory = obj.replace('/cbs_surface.vasp','')
			        self.make_directory(self.working_directory)
			    except:
			        print('Check your surface POSCAR file path')
			        raise FileNotFoundError
		else:
			raise TypeError

	def scf_setup(self, pseudopotentials, custom_option=None):
		self.formula = Composition(self.structure.get_chemical_formula()).reduced_formula
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
			pseudo = pseudopotentials
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
		"	prefixl = scf\n" \
		"	band_file = cbs\n" \
		"	ikind = 0\n" \
		"	energy0 = %f\n" \
		"	denergy = -%f\n" \
		"	ewind = 4.0\n" \
		"	epsproj = 1.d-8\n" \
		"	nz1 = 11\n" \
		" /\n"%(max_energy, denergy)
		pwcond_str += kmesh
		pwcond_str += '    %i'%(energy_range/denergy)
		
		with open(self.QE_CBS_directory+"/qe.in","w") as pwcond_input:
			pwcond_input.write(pwcond_str)

		if not os.path.islink(self.QE_CBS_directory+'/scf.save'):
			if not os.path.exists(self.QE_CBS_directory+'/scf.save'):
				os.symlink('../SCF/scf.save', self.QE_CBS_directory+'/scf.save')

	def runfile_setup(self,runfile_name='run.sh', nodes='1', processors='32', queue_name='Null', scheduler=None):
		self.read_path_pw()
		self.make_runfile(self.QE_SCF_directory+'/'+runfile_name, nodes, processors, queue_name, scheduler)
		self.read_path_pwcond()
		self.make_runfile(self.QE_CBS_directory+'/'+runfile_name, nodes, processors, queue_name, scheduler)

class InterfaceSet(VASP_Input):
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
		if isinstance(obj, GenInterface):
			self.structure = obj.interface
			self.working_directory = obj.working_directory
		elif isinstance(obj, str):
			    try:
			        self.structure = read(obj)
			        self.project_directory = os.getcwd() + '/Project'
			        self.make_directory(self.project_directory)
			        self.working_directory = self.project_directory + '/Interface'
			        self.make_directory(self.working_directory)
			    except:
			        print('Check your interface POSCAR file path')
			        raise FileNotFoundError
		else:
			raise TypeError

	def relax_setup(self, custom_incar_params=None):
		self.relax_directory = self.working_directory+'/Relax'
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
		self.chg_directory = self.working_directory+'/Chg'
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
		self.dos_directory = self.working_directory+'/DOS'
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

	def runfile_setup(self,runfile_name='run.sh', nodes='4', processors='16', queue_name='Null', scheduler=None):
		self.read_path_vasp()
		for dirpath, dirnames, filenames in os.walk(self.working_directory):
			if 'POSCAR' in filenames:
				self.make_runfile(dirpath+'/'+runfile_name, nodes, processors, queue_name, scheduler=scheduler)


class GetSBHResult(BasicTool):
	def __init__(self, calc_project = None):
		self.check_object_type(calc_project)
		super().__init__()

	def get_vasp_result(self):
		print('\n\n############### Retrieving bulk calculation results ###############')
		self.get_bulk_dos(self.bulk_dos_path)
		self.get_bulk_corrected(self.bulk_corrected_path)
		self.scissor_bulk_dos()
		self.get_bulk_dielec(self.bulk_dielec_path)
		print('\n\n############### Retrieving interface calculation results ###############')
		self.get_interface_dos(self.interface_dos_path)
		self.get_semi_1st_layer_dos(15.9,18.5)
		self.get_metal_1st_layer_dos(12.7,15.3)
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

	def qe_setup(self, pseudopotentials, scheduler = None):
		self.surface_calc_set = SurfaceSet(self.surface_path+'/cbs_surface.vasp')
		self.surface_calc_set.scf_setup(pseudopotentials)
		self.original_bandgap = np.genfromtxt(self.result_path + '/vasp_result.dat', comments='!', usecols=0)[0]
		with open(self.result_path+'/k_mesh.dat','r') as kmesh_dat:
			kmesh = kmesh_dat.read() 
		self.surface_calc_set.cbs_setup(self.original_bandgap, kmesh)
		self.surface_calc_set.runfile_setup(scheduler = scheduler)

	def get_qe_result(self):
		print('\n\n############### Retrieving QE calculation results ###############')
		self.get_qe_scf()
		self.get_qe_cbs()
		self.scissor_cbs()
		self.write_qe_result()
		return

	def check_object_type(self, obj):
		if obj == None:
			self.project_directory = os.getcwd()
		elif isinstance(obj, str):
			self.project_directory = obj
		else:
			raise TypeError

		try:
			required_bulk_dirs = ['DOS', 'Correction', 'Dielec', 'Chg', 'Relax']
			bulk_path = self.path_checker(self.project_directory+'/Bulk', required_bulk_dirs)
			self.bulk_dos_path = bulk_path + '/DOS'
			self.bulk_corrected_path = bulk_path + '/Correction'
			self.bulk_dielec_path = bulk_path + '/Dielec'
			self.interface_dos_path = self.project_directory +'/Interface/DOS'
			self.surface_path = self.project_directory + '/Surface'
			self.result_path = self.project_directory + '/Result'
			self.make_directory(self.result_path)
		except:
			raise FileNotFoundError(f"Can't find Project directory")

		# elif isinstance(obj, BulkSet):
		# 	self.bulk_dos_path = obj.dos_directory
		# 	self.bulk_corrected_path = obj.corrected_directory
		# 	self.bulk_dielec_path = obj.dielec_directory
		# 	pass

		# elif isinstance(obj, InterfaceSet):
		# 	self.interface_dos_path = obj.dos_directory


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

		pbe_er = get_dielec(bulk_dielec_path+'/PBE')
		lda_er = get_dielec(bulk_dielec_path+'/LDA')

		print('  -. Dielectric constant calculated with PBE: %f'%pbe_er)
		print('  -. Dielectric constant calculated with LDA: %f\n'%lda_er)
		choice = input("  Choose either 'pbe' or 'lda' for dielectric constant with PBE or LDA: ").strip().lower()

		if choice == 'pbe':
			self.er = pbe_er
		elif choice == 'lda':
			self.er = lda_er
		else:
			print('\nError: Invalid input provided!! Please type lda or pbe!! \n')
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

	def get_semi_1st_layer_dos(self, min_z, max_z):
		print("\n  -. Extracting atoms list & deriving DOS of semiconductor 1st layer from given input ")
		self.layer_order = 1
		self.material_type = 'semiconductor'
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

	def get_metal_1st_layer_dos(self, min_z, max_z):
		print("\n  -. Extracting atoms list & deriving DOS of metal 1st layer from given input ")
		self.material_type = 'metal'
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
					#	abs_imk = imk - self.reciprocal_param/2
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
