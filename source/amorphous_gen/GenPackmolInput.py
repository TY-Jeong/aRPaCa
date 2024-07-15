## USAGE: python generate_packmol_input.py {density; g/cm3} {N_cation} {N_O}
import sys

if len(sys.argv) != 4:
    sys.exit("ERROR! CHECK YOUR ARGUMENTS! DENSITY, N_Hf, N_O")
else:
    d = float(sys.argv[1])
    N_Hf = int(sys.argv[2])
    N_O = int(sys.argv[3])

u  =  1.66053906660E-27 # kg
m_Hf = 178.490 * u    # kg, grep POMASS POTCAR_Ta
m_O = 15.999 * u        # kg, grep POMASS POTCAR_O

m = (N_Hf*m_Hf + N_O*m_O) * 1000    	# gram
alat = (m/d)**(1/3) * 10**8         	# Angstrom
alat_pbc = alat * 0.98 - 2		# For PBC

with open("./packmol.inp",'w') as f:
    f.write("# d=%.2fg/cm^3, N_Hf= %d, N_O= %d, HfO%.2f\n"%(d, N_Hf, N_O, N_O/N_Hf))
    f.write("tolerance 2.0\n")
    f.write("filetype pdb\n")
    f.write("output packmol_output.pdb\n\n")
    f.write("structure Hf.pdb\n")       # pdb file for Ta atom
    f.write("  number %d \n"%N_Hf)
    f.write("  inside cube 0. 0. 0. %.6f \n"%alat_pbc)
    f.write("end structure\n\n")
    f.write("structure O.pdb\n")        # pdb file for O atom
    f.write("  number %d \n"%N_O)
    f.write("  inside cube 0. 0. 0. %.6f \n"%alat_pbc)
    f.write("end structure")

print("Lattice parameter (cubic) = %.7f"%alat)
print("DONE!")
