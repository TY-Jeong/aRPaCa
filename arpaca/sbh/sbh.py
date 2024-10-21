import math
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import sys

from arpaca.utils import *


class ReadResult(BasicTool):
    def __init__(self, result_path):
        print("\n\n############### Reading first-principles calculation results ###############\n")
        required_files = ['dos_bulk.dat', 'cbs.dat', 'DOStot.dat', 'vasp_result.dat', 'qe_result.dat', 'k_mesh.dat', 'kpdos_int_2.dat']
        self.result_path = self.path_checker(result_path, required_files)
        self.Lsc = 1.0e8
        self.Sig_gate = 0.0
        self.read_vasp_result()
        self.calc_pol()
        self.read_qe_result()
        self.Surface = self.V_D0 / self.Lz  # surface of the cell
        self.calc_reciprocal_param()
        self.read_k_mesh()
        self.print_input_parameters()
        self.read_CBS_data()
        self.calc_CBS_limits()
        self.set_limits_DOS() 
        self.read_DOS()

    def read_vasp_result(self):
        self.vasp_dat = np.genfromtxt(self.result_path + '/vasp_result.dat', comments='!', usecols=0)
        self.gap = self.vasp_dat[1]
        self.er = self.vasp_dat[2]
        self.V_DSC = self.vasp_dat[3]
        self.V_D0 = self.vasp_dat[4]
        self.Lz = self.vasp_dat[5]
        self.Lz_int = self.vasp_dat[6]
        self.z2 = self.vasp_dat[7]

        self.EVBM = 0.0
        self.ECBM = self.gap

        self.EFermi_input = self.gap - self.vasp_dat[8]

        if self.vasp_dat[8] < (self.ECBM - self.vasp_dat[8]):
            #self.EFermi2 = self.ECBM
            self.EFermi2 = self.EVBM
        else:
            #self.EFermi2 = self.EVBM
            self.EFermi2 = self.ECBM

    def calc_pol(self):
        e0_SI = 8.8541878128e-12
        C1 = 6.24150975e18
        self.e0 = e0_SI * C1 * 1.0e-10
        self.kappa = self.e0 * (self.er - 1)

    def read_qe_result(self):
        with open(self.result_path+'/qe_result.dat','r') as qe_dat:
            self.alat = float(qe_dat.readline().split()[0])
            self.b1 = list(map(float, qe_dat.readline().split()[:3]))
            self.b2 = list(map(float, qe_dat.readline().split()[:3]))
            self.b3 = list(map(float, qe_dat.readline().split()[:3]))
            self.cz = float(qe_dat.readline().split()[0])
            self.ckA = np.pi* 2.0 / self.cz

    def calc_reciprocal_param(self):
        self.a2p = 2.0 * np.pi / self.alat
        self.b1 = [x * self.a2p for x in self.b1]
        self.b2 = [x * self.a2p for x in self.b2]
        self.b3 = [x * self.a2p for x in self.b3]

    def read_k_mesh(self):
        with open(self.result_path + '/k_mesh.dat', 'r') as file:
            lines = file.readlines()
            self.Nk = int(lines[0].strip())
            self.kp = np.zeros((3, self.Nk))
            self.wk = np.zeros(self.Nk)
            for i in range(1,self.Nk + 1):
                data = lines[i].split()
                self.kp[0, i - 1] = float(data[0])
                self.kp[1, i - 1] = float(data[1])
                self.wk[i - 1] = float(data[2])
        self.sumk = np.sum(self.wk)
        self.kr9 = 100
        for k in range(self.Nk):
            kr = np.sqrt(self.kp[0, k] ** 2 + self.kp[1, k] ** 2)
            if (k + 1) > 1 and kr < self.kr9:
                self.k9 = k + 1
                self.kr9 = kr

    def print_input_parameters(self):    
        print(" length of the cell in the interface calculation (Lz) =", self.Lz, "Å")
        print(" width of the interfacial layer              (Lz_int) =", self.Lz_int, "Å")
        print(" volume of the interfacial layer               (V_D0) =", self.V_D0, "Å^3")
        print(" position of 2nd layer of the interface           (z2) =", self.z2, "Å")
        print(" crystal parameter                             (alat) =", self.alat, "Å")
        print(" reciprocal vectors (in 2*pi/alat)")
        print("   ", self.b1[0], self.b1[1], self.b1[2])
        print("   ", self.b2[0], self.b2[1], self.b2[2])
        print("   ", self.b3[0], self.b3[1], self.b3[2])
        print(" size of the cell in z direction for CBS         (cz) =", self.cz, "Å")
        print(" volume of the cell of the bulk semiconductor (V_DSC) =", self.V_DSC, "Å^3")
        print(" Valence band maximum                          (EVBM) =", self.EVBM, "eV")
        print(" Conduction band minimum                       (ECBM) =", self.ECBM, "eV")
        print(" Surface of interface cell                  (Surface) =", self.Surface, "Å^2")
        print(" dielectric constant of the bulk                 (er) =", self.er)

    def read_CBS_data(self):
        with open(self.result_path+'/cbs.dat', 'r') as f:
            self.rNk, self.Npt1 = map(int, f.readline().split())
            self.ImK1 = np.zeros((self.Npt1, self.Nk))
            self.Ef1 = np.zeros(self.Npt1)
            
            if self.rNk != self.Nk:
                print("ERROR with Nk")
                raise SystemExit()

            for k in range(self.Nk):
                k1 = int(f.readline())
                
                if k1 != k+1:
                    print("ERROR with k1")
                    raise SystemExit()

                for i in range(self.Npt1):
                    self.Ef1[i], self.ImK1[i, k] = map(float, f.readline().split()) #, self.ImKH1[i, k]

    def calc_CBS_limits(self):
        self.Emin1 = self.Ef1[0]
        self.Emax1 = self.Ef1[self.Npt1-1]

    def set_limits_DOS(self): #set limits for DOS integration
        # limits for integration for h
        self.Exxh1 = [self.EVBM - 6.0, self.EVBM - 4.0, self.EVBM - 2.0, self.EVBM - 1.0]
        self.Exxh2 = [self.EVBM - 4.0, self.EVBM - 2.0, self.EVBM - 1.0, self.EVBM]
        # limits for integration for e
        self.Exxe1 = [self.ECBM, self.ECBM + 1.0, self.ECBM + 2.0, self.ECBM + 4.0]
        self.Exxe2 = [self.ECBM + 1.0, self.ECBM + 2.0, self.ECBM + 4.0, self.ECBM + 6.0]
        # set limits for MIGS integration (4 intervals for accurate integration)
        self.Exxm1 = [self.EVBM, 0.10 * (self.EVBM + self.ECBM), 0.25 * (self.EVBM + self.ECBM), 0.99 * (self.EVBM + self.ECBM)]
        self.Exxm2 = [0.10 * (self.EVBM + self.ECBM), 0.25 * (self.EVBM + self.ECBM), 0.99 * (self.EVBM + self.ECBM), self.ECBM]


    def read_DOS(self):
        self.Efi2, self.PDOS2 = self.read_pdos_1(2) # PDOS of 2nd layer of interface
        self.read_pdos_0()
        self.Ef_DOS_M = np.copy(self.Efi2)
        self.DOS_M = np.copy(self.PDOS2)
        self.calc_DOS_Mtot()
        # DOS of bulk SC
        with open(self.result_path+'/dos_bulk.dat', 'r') as f:
            self.N_DOS_SC = int(f.readline().strip())
            self.Ef_DOS_SC = np.zeros(self.N_DOS_SC)
            self.DOS_SC = np.zeros(self.N_DOS_SC)
            for j in range(self.N_DOS_SC):
                self.Ef_DOS_SC[j], self.DOS_SC[j] = map(float, f.readline().split())
                if self.DOS_SC[j] < 0.0:
                    self.DOS_SC[j] = 0.0

    #read PDOS for surface (SC 1st layer + M 1st layer)
    def read_pdos_0(self):
        DOS_data = np.genfromtxt(self.result_path+'/DOStot.dat', skip_header=1)
        self.Efi0 = DOS_data[:,0]
        DOS_SC = DOS_data[:,1]
        DOS_M = DOS_data[:,2] 
        self.DOS0 = DOS_SC + DOS_M
        self.integral_range = np.zeros(2)
        vb_range = (self.Efi0 >= -2) & (self.Efi0 <= 0)
        vb_DOS = DOS_SC[vb_range]
        max_vb_index = np.argmax(vb_DOS)
        self.integral_range[0] = self.Efi0[vb_range][max_vb_index]
        cb_range = (self.Efi0 >= self.ECBM) & (self.Efi0 <= self.ECBM+2)
        cb_DOS = DOS_SC[cb_range]
        max_cb_index = np.argmax(cb_DOS)
        self.integral_range[1] = self.Efi0[cb_range][max_cb_index] - self.ECBM
        self.integral_range = np.max(np.abs(self.integral_range))

    #read PDOS for layer 2
    def read_pdos_1(self, ilayer):
        with open(self.result_path+'/kpdos_int_%s.dat'%ilayer, 'r') as f:
            kr, self.N_DOS_M = map(int, f.readline().split())
            Efi = np.zeros(self.N_DOS_M)
            PDOS = np.zeros((self.N_DOS_M, self.Nk))
            for k in range(self.Nk):
                f.readline()
                for j in range(self.N_DOS_M):
                    try:
                        Efi[j], PDOS[j, k] = list(map(float, f.readline().split()))
                    except:
                        print(k,j)
                        print(f.readline().split())
                        exit(0)
        return Efi, PDOS

    # calculate integrated DOS_M of interfacial layer
    def calc_DOS_Mtot(self):
        self.DOS_Mtot = np.zeros(self.N_DOS_M)
        for j in range(self.N_DOS_M):
            self.DOSx = 0.0
            for k in range(self.Nk):
                self.DOSx += self.PDOS2[j, k] * self.wk[k]
            self.DOS_Mtot[j] = self.DOSx / self.sumk

def spline(x, y):
    function = CubicSpline(x, y)
    return function

def ispline(x, function):
    return function(x)

def integral(Min, Max, function):
    num_points = 1000
    x_arr = np.linspace(Min, Max , num_points)
    dx = (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    y_arr = []
    for x in x_arr:
        y_arr.append(function(x))
    y_arr = np.array(y_arr)
    result = np.trapz(y_arr, x_arr)
    if math.isnan(result):
        result = 0
    return result

def function_plot(Min, Max, Function):
    num_points = 1000
    
    x_values = np.linspace(Min, Max , num_points)  
    y_values = np.array([Function(x) for x in x_values])
    
    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(min(y_values), max(y_values))
    plt.title('Plot of given function')
    plt.grid(True)
    plt.show()
    
def array_plot(x,y,x_lim=[0,10],y_lim=[0,1e-16]):
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(y_lim[0],y_lim[1])
    plt.xlim(x_lim[0],x_lim[1])
    
    plt.title('Plot of given data')
    plt.grid(True)
    plt.savefig('plot.png', dpi=300)
    plt.show()

class ElecSpline(ReadResult):
    def __init__(self, result_path):
        super().__init__(result_path)
        self.spline_start()

    def spline_start(self):
        self.interface_dos_function = spline(self.Efi0, self.DOS0)
        self.bulk_dos_function = spline(self.Ef_DOS_SC, self.DOS_SC)
        # self.kp_idos_function = spline(self.Ef_DOS_M, self.DOS_Mtot) #bspl2t
        # self.second_layer_dos_function = spline(self.Efi2, self.PDOS2) #bspl13
        self.kp_dos_function_list = [spline(self.Ef_DOS_M, self.DOS_M[:, k]) for k in range(self.Nk)]
        self.cbs_function_list = [spline(self.Ef1, self.ImK1[:, k]) for k in range(self.Nk)]

        self.calc_z_mesh()

    def spline_start_2(self, V_eln, po_new):
        self.V_eln = V_eln
        self.po_new = po_new
        self.electric_potential_function = spline(self.Zz, self.V_eln) #bspl4
        self.charge_density_function = spline(self.Zz, self.po_new) #bspl5

    def Vels(self, z):
        Vels = ispline(z, self.electric_potential_function)
        return Vels

    def pos(self, z):
        pos = ispline(z, self.charge_density_function)
        return pos
        
    def calc_z_mesh(self):
        self.Nz = 704
        dp = (np.log10(self.Lsc) + 1.0) / float(self.Nz - 4)
        pz = -1.0
        self.Zz = np.empty(self.Nz)
        for i in range(3, self.Nz):
            self.Zz[i] = 10.0 ** pz
            pz = pz + dp
        self.Zz[0] = 1e-4
        self.Zz[1] = 1e-3
        self.Zz[2] = 1e-2


class ElecCalculator(ElecSpline):
    def __init__(self, result_path, Temperature=300):
        super().__init__(result_path)
        print("\n\n############### Computing electrical properties ###############\n")
        kb = 8.6173430060e-5
        self.kbT = kb * Temperature
        self.Temperature = Temperature
        self.CNL = self.calc_CNL()
        self.calc_zero_EF()
        self.calc_po00()
        print('\n Surface CNL: %f'%self.CNL)
        print(' Bulk fermi level: %f'%self.EFermi_00)
    
    def DOS0s(self, E):
        if E < self.Efi0[0] or E > self.Efi0[self.N_DOS_M-1]:
            DOS0s = 0.0
        else:
            DOS0s = ispline(E, self.interface_dos_function)
        if DOS0s < 0.0:
            DOS0s = 0.0
        return DOS0s

    def F1S(self,E):
        dexpx = np.exp((E - self.EFermi1) / self.kbT)
        return (dexpx / (1.0 + dexpx)) * self.DOS0s(E)

    def F2S(self,E):
        dexpx = np.exp((E - self.EFermi1) / self.kbT)
        return self.DOS0s(E) / (1.0 + dexpx)
    
    def calc_CNL(self):
        self.calc_EFermiS(1e-11)
        self.CNL = self.EFermi1
        return self.CNL
        
    def calc_EFermiS(self,eps):
        a = self.EVBM+0.01
        b = self.ECBM-0.01
        while abs(a - b) > eps:
            self.EFermi1 = (a + b) / 2.0
            self.calc_po00S()
            if abs(self.po00_hS) > abs(self.po00_eS):
                a = (a + b) / 2.0
            else:
                b = (a + b) / 2.0
    
    def calc_po00S(self):
        self.po00_hS = self.poh0S()
        self.po00_eS = self.poe0S()
    
    def poh0S(self):
        eps = 1e-12
        R11 = integral(self.EVBM-self.integral_range, self.ECBM+self.integral_range , self.F1S)
        poh0S = R11 / self.V_D0
        return poh0S
    
    def poe0S(self):
        eps = 1e-12
        R11 = integral(self.EVBM-self.integral_range, self.ECBM+self.integral_range  , self.F2S)
        poe0S = -R11 / self.V_D0
        return poe0S
    
    def calc_zero_EF(self):
        self.L_n_type = False
        self.L_p_type = False
        self.EFermi_00 = 0
        self.calc_EFermi(1e-15)
        
        self.EFermi_00 = self.EFermi1

        if self.EFermi_input > self.EFermi_00: 
            self.L_n_type = True
            self.L_p_type = False
            
        elif self.EFermi_input < self.EFermi_00:
            self.L_n_type = False
            self.L_p_type = True
            
        else:
            self.L_n_type = False
            self.L_p_type = False
    
        self.EFermi1 = self.EFermi_input
    
    def calc_EFermi(self, eps):
        a = self.EVBM
        b = self.ECBM
        while abs(a - b) > eps:
            self.EFermi1 = (a + b) / 2.0
            self.calc_po00()
            if abs(self.po00_h) > abs(self.po00_e):
                a = (a + b) / 2.0
            else:
                b = (a + b) / 2.0
                
    def DOS_SCs(self, E):
        if E < self.Ef_DOS_SC[0] or E > self.Ef_DOS_SC[self.N_DOS_SC - 1]:
            return 0.0
        elif self.EVBM < E < self.ECBM:
            return 0.0
        else:
            return ispline(E, self.bulk_dos_function)
        if DOS_SCs < 0.0:
            return 0.0
    
    def calc_po00(self):
        self.po00_h = self.poh0()
        self.po00_e = self.poe0()
        self.po00 = self.po00_h + self.po00_e
        
        if self.L_n_type:
            self.po00_e0 = self.po00
            self.po00_h0 = 0.0
            
        elif self.L_p_type:
            self.po00_h0 = self.po00
            self.po00_e0 = 0.0
            
        if abs(self.EFermi1 - self.EFermi_00) < 0.01:
            print('*** WARNING ***')
            print('This is too small doping concentration. Are you sure about the input parameters?')
    
    def F1(self, E):
        dexpx = np.exp((E - self.EFermi1) / self.kbT)
        return (dexpx / (1 + dexpx)) * self.DOS_SCs(E)
    
    def F2(self, E):
        return self.DOS_SCs(E) / (1 + np.exp((E - self.EFermi1) / self.kbT))
    
    def poh0(self):
        R = integral(self.Exxh1[0], self.Exxh2[3], self.F1)
        return R / self.V_DSC
    
    def poe0(self):
        R = integral(self.Exxe1[0], self.Exxe2[3], self.F2)

        return -R / self.V_DSC


class SBCalculator(ElecCalculator):
    def __init__(self, result_path, Temperature=300):
        super().__init__(result_path, Temperature=Temperature)
        self.po_h = np.zeros(self.Nz)
        self.po_e = np.zeros(self.Nz)
        self.po_MIGS = np.zeros(self.Nz)
        self.po1 = np.zeros(self.Nz)
        self.iter = 0
        self.alfa = 0
        self.Nitscf = 1
        self.Nitscf0 = 4
        self.Nitscf1 = 3
        self.Nitscf2 = 1
        print("\n\n############### Launching self-consistent iteration cycle ###############\n")
        print(" iter     SBH      -eV(0)   dV/dz(er*e0)  po(0)     delta_po     delta_V")
        self.set_limits_zz1()
        self.set_initial_deltaE()
        for is_ in range(1, self.Nitscf + 1):
            self.Nitscf20 = 1
            self.L_pre = True
            for is0 in range(1, self.Nitscf0 + 1):
                self.min_delta_V = np.inf
                Lsuc1, Lsuc2 = self.calc_look_po_z(is0, self.zz1, self.zz2)
                
                if Lsuc1 and Lsuc2:
                    print(f"\nThe approximate solution is found with pre SCF cycle using %d iterations\n"%(self.iter))
                    break
                elif (not Lsuc1 or not Lsuc2) and is0 == self.Nitscf0:
                    self.print_scf_error(is0, self.zz1, self.zz2, Lsuc1, Lsuc2)
                    return
                self.reset_zz(Lsuc1, Lsuc2)
            self.set_new_limits()
            self.Nitscf20 = self.Nitscf2
            self.L_pre = False
            for is1 in range(1, self.Nitscf1 + 1):
                self.min_delta_V = np.inf
                Lsuc1, Lsuc2 = self.calc_look_po_z(is1, self.zz1, self.zz2)
                if Lsuc1 and Lsuc2:
                    print(f"\nThe accurate solution is found with post SCF cycle using %d iterations\n"%(self.iter))
                    break
                elif (not Lsuc1 or not Lsuc2) and is1 == self.Nitscf1:
                    self.print_scf_error(is1, self.zz1, self.zz2, Lsuc1, Lsuc2)
                    return
                self.reset_zz2(Lsuc1, Lsuc2)
            self.calc_totq()
            self.calc_deltaE()
        self.calc_check_scf()

    def calc_totq(self):
        eps = 1e-14
        pos_list = np.zeros(self.Nz)
        for i in range(self.Nz):
            pos_list[i] = self.pos(self.Zz[i])
        Sig2 = np.trapz(pos_list, self.Zz)
        self.Sig = -Sig2 - self.Sig_gate  
    
    def calc_deltaE(self):
        self.dEf1 = self.dEf
        eps = 1e-6
        if self.L_n_type:
            E1 = 0.0
            E2 = 2
        elif self.L_p_type:
            E1 = -2.0
            E2 = 0.0
        else:
            E1 = 0.0
            E2 = 0.0
        self.dEf = self.zero12(E1, E2, self.dEf, eps)
        if abs(self.dEf - 0.2) <= 0.001:
            print('*** ERROR: calc_deltaE: increase searching range')
            raise SystemExit()
    
    def zero12(self, a, b, za, eps):
        while abs(a - b) > eps:
            za = (a + b) / 2.0
            if self.L_n_type:
                if self.Fsigma(za) > 0.0:
                    b = za
                else:
                    a = za
            elif self.L_p_type:
                if self.Fsigma(za) > 0.0:
                    a = za
                else:
                    b = za
        return za
    
    def Fsigma(self,dE):
        eps = 1e-12
        self.EFermi111 = self.CNL + dE
        if dE > 0.0:
            E1 = self.CNL
            E2 = self.CNL + dE
        elif dE < 0.0:
            E1 = self.CNL + dE
            E2 = self.CNL
        else:
            E1 = self.CNL
            E2 = self.CNL
        SInt = integral(E1, E2, self.F99)
        Nsigma = abs(self.Sig) * self.Surface
        return (SInt - Nsigma)
    
    def F99(self, E):
        dexpx = np.exp((E - self.EFermi111) / self.kbT)
    
        if self.L_n_type:
            result = self.DOS0s(E) / (1.0 + np.exp((E - self.EFermi111) / self.kbT))
        elif self.L_p_type:
            result = self.DOS0s(E) * dexpx / (1.0 + dexpx)
        else:
            result = 0.0 
    
        return result
    
    def reset_zz(self, Lsuc1, Lsuc2):
        if not Lsuc1:
            self.zz1 = 0.1 * self.zz1
        elif not Lsuc2:
            self.zz2 = 10.0 * self.zz2
            
    def reset_zz2(self, Lsuc1, Lsuc2):
        if not Lsuc1:
            self.zz1 = 0.9 * self.zz1
        elif not Lsuc2:
            self.zz2 = 1.1 * self.zz2
            
    def set_new_limits(self):
        self.zz1 = 0.9 * self.za
        self.zz2 = 1.1 * self.za

    def print_scf_error(self,Natmpt, za1, za2, Lsuc1, Lsuc2):
            print("We made %i attempts to find the solution, unfortunately, it is outside of the region"%Natmpt)
            if not Lsuc1:
                print("za < %s A"%za1)
            if not Lsuc2:
                print("za > %s A"%za2)
            
            print("The parameters of the system are:")
            print("Doping concentration is %s cm^-3"%(self.po00 * 1e24))
            print("Fermi level = %s eV"%(self.EFermi1))
            
            if self.L_p_type:
                print("p-type doped semiconductor with Fermi level of %s eV below the charge neutrality level for bulk"%(self.EFermi1 - self.EFermi_00))
            
            if self.L_n_type:
                print("n-type doped semiconductor with Fermi level of %s eV higher the charge neutrality level for bulk"%(self.EFermi1 - self.EFermi_00))

    
        
    def set_limits_zz1(self):
        x = (self.EFermi1 - self.EFermi_00) / (self.ECBM - self.EFermi_00)
        zz = 1.0 / (0.00000001 * x + 0.000001 * x**2 + 0.00001 * x**3 + 0.0010 * x**4)
        self.zz1 = 0.1 * zz
        self.zz2 = 10.0 * zz
    
    def set_initial_deltaE(self):
        if self.L_n_type:
            x = (self.EFermi1 - self.EFermi_00) / (self.ECBM - self.EFermi_00)
            self.dEf = 0.005 * x**2
        elif self.L_p_type:
            x = (self.EFermi1 - self.EFermi_00) / (self.EVBM - self.EFermi_00)
            self.dEf = -0.005 * x**2
        else:
            self.dEf = 0.0
    
    def calc_look_po_z(self, Natmpt, za1, za2):
        eps = 0.0001
        Nitmax = 600
        a, b = za1, za2
        self.Lsuc1, self.Lsuc2 = True, True
        while abs(a - b) > eps:
            self.iter += 1
            
            if self.iter > Nitmax:
                print(" Overcycling")
                return False, False
    
            self.za = (a + b) / 2.0
            self.calc_diff_eV0()

            if self.min_delta_V > self.delta_V:
                self.min_delta_V = self.delta_V

            if self.EFermi1 > self.CNL + self.dEf:
                if self.diffV > 0.0:
                    b = self.za
                else:
                    a = self.za
            else:
                if self.diffV < 0.0:
                    b = self.za
                else:
                    a = self.za

        if abs(self.za - za1) <= 0.01:
            self.Lsuc1 = False
        elif abs(self.za - za2) <= 0.01:
            self.Lsuc2 = False

        if self.min_delta_V < 1e-2:
            self.Lsuc1 = True
            self.Lsuc2 = True
    
        return self.Lsuc1, self.Lsuc2
    
    def calc_diff_eV0(self): 
        self.set_initial_po_V()
        self.spline_start_2(self.V_eln, self.po_new)
        self.learning_rate = 0.02
        if not self.L_pre:
            self.calc_deltaE()
            self.learning_rate = 0.02
        for i in range(1, self.Nitscf20 + 1):
            self.calc_po1()
            self.mixing_po()
            self.calc_elpot()
            self.mixing_V()

        self.diffV = -self.V_eln[0] - (self.EFermi1 - (self.CNL + self.dEf))
        if self.L_n_type:
            self.SBH = abs(-self.V_eln[0]) + self.ECBM - self.EFermi1
        elif self.L_p_type:
            self.SBH = abs(-self.V_eln[0]) + self.EFermi1 - self.EVBM
        

        print(" %-3d  %9.5f  %9.5f  %.5e  %.4e  %.4e  %.4e" %(self.iter,self.SBH, -self.V_eln[0], self.dVn0, self.po_new[0] * 1e24, 
              self.delta_po * 1e24, self.delta_V))
    
    def set_initial_po_V(self):
        a_init = 1.0 / self.za
        V0 = -(self.EFermi1 - self.CNL)
        self.po_new = np.zeros(self.Nz)
        self.V_eln = np.zeros(self.Nz)
        
        po0 = -V0 * a_init ** 2 * self.e0 * self.er
        for i in range(self.Nz):
            self.po_new[i] = po0 * np.exp(-a_init * self.Zz[i])
            self.V_eln[i] = V0 * np.exp(-a_init * self.Zz[i])
        self.Sig = -po0 / a_init
    
        self.V_el0 = self.V_eln.copy()
        self.po_0 = self.po_new.copy()
    
    def calc_po1(self):
        self.poh_max = 0.0
        self.poe_max = 0.0
        self.poMe_max = 0.0
        self.delta_po = 0.0
        for i in range(self.Nz):
            z = self.Zz[i]
            self.eVz = -self.Vels(z)
            po2 = self.poh(z)
            self.po_h[i] = po2
            self.poh_max = max(self.poh_max, po2)

            po3 = self.poe(z)
            self.po_e[i] = po3
            self.poe_max = min(self.poe_max, po3)

            po4 = self.poMIGS(z)
            self.po_MIGS[i] = po4
            self.poMe_max = min(self.poMe_max, po4)
            self.po1[i] = po2 + po3 + po4 - self.po00  
            if self.po_new[i] > 1e-15:
                self.delta_po += (self.po1[i] - self.po_0[i]) ** 2
    
        self.delta_po /= float(self.Nz)
    
    def poh(self, z):
        R = integral(self.Exxh1[0] , self.Exxh2[3] , self.poh_2)
        return R / self.V_DSC
    
    def poh_2(self, E):
        dexpx = np.exp((E - self.EFermi1 + self.eVz)/self.kbT)
        poh_2 = self.DOS_SCs(E)*(dexpx/(1+dexpx))
        return poh_2
    
    def poe(self, z):
        R = integral(self.Exxe1[0] , self.Exxe2[3] , self.poe_2)
        return - R / self.V_DSC

    
    def poe_2(self, E):
        dexpx = np.exp((E - self.EFermi1 + self.eVz)/self.kbT)
        poe_2 = self.DOS_SCs(E)/(1+dexpx)
        return poe_2
        
    def poMIGS(self,z):
        self.zp = z
        if z < 1000.0:
            R = integral(self.Exxm1[0], self.Exxm2[3], self.poMIGS_2)
        else:
            R = 0
        return - R / self.V_D0

    def poMIGS_2(self, E):
        dexpx = np.exp((E - self.EFermi2 + self.eVz)/self.kbT)
        return self.DMIGS(E) / (1.0 + dexpx)

    def DMIGS(self, E):
        if self.zp > self.z2: 
            return 0.0
        else:
            zconnect = 1.5
            if self.zp < zconnect:
                Dx = 0
                for k in range(self.Nk):
                    cx = 1/np.exp(-2.0 * self.ImKs2(E, 0) * self.z2)
                    #self.calc_sep(E, self.ImKHs2(E, k), self.ImKs2(E, k))
                    if k == 0:
                        self.DLDH1 = cx
                    Dx += self.wk[k] * self.DOS_Ms(E, k) * np.exp(-2.0 * self.ImKs2(E, k) * self.ckA * abs(self.zp)) * cx
                DMIGS1 = Dx / self.sumk
            else:
                DMIGS1 = 1e20
    
            if abs(self.zp) > 0.0:
                ImKs2x = self.ImKs2(E, 0)
                DMIGS_G1 = self.DOS_Ms(E, 0) * np.exp(-2.0 * ImKs2x * self.ckA * abs(self.zp)) * (ImKs2x * self.ckA) / abs(self.zp) / (self.a2p**2) * (1.0 - np.exp(-((self.a2p**2) / (ImKs2x * self.ckA)) * abs(self.zp)))
            else:
                DMIGS_G1 = self.DOS_Ms(E, 0)
    
            DMIGS_G1 *= self.DLDH1
    
            if self.zp < zconnect + 0.5:
                if DMIGS1 > DMIGS_G1:
                    return DMIGS_G1
                else:
                    return DMIGS_G1
            else:
                return DMIGS_G1
            
    def DOS_Ms(self, E, k):
        if E < self.Ef_DOS_M[0] or E > self.Ef_DOS_M[-1]:
            return 0.0
        else:
            DOS_Ms = ispline(E, self.kp_dos_function_list[k])
            if DOS_Ms < 0.0:
                return 0.0
            else:
                return DOS_Ms
            

    def ImKs2(self,E, k):
        if self.Emin1 <= E <= self.Emax1:
            return ispline(E, self.cbs_function_list[k])
        else:
            return 1e5 
    
    def mixing_po(self): 
        sigmoid_a = 1 / (1 + np.exp(-self.alfa))
        self.alfa = self.alfa + self.learning_rate * (1 - sigmoid_a)

        for i in range(self.Nz):
            self.po_new[i] = (self.alfa) * self.po_0[i] + (1-self.alfa) * self.po1[i]
        self.charge_density_function = spline(self.Zz, self.po_new)

    
    def calc_elpot(self):
        self.V_el1 = np.zeros(self.Nz)
        self.El_f = np.zeros(self.Nz)
    
        for i in range(self.Nz):
            self.V_el1[i] = self.Vel(self.Zz[i])
            
        for i in range(1, self.Nz-1):
            self.El_f[i] = -(self.V_el1[i + 1] - self.V_el1[i - 1]) / (self.Zz[i + 1] - self.Zz[i - 1])
    
        self.El_f[self.Nz - 1] = 0.0
        self.El_f[0] = self.El_f[1] - (self.El_f[2] - self.El_f[1]) * (self.Zz[1] - self.Zz[0]) / (self.Zz[2] - self.Zz[1])
    
        self.delta_V = 0.0
        for i in range(self.Nz):
            delta_Vi = abs(self.V_el1[i] - self.V_el0[i])
            self.delta_V = self.delta_V + delta_Vi ** 2

        self.delta_V = np.sqrt(self.delta_V) / float(self.Nz)
    
    def F6(self, zs):
        return -self.pos(zs)*(zs-self.zp)
    
    def Vel(self, z):
        eps = 1e-12
        self.zp = z
        intpo21 = 0
        intpo22 = 0
        intpo23 = 0
        intpo24 = 0
        intpo25 = 0
        intpo26 = 0
        if z < 10.0:
            intpo21 = integral(z, 10.0, self.F6)
            intpo22 = integral(10.0, 100.0, self.F6)
            intpo23 = integral(100.0, 1000.0, self.F6)
            intpo24 = integral(1000.0, 10000.0, self.F6)
            intpo25 = integral(10000.0, 100000.0, self.F6)
            intpo26 = integral(100000.0, self.Zz[-1], self.F6)
        elif z < 100.0:
            intpo22 = integral(z, 100.0, self.F6)
            intpo23 = integral(100.0, 1000.0, self.F6)
            intpo24 = integral(1000.0, 10000.0, self.F6)
            intpo25 = integral(10000.0, 100000.0, self.F6)
            intpo26 = integral(100000.0, self.Zz[-1], self.F6)
            intpo2 = intpo22 + intpo23 + intpo24 + intpo25 + intpo26
        elif z < 1000.0:
            intpo23 = integral(z, 1000.0, self.F6)
            intpo24 = integral(1000.0, 10000.0, self.F6)
            intpo25 = integral(10000.0, 100000.0, self.F6)
            intpo26 = integral(100000.0, self.Zz[-1], self.F6)
            intpo2 = intpo23 + intpo24 + intpo25 + intpo26
        elif z < 10000.0:
            intpo24 = integral(z, 10000.0, self.F6)
            intpo25 = integral(10000.0, 100000.0, self.F6)
            intpo26 = integral(100000.0, self.Zz[self.Nz-1], self.F6)
        elif z < 100000.0:
            intpo25 = integral(z, 100000.0, self.F6)
            intpo26 = integral(100000.0, self.Zz[self.Nz-1], self.F6)
        else:
            intpo26 = integral(z, self.Zz[self.Nz-1], self.F6)

        intpo2 = intpo21 + intpo22 + intpo23 + intpo24 + intpo25 + intpo26
        Vel = (1.0 / (self.er * self.e0) * intpo2) + (1.0 / (self.er * self.e0) * self.Sig_gate * (z - self.Lsc))
        return Vel
    
    def mixing_V(self):
        dV = np.zeros(self.Nz)
        dV1 = np.zeros(self.Nz)
        self.El_f2 = np.zeros(self.Nz)
        alfa_Sig = 1.0
        for i in range(2, self.Nz-1):
            self.El_f2[i] = -(self.V_el1[i+1] - self.V_el1[i-1]) / (self.Zz[i+1] - self.Zz[i-1])
    
        self.El_f2[-1] = 0.0
        self.El_f2[1] = self.El_f2[2] - (self.El_f2[3] - self.El_f2[2]) * (self.Zz[2] - self.Zz[1]) / (self.Zz[3] - self.Zz[2])
        self.El_f2[0] = self.El_f2[1] - (self.El_f2[2] - self.El_f2[1]) * (self.Zz[1] - self.Zz[0]) / (self.Zz[2] - self.Zz[1])
    
        for i in range(self.Nz):
            self.V_eln[i] = (self.alfa) * self.V_el0[i] + (1-self.alfa) * self.V_el1[i]
        #self.V_eln = self.V_el1
        self.electric_potential_function = spline(self.Zz, self.V_eln)
        self.dVn0 = self.electric_potential_function.c[2][0] * (self.er * self.e0)
        
    def calc_check_scf(self):
        acc_scf = self.delta_po / self.po_new[0]
        if acc_scf < 1e-5:
            self.L_conv = True
        ddEf = abs(self.dEf - self.dEf1)
        print(f"accuracy of scf cycle   %.3e"%acc_scf)
        print(f"accuracy reached for deltaE   %.3e"%ddEf)
    

class SBPlot(SBCalculator):
    def __init__(self, result_path, Temperature=300):#, test=True):
        # if test:
        #     result = ReadResult(result_path)
        #     self.EFermi1 = result.EFermi_input
        #     self.ECBM = result.ECBM
        #     self.plot_results()
        #     return
        super().__init__(result_path)
        print("\n\n############### Plotting Schottky Barrier (SB) results ###############\n")
        self.calc_ILW()
        self.calc_DLW()
        self.write_results()
        self.plot_results()
        self.print_results()

    def calc_ILW(self):
        if self.L_p_type:
            po0 = self.po_e[0]
        elif self.L_n_type:
            po0 = self.po_h[0]
        else:
            po0 = 0.0
    
        s0 = 1.0e20
        y_list = []
        for i in range(self.Nz):
            if self.L_p_type:
                s1 = abs(self.po_e[i] - (po0 / math.e))
            elif self.L_n_type:
                s1 = abs(self.po_h[i] - (po0 / math.e))
                y_list.append(s1)
                #print('s1',s1)
            else:
                s1 = 0.0
    
            if s1 < s0:
                im = i
                s0 = s1
        y_list = np.array(y_list)
        self.ILW = self.Zz[im]
        self.ilw_conversion_factor, self.ilw_unit = self.convert_length(self.ILW)
        self.ILW = self.ILW * self.ilw_conversion_factor
    
    def calc_DLW(self):
        po0 = -self.po00
        s0 = 1.0e20
        y_list = []
        for i in range(self.Nz):
            if self.L_p_type:
                s1 = abs((self.po_h[i] - self.po00) - (po0 / math.e))
            elif self.L_n_type:
                s1 = abs((self.po_e[i] - self.po00) - (po0 / math.e))
                y_list.append(s1)
            else:
                s1 = 0.0
    
            if s1 < s0:
                im = i
                s0 = s1
        y_list = np.array(y_list)
        self.DLW = self.Zz[im]
        self.dlw_conversion_factor, self.dlw_unit = self.convert_length(self.DLW)
        self.DLW = self.DLW * self.dlw_conversion_factor
        
    def write_results(self):

        with open('potential.dat', 'w') as potential_file:
            potential_file.write(' %d     !  "z (Å)", "Potential (eV)"\n'%self.Nz)
            for i in range(self.Nz):
                potential_file.write(f"%.5f   %.10f\n"%(self.Zz[i], -self.V_eln[i]))
    
        with open('chg_density.dat', 'w') as chg_density_file:
            chg_density_file.write(' %d     !  "z (Å)", "Chg density (cm$^-3)"\n'%self.Nz)
            for i in range(self.Nz):
                chg_density_file.write(f"%.5f  %.12e\n"%(self.Zz[i], self.po_new[i] * 1e24))

        if self.L_n_type:
            self.SB_potential = -self.V_eln + self.ECBM - self.EFermi1
        elif self.L_p_type:
            self.SB_potential = -self.V_eln + self.EFermi1 - self.EVBM

        with open('sb_profile.dat', 'w') as sb_profile_file:
            sb_profile_file.write(' %d     !  "z (Å)", "SB profile (eV)"\n'%self.Nz)
            for i in range(self.Nz):
                sb_profile_file.write(f"%.5f  %.12e\n"%(self.Zz[i], self.SB_potential[i]))

    def plot_results(self):
        # potential_data = np.genfromtxt('potential.dat', skip_header=1)
        # chg_density_data = np.genfromtxt('po.dat', skip_header=1)
        # self.Zz = potential_data[:, 0]
        # self.V_eln = - potential_data[:, 1]
        # # chg_density_data[:, 0]
        # self.po_new = -chg_density_data[:, 1]
        # self.DLW = 27999021.70

        conversion_factor, unit = self.convert_length(max(self.Zz))
        self.Zz = conversion_factor * self.Zz
        self.SB_potential = -self.V_eln + self.ECBM - self.EFermi1

        fig, ax1 = plt.subplots()
        line1, = ax1.plot(self.Zz, -self.V_eln, 'g--', label='Electrical potential')
        ax1.set_xlabel('Depth from interface (%s)'%unit)
        ax1.set_ylabel('Energy (eV)', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        line2, = ax1.plot(self.Zz, self.SB_potential, 'g-', label='SB Profile')
        ax2 = ax1.twinx()
        line3, = ax2.plot(self.Zz, self.po_new, 'b-', label='Charge Density')
        ax2.set_ylabel('Charge Density (cm$^{-3}$)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        lines = [line1, line2, line3]
        labels = [line1.get_label(), line2.get_label(), line3.get_label()]
        ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
        plt.xlim(0, self.DLW)
        plt.title('SB results')
        plt.savefig('SB_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        #plt.show()


    def convert_length(self, length_angstrom):
        length_meter = length_angstrom * 1e-10
        if length_meter >= 1e-6:
            return 1e-4, "µm"
        elif length_meter >= 1e-9:
            return 1e-1, "nm"
        else:
            return 1, "Å"

    def print_results(self):
        E_00 = -(self.V_eln[3] - self.V_eln[2]) / (self.Zz[3] - self.Zz[2])
        P_00 = self.kappa * E_00
    

        print("Fermi level of bulk semiconductor               EFermi_00 = %15.5f eV"%self.EFermi_00)
        print("Conduction bandminimum of semiconductor              ECBM = %15.5f eV"%self.ECBM)
        print("Charge neutrality level for interface vs VBM          CNL = %15.5f eV"%self.CNL)
        print("Schottky barrier height                               SBH = %15.5f eV"%self.SBH)
        print("Amplitude of Schottky potential energy               -eV0 = %15.5f eV"%-self.V_eln[0])
        # print("Energy filling level on the surface               delta E = %15.5e eV"%self.dEf)
        print("Charge on the interface                               Sig = %15.5e cm-2"%(self.Sig*1e16))
        # print("Electric field close to the surface of semiconductor E(0) = %15.5e V/A"%E_00)
        # print("Polarization close to the surface of semiconductor   P(0) = %15.5e e/A^2"%P_00)
        print("Depletion layer width                                 DLW = %15.2f %s"%(self.DLW, self.dlw_unit))
        print("Inversion layer width                                 ILW = %15.2f %s"%(self.ILW, self.ilw_unit))
        print("\nNORMAL TERMINATION")