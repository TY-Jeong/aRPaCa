"""
Post-process module after GreeKuP run.
"""

import os
import sys

## for CondAnalyze
import re
import pandas as pd
import numpy as np

## for condplot
import matplotlib.pyplot as plt

class CondAnalyze:
    """
    Gathers the conductivities of lower directories.
    The directory structure should be,
    (custom_criteria_with_numbers)/(temperature)K/(snapshot_number).
    Run this class after all GreeKuP calculations are done.
    Every directory with 3_cond/ReCONDUCTIVITY_DCCOND are gathered and listed.

    """
    def __init__(self,
                 poscar="POSCAR",
                 outcar="OUTCAR",
                 output_file="data.csv",
                 output_write=False,
                 cond_dir="3_cond",
                 cond="ReCONDUCTIVITY_DCFORM"):
        self.outcar = outcar
        self.poscar = poscar
        self.output_file = output_file
        self.output_write = output_write  ## please assign True to save csv.
        self.cond_dir = cond_dir
        self.cond = cond

        self.check_cond = self.cond_dir + "/" + self.cond
        self.cwd = os.getcwd()

        self.df = pd.DataFrame() ## will be the data.csv

        self.list_dir()
        self.outcar_energy()
        self.poscar_numatom()
        self.nabla_cond()
        self.dir2types()
        self.cond_weight()

        if self.output_write is True:
            if os.path.exists(self.output_file):
                print("data.csv exists")
            else:
                # self.df[[list-of-desired-order-of-columns]]
                self.df.to_csv(self.output_file, index=False)


    def list_dir(self):
        """
        Searches the ReCONDUCTIVITY_DCCOND files and returns the list of paths.
        """
        self.dir_list = []
        for root, dirs, files in os.walk(self.cwd):  ## doing in cwd; change if needed
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if self.check_cond in file_path:
                    self.dir_list.append(file_path.replace(self.check_cond, ""))
        self.dir_list_slim = []
        for dir_long in self.dir_list:
            self.dir_list_slim.append(dir_long.replace(self.cwd, ""))


    def outcar_energy(self, energy_header = "energy [eV]"):
        """
        Gets the energy of last ionic step of OUTCAR.
        """
        self.energy_header = energy_header
        self.energy_list = []
        for snapshot in self.dir_list:
            energy = None
            outcar_path = os.path.join(snapshot, self.outcar)
            with open(outcar_path, 'r') as infile: #, open(self.output_file, 'w') as outfile:
                for line in infile:
                    match = re.search(r"free  energy", line)
                    if match:
                        energy = line.split()[4]
                if energy:
                    self.energy_list.append(energy)
                else:
                    self.energy_list.append(np.nan)
                    print("No OUTCAR energy found")  ## in where?

        self.df[self.energy_header] = self.energy_list
        self.df[self.energy_header] = pd.to_numeric(self.df[self.energy_header])


    def poscar_numatom(self, numatom_header = "No. of atoms [#]"):
        """
        Gets the number of atoms in POSCAR.
        """
        self.numatom_header = numatom_header
        self.numatom_list = []
        for snapshot in self.dir_list:
            numatom = None
            poscar_path = os.path.join(snapshot, self.poscar)
            with open(poscar_path, 'r') as infile:
                lines = infile.readlines()
                atom_line = lines[6].strip()
                atom_numbers = re.findall(r'\d+', atom_line)
                numatom = sum(int(num) for num in atom_numbers)
                self.numatom_list.append(numatom)

        self.df[self.numatom_header] = self.numatom_list  ## already int


    def nabla_cond(self, cond_header = "conductivity [S/m]"):
        """
        Gets the DC conductrivity from ReCONDUCTIVITY_DCFORM.
        """
        self.cond_header = cond_header
        self.cond_list = []
        for snapshot in self.dir_list:
            conductivity = None
            is_matched = 0
            cond_path = os.path.join(snapshot, self.check_cond)
            with open(cond_path, 'r') as infile: #, open(self.output_file, 'w') as outfile:
                for line in infile:
                    match = re.search(r"conductivity,", line)
                    if is_matched == 1:
                        conductivity = line.split()[1]
                        self.cond_list.append(conductivity)
                        break
                    if match:
                        is_matched = is_matched + 1  # check if matched and get cond from next line
                if is_matched == 0:
                    self.cond_list.append(np.nan)
                    print("No ReCONDUCTIVITY_DCFORM conductivity found")

        self.df[self.cond_header] = self.cond_list
        self.df[self.cond_header] = pd.to_numeric(self.df[self.cond_header])


    def dir2types(self,
                  first_header="composition",
                  temp_header = "temperature [K]",
                  snap_header = "snapshot No."):
        """
        Converts directories into criteria.
        composition (customizable), temperature, snapshot, and possibly more in the future.
        """

        self.first_header = first_header
        self.temp_header = temp_header
        self.snap_header = snap_header

        self.temp_list = []
        self.snap_list = []
        self.first_list = []

        separated = None
        for dirs in self.dir_list_slim:
            separated = dirs.strip("/").split("/")
            ## save from end (snap-temp-compo) for further modification that needs multiple criteria
            self.snap_list.append(separated[len(separated)-1])
            self.temp_list.append(separated[len(separated)-2].strip("K"))
            if len(separated) > 2:
                self.first_list.append(separated[len(separated)-3])
            else:
                self.first_list.append(np.nan)

        self.df[self.temp_header] = self.temp_list
        self.df[self.temp_header] = pd.to_numeric(self.df[self.temp_header])

        self.df[self.snap_header] = self.snap_list
        self.df[self.snap_header] = pd.to_numeric(self.df[self.snap_header])

        self.df[self.first_header] = self.first_list
        #self.df[self.first_header] = pd.to_numeric(self.df[self.first_header])


    def cond_weight(self,
                    energy_per_atom_header="energy per atom [eV/#]",
                    weight_header="weight",
                    wcond_header="average conductivity [S/m]",
                    wstd_header="standard deviation [S/m]"):
        """
        Convert the energies of each snapshot into weight.
        Weight is proportional to the exp(-Energy/kT).
        """
        boltzmann = 8.61733326e-5  # boltzmann cosntant eV/K

        self.energy_per_atom_header = energy_per_atom_header
        self.weight_header = weight_header
        self.wcond_header = wcond_header
        self.wstd_header = wstd_header

        self.df[self.energy_per_atom_header] = \
            self.df[self.energy_header] / self.df[self.numatom_header]

        self.df[self.weight_header] = \
            np.exp(-self.df[self.energy_per_atom_header] / (boltzmann * self.df[self.temp_header]))
        self.df_grouped = self.df.groupby([self.first_header, self.temp_header], dropna=False)
        self.df[self.weight_header] = \
            self.df[self.weight_header] / self.df_grouped[self.weight_header].transform('sum')

        self.df["wCond"] = self.df[self.weight_header] * self.df[self.cond_header]
        self.df["wCond2"] = self.df[self.weight_header] * (self.df[self.cond_header] ** 2)
        self.df_grouped = \
            self.df.groupby([self.first_header, self.temp_header], dropna=False)  # update

        print(self.df_grouped["wCond"].agg('sum'))

        # weighted average of conductivity
        self.df[self.wcond_header] = self.df_grouped["wCond"].transform('sum')

        self.df["wDeltaSquare"] = self.df[self.weight_header] * (self.df[self.cond_header] \
                                  - self.df[self.wcond_header]) ** 2
        self.df_grouped = \
            self.df.groupby([self.first_header, self.temp_header], dropna=False)  # update
        self.df[self.wstd_header] = self.df_grouped["wDeltaSquare"].transform('sum') ** 0.5



class CondPlot:
    """
    Plots conductivity from the CondAnalyze.df or the output csv file.

    additional (bool): If True, the x-axis becomes composition. If False, temperature.
    temperature (int?): Used only if additional=True. Temperature [K] to be plotted.
    """
    def __init__(self, df, plottype, fig, ax, additional=True, temperature=300):

        if isinstance(df, pd.DataFrame):
            self.df = df
        elif isinstance(df, str) and os.path.exists(df):
            try:
                self.df = pd.read_csv(df)
            except Exception as e:
                print("invalid file", e)
        else:
            raise Exception("invalid input")


        self.plottype = plottype ## 'error' or 'bubble'

        self.fig = fig ## fig, ax = plt.subplots() should be called beforehand
        self.ax = ax

        self.additional = additional
        self.temperature = temperature

        ax.set_xlabel('composition')
        ax.set_ylabel('electrical conductivity [S/m]')
        if self.plottype == 'error':
            self.plot_error(fig, ax)
        elif self.plottype == 'bubble':
            self.plot_bubble(fig, ax)
        else:
            print('invalid plottype')

        #plt.show()

    def plot_error(self, fig, ax):
        """
        Line plot (weighted average of conductivity)
        with error bar (standard deviation of conductivity).
        """

        if self.additional is True:
            self.df = self.df.dropna(subset=["composition"])  ## drop rows without composition value
            self.df = self.df[self.df["temperature [K]"] == self.temperature]  ## leaving only one T

            trimmed = []
            for compo in self.df["composition"]:
                trimmed.append(float(re.sub('[^0-9,.]', '', compo)))  ## error treating nan
            self.df["composition"] = trimmed

            self.df = self.df.sort_values("composition", ascending=True)
            ax.errorbar(self.df["composition"],
                        self.df["average conductivity [S/m]"],
                        self.df["standard deviation [S/m]"],
                        fmt='ko-', linewidth=1, capsize=5)
        else:
            self.df = self.df[self.df["composition"].isna()]  ## drop rows with composition value
            self.df = self.df.sort_values("temperature [K]", ascending=True)
            ax.errorbar(self.df["temperature [K]"],
                        self.df["average conductivity [S/m]"],
                        self.df["standard deviation [S/m]"],
                        fmt='ko-', linewidth=1, capsize=5)


    def plot_bubble(self, fig, ax):
        """
        Bubble plot (conductivity of each snapshot)
        """

        def num_to_range(num, in_min, in_max, out_min=0, out_max=750):
            # out_max = maximum bubble size. 750 seems be ok.
            # return out_min + (float(num - in_min) / float(in_max - in_min) * (out_max - out_min))
            return out_min + (float(num - 0) / float(in_max - 0) * (out_max - out_min))

        if self.additional is True:
            self.df = self.df.dropna(subset=["composition"])  ## drop rows without composition value
            self.df = self.df[self.df["temperature [K]"] == self.temperature]  ## leave only one T

            trimmed = []
            for compo in self.df["composition"]:
                trimmed.append(float(re.sub('[^0-9,.]', '', compo)))  ## error treating nan
            self.df["composition"] = trimmed

            weight_list = self.df["weight"]
            weight_min = [min(weight_list) for i in range(len(weight_list))]
            weight_max = [max(weight_list) for i in range(len(weight_list))]
            weight_list = list(map(num_to_range, weight_list, weight_min, weight_max))
            self.df["weight"] = weight_list

            self.df = self.df.sort_values("composition", ascending=True)

            ax.scatter(self.df["composition"],
                       self.df["conductivity [S/m]"],
                       s=self.df["weight"],
                       c='#66ccff', linewidth=0.5, edgecolors='b', alpha=0.25)

        else:
            self.df = self.df[self.df["composition"].isna()]  ## drop rows with composition value

            weight_list = self.df["weight"]
            weight_min = [min(weight_list) for i in range(len(weight_list))]
            weight_max = [max(weight_list) for i in range(len(weight_list))]
            weight_list = list(map(num_to_range, weight_list, weight_min, weight_max))
            self.df["weight"] = weight_list

            self.df = self.df.sort_values("temperature [K]", ascending=True)

            ax.scatter(self.df["temperature [K]"],
                       self.df["conductivity [S/m]"],
                       s=self.df["weight"],
                       c='#66ccff', linewidth=0.5, edgecolors='b', alpha=0.25)


conductivity_list = CondAnalyze()

figure, axis = plt.subplots()
CondPlot(conductivity_list.df, 'error', figure, axis)  # can input the csv file
CondPlot(conductivity_list.df, 'bubble', figure, axis)  # can input the csv file
plt.show()
