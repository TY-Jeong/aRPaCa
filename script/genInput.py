from arpaca.utils import genInput

potcar = input('kind of potcar (lda or pbe) : ')
nsw = input('iteration number (int): ')
potim = input('time step in fs (float): ')
temp = input('temperature in K (int): ')
charge = input('charge state of the system (float): ')

nsw, temp = int(nsw), int(temp)
potim, charge = float(potim), float(charge)

genInput(potcar=potcar,
         nsw=nsw,
         potim=potim,
         temp=temp,
         charge=charge,
         ncore=4)
