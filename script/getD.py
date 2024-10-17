from arpaca.mass_transport.einstein import *

symbol = input('symbol of moving atom (ex. O) : ')
path_xdatcar = input('directory containing xdatcar files (ex. xdatcar) : ')
skip = int(input('skip (int) : '))
segment = int(input('segment (int) : '))
start = int(input('start (int) : '))
xyz = input('calculate xyz component (True or False) : ')
xyz = True if xyz=='True' else False


getDiffusivity(symbol=symbol,   
               path_xdatcar=path_xdatcar,
               skip=skip,
               segment=segment,
               start=start,
               xyz=xyz)

# getDiffusivity(symbol='O',   
#                path_xdatcar='./xdatcar',
#                skip=500,
#                segment=2,
#                start=1000,
#                xyz=False)
