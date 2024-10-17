from arpaca.mass_transport.einstein import *

symbol = input('symbol of moving atom (ex. O) : ')
path_xdatcar = input('directory containing xdatcar files (ex. xdatcar) : ')
skip = int(input('skip (int) : '))
segment = int(input('segment (int) : '))
start = int(input('start (int) : '))
xyz = bool(input('calculate xyz component (bool) : '))

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
