## Arguments: [density], [chem_formula]
import sys
import argparse
from arpaca.utils import genAmorphous

parser = argparse.ArgumentParser(
    description='aRPaCa: ab initio RRAM Parameter Calculator',
    formatter_class=argparse.RawTextHelpFormatter
    )

parser.add_argument('-c', '--chem_formula',
                    type=str,
                    help='chemical formula of amorphous (ex. Hf32O64)')

parser.add_argument('-d', '--density',
                    type=float,
                    help='density of amorphous in g/cm3')

args = parser.parse_args()

genAmorphous(density=args.density, 
             chem_formula=args.chem_formula, 
             outfile=f'POSCAR_{args.chem_formula}')
