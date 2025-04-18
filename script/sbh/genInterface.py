#!/usr/bin/env python
import argparse
from arpaca.utils import *

parser = argparse.ArgumentParser(description="Script to process substrate and film surface POSCAR files.")
parser.add_argument("-s", "--substrate", type=str, required=False, default=None,
                    help="Path of the surface POSCAR of the substrate.")
parser.add_argument("-f", "--film", type=str, required=False, default=None,
                    help="Path of the surface POSCAR of the film surface.")

args = parser.parse_args()

substrate = args.substrate
film = args.film

if substrate is None and film is None:
    print("No arguments provided. Use the '--help' option to see usage details if you want.")
else:
    print(f"Substrate path: {substrate if substrate else 'Not provided'}")
    print(f"Film path: {film if film else 'Not provided'}")

interface = GenInterface(substrate, film)
interface.interface_maker()
interface.edit_tool()