#!/usr/bin/env python
import argparse
from arpaca.utils import *
import time

parser = argparse.ArgumentParser(description="Generate a surface for VASP input.")

parser.add_argument(
    "--structure_path",
    "-p",
    type=str,
    default=None,
    help="Path to the structure file (default: None)")

parser.add_argument(
    "--miller_index",
    "-m",
    type=int,
    nargs=3,
    default=None,
    help="Miller index as three integers (e.g., 1 1 1) (default: None)")

parser.add_argument(
    "--thickness",
    "-t",
    type=float,
    default=20.0,
    help="Minimum thickness of the slab in Å (default: 20)")

parser.add_argument(
    "--vacuum",
    "-v",
    type=float,
    default=10.0,
    help="Vacuum thickness in Å (default: 10)")

args = parser.parse_args()
if all(getattr(args, arg) == parser.get_default(arg) for arg in vars(args)):
	print("No arguments provided. Use the '--help' option to see usage details if you want.")

structure_path=args.structure_path
miller_index=args.miller_index
thickness=args.thickness
vacuum=args.vacuum

surface=GenSurface()
surface.slice_slab_cartesian()
#surface.slice_slab_direct()

surface.cbs_surface_maker()