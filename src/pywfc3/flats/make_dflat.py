#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:31:49 2025

@author: sshenoy

This is a calling function to call MakeDFlat class and run all the step (or
individual steps) from WFC3 ISR 2021-10 to generate WFC3 IR D-Flat.
"""


import os
import sys
import argparse

from pywfc3 import utils
from pywfc3.flats import MakeDFlat 

def main():
    "Command line code to call MakeDFlat."
    parser = argparse.ArgumentParser(description='Genarate WFC3 IR D-Flat ' +
                                     'using the input manifest and a JSON.' +
                                     'parameter file.')
    parser.add_argument('manifest', metavar='manifest', type=str, nargs=1,
                        help='Name of the input manifest listing ' +
                        'the files to process.')
    parser.add_argument('-i', '--inpath', dest='inpath', type=str,
                        action='store', default=os.getcwd(),
                        help='Full path to the input directory where the ' +
                        'input manifest is stored. Default is current ' +
                        'working directory.')
    parser.add_argument('-d', '--datadir', dest='datadir', type=str,
                        action='store', default='data',
                        help='Full path to the input data irectory where ' +
                        'the input data fits files are stored. Default is '+
                        'current working directory.')
    parser.add_argument('-o', '--outpath', dest='outdir', type=str,
                        action='store', default=None,
                        help='Name of the output directory where the ' +
                        'processed files will be placed.')
    parser.add_argument('-c', '--config', dest='config', type=str,
                        action='store', default=None,
                        help='Name of the configuration JSON file which ' + 
                        'lists all the required input parameters to run ' +
                        'the steps from WFC3_ISR_2021-10 to generate the ' +
                        'D-flat.')
    
    args = parser.parse_args()
    # print(f"All Arguments: {args}")
    
    mf = MakeDFlat.MakeDFlat()
    
    if args.config is None:
        params = mf.read_params_file()
    else:
        params = mf.read_params_file(args.config)
        
    mf.setup_directories(params)
   
    print("\n")
    print(f"INPATH: {mf.inpath}")
    print(f"DATADIR: {mf.datadir}")
    print(f"OUTPATH: {mf.outpath}\n")
    
    print(params)
    
    sys.exit()
    
    