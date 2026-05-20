#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:31:49 2025

@author: sshenoy

This is a calling function to call MakeDFlat class and run all the step (or
individual steps) from WFC3 ISR 2021-10 to generate WFC3 IR D-Flat.
"""


# import os
import sys
import argparse
from pathlib import Path
from pprint import pprint


# from pywfc3 import utils
from pywfc3.flats import MakeDFlat 

def main():
    "Command line code to call MakeDFlat."
    parser = argparse.ArgumentParser(description='Genarate WFC3 IR D-Flat ' +
                                     'using the input manifest and a JSON.' +
                                     'parameter file.')
    parser.add_argument('Manifest', metavar='Manifest', type=str, 
                        help='Name of the input manifest listing ' +
                        'the files to process.')
    parser.add_argument('-i', '--inpath', dest='InputDirectory', type=str,
                        action='store', default='./',
                        help='Full path to the input directory where the ' +
                        'input manifest is stored. Default is current ' +
                        'working directory.')
    parser.add_argument('-d', '--datadir', dest='DataDirectory', type=str,
                        action='store', default='data',
                        help='Full path to the input data irectory where ' +
                        'the input data fits files are stored. Default is '+
                        'current working directory.')
    parser.add_argument('-o', '--outpath', dest='OutputDirectory', type=str,
                        action='store', default=None,
                        help='Name of the output directory where the ' +
                        'processed files will be placed.')
    parser.add_argument('-c', '--config', dest='config', type=str,
                        action='store', default=None,
                        help='Name of the configuration JSON file which ' + 
                        'lists all the required input parameters to run ' +
                        'the steps from WFC3_ISR_2021-10 to generate the ' +
                        'D-flat.')
    parser.add_argument('-f', '--filter', dest='Filter', type=str,
                        action='store', default=None,
                        help='WFC3 IR filter/band to process. Default ' +
                        'is F140W.')
    parser.add_argument('-t', '--thresholds', dest='Thresholds', type=float,
                        nargs=3, default=[2.0, 5.0, 2.0])
    
    args = parser.parse_args()
    # print(f"All Arguments: {args}")
    # print(vars(args).items())
    
    mdf = MakeDFlat.MakeDFlat()
    
    # Get pipeline parameters from (priorities) CLI, user param file and 
    # default param file.
    params = mdf.get_pipeline_params(args)
    
    ### This is for debugging. Remove once code is robust.
    # print()
    # pprint(params)
        
    # # Set up directories path.
    mdf.setup_directories(params['InputDirectory'], is_input=True)
    mdf.setup_directories(params['DataDirectory'], is_data=True)
    mdf.setup_directories(params['OutputDirectory'], is_output=True)
    
    # Read the manifest
    files_list = mdf.read_manifest(params['Manifest'])
    
    # Make panda dataframe file IDs and FITS header info.
    files_df = mdf.make_dataframe(files_list)
    
    # Validate the pandas dataframe.
    valid_df = mdf.validate_df(files_df, band=args.Filter)
    
    # for each input file mask outliers.
    masked_df = mdf.mask_outlliers(valid_df, thresholds=params['Thresholds'],
                                   ncores=params['Cores'])
    
    # print(masked_df.head().to_string())
    print(valid_df.columns)
    sys.exit(f" Successfully processed data for {mdf.band}.")
    
    