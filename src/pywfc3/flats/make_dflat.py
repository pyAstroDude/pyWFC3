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
from pprint import pprint


# from pywfc3 import utils
from pywfc3.flats import MakeDFlat 

def main():
    "Command line code to call MakeDFlat."
    parser = argparse.ArgumentParser(description='Genarate WFC3 IR D-Flat ' +
                                     'using the input manifest and a JSON.' +
                                     'parameter file.')
    parser.add_argument('Manifest', metavar='Manifest', type=str, nargs='?',
                        help='Name of the input manifest listing ' +
                        'the files to process.')
    parser.add_argument('-p', '--param', dest='ParamFile', type=str,
                        action='store', default=None,
                        help='Name of the configuration JSON file which ' + 
                        'lists all the required input parameters to run ' +
                        'the steps from WFC3_ISR_2021-10 to generate the ' +
                        'D-flat.')
    
    args = parser.parse_args()
    
    if not args.Manifest and not args.ParamFile:
        parser.error(" The following arguments are required: \n"
                     "\t Manifest or -p/--param")
    
    mdf = MakeDFlat.MakeDFlat()
    
    # Get pipeline parameters from (priorities) CLI, user param file and 
    # default param file.
    params = mdf.get_pipeline_params(args)
    
    ### This is for debugging. Remove once code is robust.
    pprint(params)
    print()
        
    # # Set up directories path.
    mdf.setup_directories(params['InputDirectory'], is_input=True)
    mdf.setup_directories(params['DataDirectory'], is_data=True)
    
    if params['Save']:
        mdf.setup_directories(params['OutputDirectory'], is_output=True)
    
    # Read the manifest
    files_list = mdf.read_manifest(params['Manifest'])
    
    # Make panda dataframe file IDs and FITS header info.
    files_df = mdf.make_dataframe(files_list)
    
    # Validate the pandas dataframe.
    valid_df = mdf.validate_df(files_df, band=params['Filter'])
    
    # For each input file mask outliers, sources and update DQ extension.
    masked_df = mdf.mask_outliers(valid_df, thresholds=params['Thresholds'],
                                   ncores=params['Cores'])
    
    sys.exit(f" Successfully processed data for {mdf.band}.")
    
    