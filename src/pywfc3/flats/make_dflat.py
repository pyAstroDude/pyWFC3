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
    parser.add_argument('manifest', metavar='Manifest', type=str, nargs='?',
                        help='Name of the input manifest listing ' +
                        'the files to process.')
    parser.add_argument('-p', '--param', dest='yamlfile', type=str,
                        action='store', default=None,
                        help='Name of the configuration YAML file which ' + 
                        'lists all the required input parameters to run ' +
                        'the steps from WFC3_ISR_2021-10 to generate the ' +
                        'D-flat.')
    
    args = parser.parse_args()
    
    if not args.manifest and not args.yamlfile:
        parser.error(" The following arguments are required: \n"
                     "\t Manifest or -p/--param")
    
    mdf = MakeDFlat.MakeDFlat(args)
    
    # Get pipeline parameters from (priorities) CLI, user param file and 
    # default param file.
    # params = mdf.get_pipeline_params(args)
    # params = mdf.params
        
    # # Set up directories path.
    mdf.setup_directories()
    
    # Read the manifest
    files_list = mdf.read_manifest(mdf.params['files']['manifest'])
    
    # Make panda dataframe file IDs and FITS header info.
    files_df = mdf.make_dataframe(files_list)
    
    # # Validate the pandas dataframe.
    valid_df = mdf.validate_df(files_df, 
                               band=mdf.params['instrument']['filter'])
    
    # # For each input file mask outliers, sources and update DQ extension.
    masked_df = mdf.mask_outliers(valid_df, 
                                  thresholds=mdf.params['processing']['thresholds'], 
                                  ncores=mdf.params['processing']['cores'])
    
    raw_df = mdf.get_raw_data(masked_df)
    
    unflt_df = mdf.run_calw3_pipe(raw_df, 
                                  outdir=mdf.params['paths']['output'],
                                  ncores=mdf.params['processing']['cores'])
    
    print(unflt_df.to_string())
    # pprint(params)
    # print()
    # sys.exit(f" Successfully processed data for {mdf.band}.")
    
    