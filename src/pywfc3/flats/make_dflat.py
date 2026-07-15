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


from pywfc3 import utils
from pywfc3.flats import MakeDFlat 

def main():
    "Command line code to call MakeDFlat."
    parser = argparse.ArgumentParser(description='Generate WFC3 IR D-Flat ' +
                                     'using the input manifest and a YAML ' +
                                     'parameter file.')
    parser.add_argument('manifest', metavar='Manifest', type=str, nargs='?',
                        help='Name of the input manifest listing ' +
                        'the files to process.')
    parser.add_argument('-p', '--param', dest='yamlfile', type=str,
                        action='store', default=None,
                        help='Name of the YAML parameter file which ' + 
                        'lists all the required input parameters to run ' +
                        'the steps from WFC3_ISR_2021-10 to generate the ' +
                        'D-flat.')
    
    args = parser.parse_args()
    
    if not args.manifest and not args.yamlfile:
        parser.error(" The following arguments are required: \n"
                     "\t Manifest or -p/--param")
    
    mdf = MakeDFlat.MakeDFlat(args)
        
    # Set up directories path.
    mdf.setup_directories()
    
    # Read the manifest
    files_list = mdf.read_manifest(mdf.params['files']['manifest'])
    
    # Make panda dataframe file IDs and FITS header info.
    files_df = mdf.make_dataframe(files_list)
    
    # Validate the pandas dataframe.
    valid_df = mdf.validate_df(files_df, 
                               band=mdf.params['instrument']['filter'])
    
    # For each input file mask outliers, sources and update DQ extension.
    masked_df = mdf.mask_outliers(valid_df, 
                                  thresholds=mdf.params['processing']['thresholds'], 
                                  ncores=mdf.params['processing']['cores'])
    
    raw_df = mdf.get_raw_data(masked_df)
    
    unflt_df = mdf.run_calw3_pipe(raw_df, 
                                  outdir=mdf.params['paths']['output'],
                                  ncores=mdf.params['processing']['cores'])
    
    comskflt_df, cmf_dict = mdf.update_mask(unflt_df, 
                                  save=mdf.params['processing']['save'])
    
    flat_result = mdf.generate_flat(comskflt_df)
    
    if isinstance(flat_result, dict):
        print(f"Successfully generated time-dependent flats for active blobs: {list(flat_result.keys())}")
        dflat_paths = mdf.get_new_dflat(flat_result)
        print(f"Successfully generated ratioed flats for active blobs: {list(dflat_paths.values())}")
        updated_dflat_path = mdf.update_dflat_with_blobs(flat_result, dflat_paths)
        print(f"Successfully generated updated D-flat at: {updated_dflat_path}")
    else:
        flat, flat_unc, mask = flat_result
        print("Successfully generated standard master flat field.")
        dflat_path, dflat_hdul = mdf.get_current_flat('dflat')
        dflat_hdul.close() 

    mdf.save_pipeline_params()

if __name__ == "__main__":
	main()
