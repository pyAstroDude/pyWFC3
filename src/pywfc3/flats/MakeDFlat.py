#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:34:35 2025

@author: sshenoy

This is the class that can be used to run the procedures list in WFC3 ISR 
2021-10 to generate a WFC3 IR D-Flat.
"""

import sys
import json
import warnings
import importlib.resources
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from pywfc3 import utils 

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self):
        self.params = None
        self.inpath = None
        self.datadir = None
        self.outpath = None
        self.manifest = None
        self.filelist = None
        self.df = None
        self.hdul = None
        self.mode = None
        self.band = None
        self.mask_outside = []
        self.sigma = None
        self.edge = 8
        self.rate_outpath = None
    
    
    
    def read_params_file(self, paramfile=None):
        """
        Read the user supplied parameter file and set up the parameter 
        dictionary. If file is the wrong type or not found then use 
        deafult parameter JSONfile.

        Parameters
        ----------
        paramfile : JSON, optional
            Name of input parameter file in JSON format. Default is the 
            dflat.json file that is provided in the pywfc3 package.

        Returns
        -------
        A python dictionary of parameters key:value pairs.

        """
        
        # Determine the path to the default dflat.json file using importlib
        params = {}
        try:
            default_param_dir = importlib.resources.files('pywfc3.parameters')
            default_param_file = default_param_dir.joinpath('dflat.json')
            with importlib.resources.as_file(default_param_file) as p_file:
                try:
                    with open(p_file, 'r') as f:
                        params = json.load(f)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Error decoding default parameter file {p_file}: {e}")
        except ModuleNotFoundError:
            warnings.warn("Could not find pywfc3.parameters module. Default parameters not loaded.")
            params = {}

        # If user provided a paramfile, check if it exists and update params
        if paramfile:
            if Path(paramfile).exists():
                try:
                    with open(paramfile, 'r') as f:
                        user_params = json.load(f)
                        params.update(user_params)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Error decoding user parameter file {paramfile}: {e}")
            else:
                 warnings.warn(f"User parameter file not found: {paramfile}")
                 
        self.params = params

        return params
    
    
    def setup_directories(self, params):
        """
        Check if the input and data directories exist and create the output
        directory if it does not exist.

        Parameters
        ----------
        params : dict
            Dictionary of parameters.

        Returns
        -------
        None.

        """
        
        self.inpath = utils.check_directory(params['InputDirectory'])
        self.datadir = utils.check_directory(params['DataDirectory'], 
                                             data_dir=True)
        self.outpath = utils.get_output_directory(name=params['OutputDirectory'])
        
        
    def read_manifest(self, manifest):
        """ Read the input manifest and generate a list of files
        to process."""
        
        # Define the input manifest and check its existence.
        try:
            input_manifest = Path(manifest).resolve(strict=True)
            
            with input_manifest.open() as f:
                flist = [line.strip() for line in f]
            
            if len(flist) == 0:
                sys.exit(f"nput manifest, {manifest}, is empty.")
            else:
                flist.sort()
                self.filelist = flist
            
            self.manifest = input_manifest
        except FileNotFoundError():
            sys.exit(f"Input manifest, {manifest}, does not exist.")
        
        return self.filelist
    
    
    def make_dataframe(self, filelist):
        """Read the filelist and genarate a data frame with the 
        filenames and all the relevent metadata."""
        
        meta_data = {'FILEPATH': [],
                     'FILENAME': [],
                     'INSTRUME': [],
                     'OBSTYPE': [],
                     'FILTER': [],
                     'DETECTOR': [],
                     'SAMP_SEQ': [],
                     'NSAMP': [],
                     'DATE-OBS': []
                     }
        
        for file in filelist:
            try:
                hdr = fits.getheader(file)
                meta_data['FILEPATH'].append(Path(file).resolve(strict=True))
                for key in meta_data.keys():
                    if not key == 'FILEPATH':
                        meta_data[key].append(hdr[key])    
                
            except (OSError, FileNotFoundError) as error:
                print("\nERROR: File maybe missing or corrupted.")
                print(f"ERROR: {error}\n")
        
        input_df = pd.DataFrame(meta_data)
        
        self.df = input_df
        
        return self.df
        
    
    def is_wfc3_band(self, band, mode='IR'):
        
        if band is None:
            print(" Filter is set to None. Returning False.")
            return False
        else:
            u_band = band.upper()
            
        if mode is None:
            print(" None is not a valid WFC3 detector. Returning False.")
            return False
        else:
            u_mode = mode.upper()
        
        self.band = u_band
        self.mode = u_mode
        
        valid_modes = ['IR', 'UVIS']
        
        if u_mode not in valid_modes:
            print(f" The requested detector {u_mode} is not a valid WFC3 " +
                  " observing detector. Returing False.")
            return False
        
        if u_mode == 'IR':
            valid_filters = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W', 
                             'F098M', 'F127M', 'F139M', 'F153M', 
                             'F126N', 'F128N', 'F130N', 'F132N', 'F164N', 
                             'F167N']
            if u_band in valid_filters:
                yes_no = True
            else:
                yes_no = False
        elif u_mode == 'UVIS':
            valid_filters = ['Add uvis filters here.']
            if u_band in valid_filters:
                yes_no = True
            else:
                yes_no = False
            
        return yes_no
    
    def validate_df(self, input_df, band=None):
        
        if band is None:
            msg = " FILTER/band is undefined. Will look in the data "
            msg = msg + "header for FILTER value."
            print(msg)
            if not 'FILTER' in input_df.columns:
                sys.exit(" FILTER column is not found in the input " +
                      " dataframe. Exiting ......")
            else:
                barr = np.unique(input_df['FILTER'])
                print(f" Found {barr} filter/s in the input data.")
                if len(barr) > 1:
                    print(" Multiple filter values found in the input " +
                          "data. Processing only {barr[0]} filter.")
            
            band = barr[0]
        
        if self.is_wfc3_band(band):
            self.band = band
        else:
            err_msg = f" Filter, {band}, is not a part of MIRI imager "
            err_msg = err_msg + " filter suite."
            sys.exit(err_msg)
        
        orig_len = input_df.shape[0]
        
        tmp_df = input_df.copy()
        
        tmp_df = tmp_df.drop(tmp_df[tmp_df['INSTRUME']!='WFC3'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['OBSTYPE']!='IMAGING'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['DETECTOR']!='IR'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['FILTER']!=band].index)
        
        new_len = tmp_df.shape[0]
        
        if orig_len != new_len and orig_len > new_len:
            print(f"{orig_len - new_len} invlid datafiles were excluded " +
                  "from procesing.")
        elif new_len > orig_len:
            print(f" Original number of files: {orig_len}")
            print(f" After validate number of files: {new_len}")
            sys.exit(" Something is wrong here. Validation added " +
                     " additional data.")
            
        self.df = tmp_df.copy()
        
        return self.df
        
    
    def get_delta_threshold(self, band):
        if band is None:
            print(" Checking for valid MIRI filter.....")
            if self.band is None:
                sys.exit(" MIRI filter undefined.")
            else:
                u_band = self.band
        else:
            u_band = band
            
        if not self.is_wfc3_band(u_band):
            print(f" {u_band} is not a valid MIRI filter/band.")
            sys.exit(" Exiting ......")
        
        delta_thresholds = {'F105W': 0, 'F110W': 0, 'F125W': 0, 'F140W': 10, 
                            'F160W': 0, 'F098M': 0, 'F127M': 0, 'F139M': 0, 
                            'F153M': 0, 'F126N': 0, 'F128N': 0, 'F130N': 0, 
                            'F132N': 0, 'F164N': 0, 'F167N': 0}
        
        thres = delta_thresholds[band]
        
        return thres
        
    def get_stats(self, in_data):
        stats = {'Min': [], 'Max': [], 'Mean': [], 'Mean Unc': [], 
                 'Median': [], 'StdDev': [], 'Mode': [], 'Var': [], 
                 'Skew': [], 'Kurt': []}
        
        return stats
    
    
    def generate_flat(self, input_df, mask=None, dthres=None, method="mean",
                      outfile=None, save=False):
        #place holder 
        outflat = []
        
        
        return outflat