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
        
    
    def validate_df(self, input_df):
        
        return self.df
    
    
    def get_mask(self, m_file, mask_out=None, edge=None, save=False):
        # Place holder
        flat_mask = []
        
        return flat_mask
        
    
    def get_delta_threshold(self, band):
        # Place holder
        thres = []
        
        return thres
        
    
    def generate_flat(self, input_df, mask=None, dthres=None, method="mean",
                      outfile=None, save=False):
        #place holder 
        outflat = []
        
        
        return outflat