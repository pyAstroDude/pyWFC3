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
import multiprocessing
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
        self.out_params = None
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
    
    
    def get_pipeline_params(self, cl_args):
        
        params = {}
        print("\n Generating Pipeline Parameters.\n")
        
        
        # 1. Start with Default JSON
        try:
            default_param_dir = importlib.resources.files('pywfc3.parameters')
            default_param_file = default_param_dir.joinpath('dflat.json')
            with importlib.resources.as_file(default_param_file) as p_file:
                try:
                    with open(p_file, 'r') as f:
                        params.update(json.load(f))
                except json.JSONDecodeError as e:
                    warnings.warn(" WARNING: Could not decode default\n" + 
                                  f" parameter file {p_file}: \n{e}")
        except ModuleNotFoundError:
            warnings.warn(" WARNING: Could not find pywfc3.parameters module.\n" + 
                          " Default parameters not loaded.")
    
        # 2. Layer User-Provided JSON (Medium Priority)
        if cl_args.config is not None:
            usr_cfg_file = Path(cl_args.config).resolve()
            if usr_cfg_file.is_file():
                try:
                    with open(usr_cfg_file, 'r') as f:
                        params.update(json.load(f))
                except json.JSONDecodeError as e:
                    warnings.warn(" WARNING: Could not decode user's\n" + 
                                  f" parameter file {usr_cfg_file.name}:\n{e}")
    
        # 3. Layer CLI Overrides (Highest Priority)
        # Only update keys if the user actually passed them via command line
        for key, value in vars(cl_args).items():
            if value is not None and key != 'config':
                cl_overrides = {key: value}
                
        params.update(cl_overrides)
        
        self.params = params
        self.out_params = params.copy()
    
        return params
    
    
    def setup_directories(self, dir_name, is_input=False, is_data=False,
                          is_output=False):
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
        
        # Set input (working) directory
        if is_input:
            self.inpath = utils.check_directory(dir_name)
            self.out_params['InputDirectory'] = str(self.inpath)
            print(f" Found Input Directory: \n {self.inpath}\n")
        
        # Set up data directory.
        if is_data:
            self.datadir = utils.check_directory(dir_name, data_dir=True)
            self.out_params['DataDirectory'] = str(self.datadir)
            print(f" Found Data Directory: \n {self.datadir}\n")
        
        # Set up output directory.
        if is_output:
            if dir_name is not None:
                out_name = Path(dir_name) / self.params['Filter']
            else:
                out_name = Path('./proc') / self.params['Filter']
                
            self.outpath = utils.get_output_directory(name=out_name)
            
            print(f" Setting Output Directory: \n {self.outpath}\n")  
            self.out_params['OutputDirectory'] = str(self.outpath)
            
        
        
    def read_manifest(self, manifest):
        """ Read the input manifest and generate a list of files
        to process."""
        
        print(" Reading input manifest: ")
        try:
            input_manifest = Path(manifest).resolve(strict=True)
            
            with input_manifest.open() as f:
                flist = [line.strip() for line in f if not line.lstrip().startswith("#")]
            
            if len(flist) == 0:
                sys.exit(f"    Input manifest, {manifest}, is empty.")
            else:
                flist.sort()
                self.filelist = flist
            
            self.manifest = input_manifest
        except FileNotFoundError():
            sys.exit(f"    Input manifest, {manifest}, does not exist.")
            
        self.params['InputFiles'] = self.filelist
        
        print(f"    Found {len(self.filelist)} files.\n")
        
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
                print(" WARNING: File maybe missing or corrupted.")
                print(f" WARNING: {error}\n")
        
        input_df = pd.DataFrame(meta_data)
        
        self.df = input_df
        
        return self.df
        
    
    def is_wfc3_band(self, band, mode='IR'):
        
        if band is None:
            print(" Filter is set to None. Returning False.\n")
            return False
        else:
            u_band = band.upper()
            
        if mode is None:
            print(" None is not a valid WFC3 detector. Returning False.\n")
            return False
        else:
            u_mode = mode.upper()
        
        self.band = u_band
        self.mode = u_mode
        
        valid_modes = ['IR', 'UVIS']
        
        if u_mode not in valid_modes:
            print(f" The requested detector {u_mode} is not a valid WFC3 " +
                  " detector. Returing False.\n")
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
            msg = msg + "header for FILTER value.\n"
            print(msg)
            if not 'FILTER' in input_df.columns:
                sys.exit(" FILTER column is not found in the input " +
                      " dataframe. Exiting ......\n")
            else:
                barr = np.unique(input_df['FILTER'])
                print(f" Found {barr} filter/s in the input data.\n")
                if len(barr) > 1:
                    print(" Multiple filter values found in the input " +
                          "data. Processing only {barr[0]} filter.\n")
            
            band = barr[0]
        
        if self.is_wfc3_band(band):
            self.band = band
        else:
            err_msg = f" Filter, {band}, is not a part of MIRI imager "
            err_msg = err_msg + " filter suite.\n"
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
                  "from procesing.\n")
        elif new_len > orig_len:
            print(f" Original number of files: {orig_len}")
            print(f" After validate number of files: {new_len}")
            sys.exit(" Something is wrong here. Validation added " +
                     " additional data.\n")
            
        self.df = tmp_df.copy()
        
        return self.df
        
    
    @classmethod    
    def _get_stats(cls, filename, thresholds=None):
        stats = {'Filename': [], 'TotPix': [], 'NegCnt': [], 'Min': [], 
                 'Max': [], 'Mean': [], 'MeanUnc': [], 'Var': [], 
                 'Skew': [], 'Kurt': [], 'Mode': [], 'SCS_Sigma': [], 
                 'SCS_Mean': [], 'SCS_Median': [], 'SCS_Std': [], 
                 'UpSigma': [], 'LoSigma': [], 'HiThres': [], 
                 'LoThres':[], 'NanCount': [], 'GoodCount':[], '%Good': []}
        
        
        
        return stats
    
    
    def mask_outlliers(self, clean_df, thresholds=None, ncores=5):
        
        if thresholds is None:
            thresholds = [2.0, 5.0, 2.0]
            
        if 'FILEPATH' in clean_df.columns:
            worker_args = [(file, thresholds) for file in clean_df['FILEPATH']]
            
        with multiprocessing.Pool(processes=ncores) as pool:
            masked_file = pool.map(self._get_stats, worker_args)
            
        masked_df = clean_df.copy()
        masked_df['MaskedFile'] = masked_file
        
        return masked_df
        
        
    def generate_flat(self, input_df, mask=None, dthres=None, method="mean",
                      outfile=None, save=False):
        #place holder 
        outflat = []
        
        
        return outflat