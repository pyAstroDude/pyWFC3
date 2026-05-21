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
from scipy import stats
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from pywfc3 import utils 

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self):
        
        self.params = None
        self.df = None
        
        
        
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
                                  f" parameter file {p_file}: \n{ e}")
        except ModuleNotFoundError:
            warnings.warn(" WARNING: Could not find pywfc3.parameters module.\n" + 
                          " Default parameters not loaded.")
        
        # 2. Layer User-Provided JSON (Medium Priority)
        if cl_args.ParamFile is not None:
            usr_cfg_file = Path(cl_args.ParamFile).resolve()
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
            if value is not None:
                cl_overrides = {key: value}
                params.update(cl_overrides)
        
        self.params = params
    
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
            inpath = utils.check_directory(dir_name)
            self.params['InputDirectory'] = str(inpath)
            print(f" Found Input Directory: \n    {inpath}\n")
        
        # Set up data directory.
        if is_data:
            datadir = utils.check_directory(dir_name, data_dir=True)
            self.params['DataDirectory'] = str(datadir)
            print(f" Found Data Directory: \n    {datadir}\n")
        
        # Set up output directory.
        if is_output:
            if dir_name is not None:
                out_name = Path(dir_name) / self.params['Filter']
            else:
                out_name = Path('./proc') / self.params['Filter']
                
            outpath = utils.get_output_directory(name=out_name)
            
            print(f" Setting Output Directory: \n    {outpath}\n") 
            self.params['OutputDirectory'] = str(outpath)
            
        
        
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
                filelist = flist
            
            self.params['Manifest'] = str(input_manifest)
        except FileNotFoundError():
            sys.exit(f"    Input manifest, {manifest}, does not exist.")
            
        self.params['InputFiles'] = filelist
        
        print(f"    Found {len(filelist)} files.\n")
        
        return filelist
    
    
    
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
        
    
    
    @staticmethod
    def _stats_meta():
        stats_hdr = {'FILENAME': None, 'TotPix': None, 'NegCnt': None, 
                     'Min': None, 'Max': None, 'Mean': None, 'MeanUnc': None, 
                     'Var': None, 'Skew': None, 'Kurt': None, 'Mode': None, 
                     'OutlierSigma': None, 'SCS_Mean': None, 
                     'SCS_Median': None, 'SCS_Std': None, 'BiasSigma': None, 
                     'SourceSigma': None, 'BiasThres': None, 
                     'SourceThres':None, 'NanCount': None, 'GoodCount':None, 
                     '%Good': None}
        
        return stats_hdr
    
        
    
    def mask_single_file(self, args):
        
        filename, ext_id, thresholds, save = args
        
        file_stats = self._stats_meta()
        
        file_stats['FILENAME'] = Path(filename).name
        
        if (thresholds is None) or (len(thresholds) < 3):
            outlier_sigma, bias_sigam, source_sigma = self.params['Thresholds']
        else:
            outlier_sigma, bias_sigam, source_sigma = thresholds
    
        file_stats['OutlierSigma'] = outlier_sigma
        file_stats['BiasSigma'] = bias_sigam
        file_stats['SourceSigma'] = source_sigma 
    
        hdul = fits.open(filename)
        
        data_stats = stats.describe(hdul['sci'].data, axis=None)
        sem = stats.sem(hdul['sci'].data, axis=None)
        mode = stats.mode(hdul['sci'].data, axis=None).mode
        men, med, std = sigma_clipped_stats(hdul['sci'].data, sigma=outlier_sigma)
        
        bias_thre = med - bias_sigam * std
        source_thre = med + source_sigma * std
        
        bias_mask = (hdul['sci'].data < bias_thre)
        source_mask = (hdul['sci'].data > source_thre)
        
        hdul['sci'].data[bias_mask] = np.nan
        hdul['sci']. data[source_mask] = np.nan
        
        hdul['err'].data[bias_mask] = np.nan
        hdul['err']. data[source_mask] = np.nan
        
        hdul['dq'].data[:, :] = 0
        
        hdul['dq'].data[bias_mask] = 1
        hdul['dq']. data[source_mask] = 1
        
        nan_cnt = np.count_nonzero(np.isnan(hdul['sci'].data))
        good_cnt = np.count_nonzero(~np.isnan(hdul['sci'].data))
    
        per_good = 100 * good_cnt / data_stats.nobs
        
        file_stats['TotPix'] = data_stats.nobs
        file_stats['NegCnt'] = (hdul['sci'].data < 0).sum()
        file_stats['Min'] = data_stats.minmax[0]
        file_stats['Max'] = data_stats.minmax[1]
        file_stats['Mean'] = data_stats.mean
        file_stats['Var'] = data_stats.variance
        file_stats['Skew'] = data_stats.skewness
        file_stats['Kurt'] = data_stats.kurtosis
        
        file_stats['MeanUnc'] = sem
        
        file_stats['Mode'] = mode
    
        file_stats['SCS_Mean'] = men
        file_stats['SCS_Median'] = med
        file_stats['SCS_Std'] = std
        
        file_stats['BiasThres'] = bias_thre
        file_stats['SourceThres'] = source_thre
        
        file_stats['NanCount'] = nan_cnt
        file_stats['GoodCount'] = good_cnt
        file_stats['%Good'] = per_good
        
        masked_path = Path(self.params['OutputDirectory'])
        masked_filename = Path(filename).name.replace('flt', 'msk')
        masked_fullpath = masked_path / masked_filename
        
        file_stats['MaskedFile'] = masked_fullpath
        
        if save:
            hdul.writeto(masked_fullpath, overwrite=True)
        
        hdul.close()
        
        return file_stats
    
    
    
    def mask_outliers(self, clean_df, thresholds=None, ncores=1):
        
        if (thresholds is None) or (len(thresholds)<3):
            thresholds = self.params['Thresholds']
        
        if 'FILEPATH' in clean_df.columns:
            ext_id = 'sci'
            save = self.params['Save']
            worker_args = [(file, ext_id, thresholds, save) \
                           for file in clean_df['FILEPATH']]
            
        with multiprocessing.Pool(processes=ncores) as pool:
            masked_data = pool.map(self.mask_single_file, worker_args)
        
        masked_df = pd.DataFrame(list(masked_data))
        
        merged_df = pd.merge(clean_df, masked_df, on='FILENAME', how='left')
            
        self.df = merged_df
        
        return self.df
        
    
    
    @classmethod
    def _run_single_calw3(cls, args): 
        unflattened_flt = []
        
        # FLATCORR = 'OMIT"
        
        return unflattened_flt
    
    
    
    def run_calw3_pipe(self, input_df):
        
        # multiprocess with N cores.
        
        return # unflattened_df
    
    
    
    def update_mask(self, unlaflattened_df):
        
        # Read DQ from unflattend, add DQ from MSK 
        
        return # unflatted_mask_df
    
    
    
    def generate_flat(self, input_df, mask=None, dthres=None, method="mean",
                      outfile=None, save=False):
        
        #  ensure the combined mask is used. for stacking.
        
        
        outflat = []
        
        
        return outflat
    
    
    
    def update_dflat(self, input_df):
        
        # divide the outflat with pflat.
        
        # add/update the old dflat with new blobs.
        
        return