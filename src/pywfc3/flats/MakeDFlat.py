#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:34:35 2025

@author: sshenoy

This is the class that can be used to run the procedures list in WFC3 ISR 
2021-10 to generate a WFC3 IR D-Flat.
"""

import sys
import yaml
import logging
import warnings
import subprocess
import multiprocessing
import importlib.resources

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from scipy import stats
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from pywfc3 import utils 

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self, cl_args=None):
        
        if cl_args is not None:
            self.params = self.get_pipeline_params(cl_args)
            
        self.df = None
        
        self._setup_main_logger()
        self._init_step_loggers()
        
        
    
    def get_pipeline_params(self, cl_args):
        
        params = {}
        
        # 1. Start with Default JSON
        try:
            default_param_dir = importlib.resources.files('pywfc3.parameters')
            default_param_file = default_param_dir.joinpath('dflat.yaml')
            with importlib.resources.as_file(default_param_file) as p_file:
                try:
                    with open(p_file, 'r', encoding='utf-8') as f:
                        params.update(yaml.safe_load(f))
                except yaml.YAMLError as exc:
                    warn_msg = "ERROR reading default YMAL file.\n{exc}"
                    self.logger.info(warn_msg)
                    warnings.warn(warn_msg)
        except ModuleNotFoundError:
            warn_msg = "WARNING: Could not find pywfc3.parameters module" + \
                "Default parameters not loaded."
            self.logger.info(warn_msg)
            warnings.warn(warn_msg)
            
        
        
        # 2. Layer User-Provided JSON (Medium Priority)
        if cl_args.yamlfile is not None:
            usr_cfg_file = Path(cl_args.yamlfile).resolve()
            if usr_cfg_file.is_file():
                try:
                    with open(usr_cfg_file, 'r', encoding='utf-8') as f:
                        # params.update(json.load(f))
                        params.update(yaml.safe_load(f))
                # except json.JSONDecodeError as e:
                except yaml.YAMLError as exc:
                    warn_msg = "ERROR reading user parameter file "
                    f"{usr_cfg_file.name}: \n{exc}"
                    self.logger.info(warn_msg)
                    warnings.warn(warn_msg)
                    
        
        # 3. Layer CLI Overrides (Highest Priority)
        # Only update keys if the user actually passed them via command line
        for key, value in vars(cl_args).items():
            if value is not None:
                cl_overrides = {key: value}
                params['files'].update(cl_overrides)
        
        return params
    
    
    def _setup_main_logger(self):
        """Sets up the unique session master log and console output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not hasattr(self, "params"):
            self.params = {}
            log_directory = Path('./logs')
            log_directory.mkdir(parents=True, exist_ok=True)
            self.params['paths']['logdir'] = log_directory
            
        #Check if log directory exists, if not create it.
        log_directory = Path(self.params['paths']['logdir']).resolve()
        if not log_directory.is_dir():
            log_directory.mkdir(parents=True, exist_ok=True)
        
        self.main_log_path =  log_directory / f"dflat_{timestamp}.log"
        
        self.logger = logging.getLogger("DFlat_Pipe")
        self.logger.setLevel(logging.INFO)
        
        # Clear old handlers (crucial for Jupyter)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        fmt = logging.Formatter('%(name)s %(asctime)s: %(message)s', 
                                datefmt='%H:%M:%S')
        
        # Handler 1: Master File
        fh = logging.FileHandler(self.main_log_path)
        self.logger.addHandler(fh)
        
        # Handler 2: Console (Screen)
        ch = logging.StreamHandler()
        self.logger.addHandler(ch)
        
        # Print start pipeline message.
        self.logger.info("\n \t Starting DFlat pipeline.\n")
        
        # Now set the correct format.
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
            
        
    def _init_step_loggers(self):
        """Initializes empty loggers for CRDS and CALWF3."""
        self.logger_crds = logging.getLogger("DFlat_CRDS")
        self.logger_crds.setLevel(logging.DEBUG)
        
        self.logger_cal = logging.getLogger("DFlat_CALWF3")
        self.logger_cal.setLevel(logging.DEBUG)
        
        
        
    def _update_step_handler(self, logger_obj, new_log_path):
        """Swaps the file target for CRDS or CALWF3 logs."""
        for handler in logger_obj.handlers[:]:
            handler.close()
            logger_obj.removeHandler(handler)
        
        fmt = logging.Formatter('[%(name)s] [%(asctime)s] %(message)s', 
                                datefmt='%H:%M:%S')
        fh = logging.FileHandler(new_log_path)
        fh.setFormatter(fmt)
        logger_obj.addHandler(fh)
    
    
    
    def setup_directories(self):
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
        
        if hasattr(self, 'params'):
            # Check for input directory
            inpath = utils.check_directory(self.params['paths']['input'])
            self.params['paths']['input'] = str(inpath)
            self.logger.info(f"Found Input Directory: \n    {inpath}\n")
            
            # Check for data directory
            datadir = utils.check_directory(self.params['paths']['data'], 
                                            data_dir=True)
            self.params['paths']['data'] = str(datadir)
            self.logger.info(f"Found Data Directory: \n    {datadir}\n")
            
            # Set output directory
            if self.params['processing']['save']:
                # Set output directory
                outpath = utils.get_output_directory(name=self.
                                                     params['paths']['output'])
                self.logger.info("Setting Output Directory: "
                                 f"\n    {outpath}\n") 
                self.params['paths']['output'] = str(outpath)
                
                # Set CSV directory
                csvdir = Path(self.params['paths']['output']) / 'csv_files'
                csvpath = utils.get_output_directory(name=csvdir)
                self.logger.info("Setting CSV Directory: "
                                 f"\n    {csvpath}\n")
                self.params['paths']['csvdir'] = str(csvpath)
            
            
            
    def read_manifest(self, manifest):
        """ Read the input manifest and generate a list of files
        to process."""
        
        self.logger.info("Reading input manifest: ")
        try:
            input_manifest = Path(manifest).resolve(strict=True)
            
            with input_manifest.open() as f:
                flist = [line.strip() for line in f if not line.lstrip().startswith("#")]
            
            if len(flist) == 0:
                sys.exit(f"    Input manifest, {manifest}, is empty.")
            else:
                flist.sort()
                filelist = flist
            
            self.params['files']['manifest'] = str(input_manifest)
        except FileNotFoundError():
            sys.exit(f"    Input manifest, {manifest}, does not exist.")
            
        self.params['files']['input'] = filelist
        
        self.logger.info(f"    Found {len(filelist)} files.\n")
        
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
                self.logger.info("WARNING: File maybe missing or corrupted.")
                self.logger.info(f"WARNING: {error}\n")
        
        input_df = pd.DataFrame(meta_data)
        
        self.df = input_df
        
        return self.df
        
    
    
    def is_wfc3_band(self, band, mode='IR'):
        
        if band is None:
            self.logger.info("Filter is set to None. Returning False.\n")
            return False
        else:
            u_band = band.upper()
            
        if mode is None:
            self.logger.info("None is not a valid WFC3 detector. Returning False.\n")
            return False
        else:
            u_mode = mode.upper()
        
        self.band = u_band
        self.mode = u_mode
        
        valid_modes = ['IR', 'UVIS']
        
        if u_mode not in valid_modes:
            self.logger.info(f"The requested detector {u_mode} is not a "
                             "valid WFC3 detector. Returing False.\n")
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
            msg = "FILTER/band is undefined. Will look in the data "
            msg = msg + "header for FILTER value.\n"
            self.logger.info(msg)
            if not 'FILTER' in input_df.columns:
                sys.exit("FILTER column is not found in the input " +
                      " dataframe. Exiting ......\n")
            else:
                barr = np.unique(input_df['FILTER'])
                self.logger.info(f"Found {barr} filter/s in the input data.\n")
                if len(barr) > 1:
                    self.logger.info("Multiple filter values found in the "
                                     f"input data. Processing only {barr[0]} "
                                     "filter.\n")
            
            band = barr[0]
        
        if self.is_wfc3_band(band):
            self.band = band
        else:
            err_msg = f"Filter, {band}, is not a part of MIRI imager " + \
                        "filter suite.\n"
            sys.exit(err_msg)
        
        orig_len = input_df.shape[0]
        
        tmp_df = input_df.copy()
        
        tmp_df = tmp_df.drop(tmp_df[tmp_df['INSTRUME']!='WFC3'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['OBSTYPE']!='IMAGING'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['DETECTOR']!='IR'].index)
        tmp_df = tmp_df.drop(tmp_df[tmp_df['FILTER']!=band].index)
        
        new_len = tmp_df.shape[0]
        
        if orig_len != new_len and orig_len > new_len:
            self.logger.info(f"{orig_len - new_len} invlid datafiles were "
                             "excluded from procesing.\n")
        elif new_len > orig_len:
            self.logger.info(f"Original number of files: {orig_len}")
            self.logger.info(f"After validate number of files: {new_len}")
            sys.exit("Something is wrong here. Validation added " +
                     "additional data.\n")
            
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
        
        filename, ext_id, thresholds = args
        
        file_stats = self._stats_meta()
        
        file_stats['FILENAME'] = Path(filename).name
        
        if (thresholds is None) or (len(thresholds) < 3):
            outlier_sigma, \
            bias_sigam, \
            source_sigma = self.params['processing']['thresholds']
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
        
        masked_path = Path(self.params['paths']['output'])
        masked_filename = Path(filename).name.replace('flt', 'msk')
        masked_fullpath = masked_path / masked_filename
        
        file_stats['MaskedFile'] = masked_fullpath
        
        if self.params['processing']['save']:
            hdul.writeto(masked_fullpath, overwrite=True)
        
        hdul.close()
        
        return file_stats
    
    
    
    def mask_outliers(self, clean_df, thresholds=None, ncores=1):
        
        if (thresholds is None) or (len(thresholds)<3):
            thresholds = self.params['processing']['thresholds']
        
        if 'FILEPATH' in clean_df.columns:
            ext_id = 'sci'
            worker_args = [(file, ext_id, thresholds) \
                           for file in clean_df['FILEPATH']]
            
        with multiprocessing.Pool(processes=ncores) as pool:
            masked_data = pool.map(self.mask_single_file, worker_args)
        
        masked_df = pd.DataFrame(list(masked_data))
        
        merged_df = pd.merge(clean_df, masked_df, on='FILENAME', how='left')
        
        if self.params['processing']['save']:
            csv_path = Path(self.params['paths']['csvdir']).resolve()
            csv_name = self.params['instrument']['filter'] + "_mask_stat.csv"
            merged_df.to_csv(csv_path / csv_name, index=False)
            
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