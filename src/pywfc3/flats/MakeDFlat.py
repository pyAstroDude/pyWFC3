#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:34:35 2025

@author: sshenoy

This is the class that can be used to run the procedures list in WFC3 ISR 
2021-10 to generate a WFC3 IR D-Flat.
"""

import os
import sys
import yaml
import shutil
import logging
import warnings
import contextlib
import subprocess
import multiprocessing
from importlib import resources

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from scipy import stats
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

import crds
from wfc3tools import calwf3
from astroquery.mast import Observations

from pywfc3 import utils 

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self, cl_args=None):
        
        if cl_args is not None:
            self.params = self.get_pipeline_params(cl_args)
            
        self.df = None
        
        self._setup_main_logger()
        self._init_step_loggers()
        
        # Set class-level attributes for multiprocessing safety
        self.__class__.params = self.params
        self.__class__.logger = self.logger
        self.__class__.logger_cal = self.logger_cal
        
        
    
    def get_pipeline_params(self, cl_args):
        
        params = {}
        
        # 1. Start with Default JSON
        try:
            default_param_dir = resources.files('pywfc3.parameters')
            default_param_file = default_param_dir.joinpath('dflat.yaml')
            with resources.as_file(default_param_file) as p_file:
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
        
        
        
    @staticmethod
    def _update_step_handler(logger_obj, new_log_path):
        """Swaps the file target for CRDS or CALWF3 logs."""
        for handler in logger_obj.handlers[:]:
            handler.close()
            logger_obj.removeHandler(handler)
        
        fmt = logging.Formatter('[%(name)s] [%(asctime)s] %(message)s', 
                                datefmt='%H:%M:%S')
        fh = logging.FileHandler(new_log_path)
        fh.setFormatter(fmt)
        logger_obj.addHandler(fh)
        
    @classmethod
    def _init_worker(cls, params):
        """Initializes class attributes in the pool worker processes."""
        cls.params = params
        cls.logger = logging.getLogger("DFlat_Pipe")
        cls.logger_cal = logging.getLogger("DFlat_CALWF3")
    
    
    
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
                self.params['paths']['output'] = outpath
                
                # Set CSV directory
                csvdir = Path(self.params['paths']['output']) / 'csv_files'
                csvpath = utils.get_output_directory(name=csvdir)
                self.logger.info("Setting CSV Directory: "
                                 f"\n    {csvpath}\n")
                self.params['paths']['csvdir'] = csvpath
            
            
            
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
        
        # Compute descriptive statistics on the SCI extension
        data_stats = stats.describe(hdul['SCI', 1].data, axis=None)
        sem = stats.sem(hdul['SCI', 1].data, axis=None)
        mode = stats.mode(hdul['SCI', 1].data, axis=None).mode
        men, med, std = sigma_clipped_stats(hdul['SCI', 1].data,
                                            sigma=outlier_sigma)
        
        bias_thre = med - bias_sigam * std
        source_thre = med + source_sigma * std
        
        # Generate boolean masks for outlier pixels
        bias_mask = (hdul['SCI', 1].data < bias_thre)
        source_mask = (hdul['SCI', 1].data > source_thre)
        
        # Apply NaN masking to SCI and ERR extensions
        hdul['SCI', 1].data[bias_mask] = np.nan
        hdul['SCI', 1].data[source_mask] = np.nan
        
        hdul['ERR', 1].data[bias_mask] = np.nan
        hdul['ERR', 1].data[source_mask] = np.nan
        
        # Reset and update the DQ extension with outlier flags
        hdul['DQ', 1].data[:, :] = 0
        
        hdul['DQ', 1].data[bias_mask] = 1
        hdul['DQ', 1].data[source_mask] = 1
        
        nan_cnt = np.count_nonzero(np.isnan(hdul['SCI', 1].data))
        good_cnt = np.count_nonzero(~np.isnan(hdul['SCI', 1].data))
    
        per_good = 100 * good_cnt / data_stats.nobs
        
        file_stats['TotPix'] = data_stats.nobs
        file_stats['NegCnt'] = (hdul['SCI', 1].data < 0).sum()
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
        
        file_stats['MaskedFile'] = str(masked_fullpath)
        
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
        
    
    
    def get_raw_data(self, masked_df, max_retries=3):
        
        raw_df = masked_df.copy()
        
        # 1. Generate RawFile strings (Assumes MaskedFile is already str)
        raw_df['RawFile'] = raw_df['MaskedFile'].str.replace('msk', 
                                                             'raw', 
                                                             case=False)
        
        # 2. Get unique files to avoid redundant network/disk checks
        unique_raw_files = raw_df['RawFile'].unique()
        
        # 3. Retry loop
        for attempt in range(1, max_retries + 1):
            # Identify what is currently missing on disk
            missing_files = [f for f in unique_raw_files \
                             if not Path(f).exists()]
            
            if not missing_files:
                self.logger.info("Verification Complete: "
                                      "All raw files are downloaded.")
                break
            
            self.logger.info(f"Attempt {attempt}/{max_retries}: "
                                  f"{len(missing_files)} files missing. "
                                  "Starting download...")
            
            for raw_path_str in missing_files:
                raw_path_obj = Path(raw_path_str)
                mast_url = f"mast:hst/product/{raw_path_obj.name}"
                
                try:
                    Observations.download_file(
                        mast_url, 
                        local_path=str(raw_path_obj), 
                        cache=True
                    )
                except Exception as e:
                    self.logger.error("Download failed for "
                                           f"{raw_path_obj.name}: {e}")
        
        # 4. Final check after all retries
        final_missing = [f for f in unique_raw_files if not Path(f).exists()]
        if final_missing:
            self.logger.warning("Final Count Mismatch: "
                                     f"{len(final_missing)} files still "
                                     f"missing after {max_retries} attempts.")
        
        self.df = raw_df
        
        return self.df
                
        
        
    @classmethod
    def update_dummy_in_raw(cls, raw_file):
        """Updates PFLTFILE and DFLTFILE headers in raw files with dummy files."""
        
        wfc3_dum_files = {'F098M': {'pflat': '4ac1921ji_1s_pfl.fits', 
                                    'dflat':'4ac1829li_1s_dfl.fits'}, 
                          'F105W': {'pflat': '4ac19225i_1s_pfl.fits', 
                                    'dflat':'4ac1816oi_1s_dfl.fits'}, 
                          'F110W': {'pflat': '4ac1921ri_1s_pfl.fits', 
                                    'dflat':'4ac18192i_1s_dfl.fits'}, 
                          'F125W': {'pflat': '4ac1921ii_1s_pfl.fits', 
                                    'dflat':'4ac1825si_1s_dfl.fits'}, 
                          'F140W': {'pflat': '4ac19224i_1s_pfl.fits', 
                                    'dflat':'4ac18187i_1s_dfl.fits'}, 
                          'F160W': {'pflat': '4ac1921li_1s_pfl.fits', 
                                    'dflat':'4ac18231i_1s_dfl.fits'}
                          }
        
        band = fits.getval(raw_file, 'FILTER', ext=0) 
        
        pflat_name = wfc3_dum_files[band]['pflat']
        dflat_name = wfc3_dum_files[band]['dflat']
        
        pflat_value = "nref$"+pflat_name
        dflat_value = "nref$"+dflat_name
            
        fits.setval(raw_file, 'PFLTFILE', value=pflat_value, ext=0)
        fits.setval(raw_file, 'DFLTFILE', value=dflat_value, ext=0)
        
        return raw_file, pflat_name, dflat_name
    
    
    
    @classmethod
    def _run_single_calw3(cls, args): 
        
        unflattened_flt = {'RawFile': None, 'FltFile': None, 'ImaFile': None}
        
        raw_file, pflat, dflat = cls.update_dummy_in_raw(args)
        
        nref_dir = os.environ.get('nref')
        pflat_path = Path(nref_dir) / pflat
        dflat_path = Path(nref_dir) / dflat
        
        ref_path = resources.files('pywfc3.data')
        
        if not pflat_path.is_file():
            pref_file = ref_path.joinpath(pflat)
            if pref_file.is_file():
                shutil.copy2(pref_file, pflat_path)
            else:
                cls.logger.info(f"Dummy reference flat {pflat} not found in "
                                 f"pref_file.parent")
        
        if not dflat_path.is_file():
            dref_file = ref_path.joinpath(dflat)
            if dref_file.is_file():
                shutil.copy2(dref_file, dflat_path)
            else:
                cls.logger.info(f"Dummy reference flat {dflat} not found in "
                                 f"rdef_file.parent")
            
        log_path = Path(raw_file).parent
        log_file = Path(raw_file).name.replace('fits', 'log')
        cls._update_step_handler(cls.logger_cal, 
                                 str(log_path / log_file))
        
        # Convert raw_file path to a relative path to respect calwf3's <95-char limit
        rel_raw_file = os.path.relpath(raw_file)
        cls.logger.info("MARK I: This is a test.")
        try:
            flt_path = raw_file.replace('_raw', '_flt')
            ima_path = raw_file.replace('_raw', '_ima')
            
            if not Path(flt_path).is_file() and not Path(ima_path).is_file():
                calwf3(rel_raw_file, save_tmp=True, 
                       verbose=True, log_func=cls.logger_cal.info)
            else:
                file_found_msg = f"""Found {Path(flt_path).name} or 
                    {Path(ima_path).name}. Check flat file keywords in the 
                    header to ensure dummy flat was used in the reduction."""
                cls.logger.info(file_found_msg)
                cls.logger_cal.info(file_found_msg)
        
            unflattened_flt['RawFile'] = raw_file
            unflattened_flt['FltFile'] = flt_path \
                if Path(flt_path).is_file() else None
            unflattened_flt['ImaFile'] = ima_path \
                if Path(ima_path).is_file() else None
        except Exception as e:
            cls.logger.error(f"calwf3 execution failed for {raw_file}: {e}")
            
            unflattened_flt['RawFile'] = raw_file
            unflattened_flt['FltFile'] = None
            unflattened_flt['ImaFile'] = None
            
        return unflattened_flt
    
    
    
    def run_calw3_pipe(self, input_df, outdir=None, ncores=1):
        
        if outdir is None:
            outpath = self.params['paths']['output']
        else:
            outpath = utils.get_output_directory(name=outdir)
            self.params['paths']['output'] = outpath
            
        # 1. Get unique parent paths from the RawFile column
        unique_parents = {Path(f).resolve().parent for f in input_df['RawFile']}
        
        # 2. Safety check: Ensure there is only one parent path
        if len(unique_parents) > 1:
            self.logger.warning("Multiple source directories detected: "
                                f"{unique_parents}")
        
        # Get the existing parent (using the first one found)
        current_parent = next(iter(unique_parents))
        
        # 3. If it doesn't match outpath, update the whole column
        if current_parent == Path(outpath).resolve():
            self.logger.info("Raw file paths match output directory.")
        else:
            self.logger.info(f"Updating raw file paths: {current_parent} "
                             f"-> {outpath}")
            
            # Reconstruct paths: outpath + filename
            input_df['RawFile'] = input_df['RawFile'].apply(
                lambda x: str(Path(outpath) / Path(x).name)
            )
            
            self.logger.info("Raw file column updated successfully.")
        
        # 4. Get dummy flats
        
        os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
        os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
        os.environ['CRDS_PATH'] = '/Users/sshenoy/crds_cache'
        
        iref_path = self.params['paths']['iref']
        if iref_path:
            iref_path = os.path.relpath(iref_path)
            if not iref_path.endswith('/'):
                iref_path += '/'
        os.environ['iref'] = iref_path
        
        new_ref_path = utils.get_output_directory(self.params['paths']['nref'])
        if new_ref_path:
            new_ref_path = os.path.relpath(new_ref_path)
            if not new_ref_path.endswith('/'):
                new_ref_path += '/'
        os.environ['nref'] = new_ref_path

        worker_args = [file for file in input_df['RawFile']]
        
        with multiprocessing.Pool(processes=ncores, 
                                  initializer=self.__class__._init_worker, 
                                  initargs=(self.params,)
                                  ) as pool:
            unflt_data = pool.map(self._run_single_calw3, worker_args)
        
        unflt_df = pd.DataFrame(list(unflt_data))
        
        merged_df = pd.merge(input_df, unflt_df, on='RawFile', how='left')
        
        if self.params['processing']['save']:
            csv_path = Path(self.params['paths']['csvdir']).resolve()
            csv_name = self.params['instrument']['filter'] + "_unflt.csv"
            merged_df.to_csv(csv_path / csv_name, index=False)
        
        return merged_df
    
    
    
    def update_mask(self, unflattened_df, save=None):
        """Combine DQ extensions from FltFile and MaskedFile into a
        single combined mask using bitwise OR.

        Parameters
        ----------
        unflattened_df : pd.DataFrame
            Output from run_calw3_pipe. Must contain 'FltFile' and
            'MaskedFile' columns.
        save : bool, optional
            If True, save the combined mask as a copy of the flt file
            with the DQ extension replaced. Defaults to
            self.params['processing']['save'].

        Returns
        -------
        tuple of (pd.DataFrame, dict)
            updated_df : DataFrame with 'CombMaskFile' column added
                (paths to *_cmf.fits files if save=True, else None).
            mask_dict : Dictionary mapping filename IDs
                (e.g. 'idjb10xvq_flt.fits') to combined DQ arrays.
        """

        if save is None:
            save = self.params['processing']['save']

        mask_dict = {}
        cmf_paths = []

        for idx, row in unflattened_df.iterrows():
            flt_file = row.get('FltFile')
            msk_file = row.get('MaskedFile')

            # Skip rows where calwf3 failed (FltFile is None)
            if flt_file is None or pd.isna(flt_file):
                self.logger.warning(
                    f"Skipping row {idx}: FltFile is None "
                    f"(MaskedFile: {msk_file}). calwf3 may have failed."
                )
                cmf_paths.append(None)
                continue

            # Skip rows where MaskedFile is also missing
            if msk_file is None or pd.isna(msk_file):
                self.logger.warning(
                    f"Skipping row {idx}: MaskedFile is None "
                    f"(FltFile: {flt_file})."
                )
                cmf_paths.append(None)
                continue

            flt_path = Path(flt_file)
            file_id = flt_path.name  # e.g., idjb10xvq_flt.fits

            # Read DQ extension from the unflattened flt file
            with fits.open(flt_file) as flt_hdul:
                flt_dq = flt_hdul['DQ', 1].data.copy()

            # Read DQ extension from the outlier-masked file
            with fits.open(msk_file) as msk_hdul:
                msk_dq = msk_hdul['DQ', 1].data.copy()

            # Validate that both DQ arrays have the same shape
            if flt_dq.shape != msk_dq.shape:
                self.logger.warning(
                    f"Skipping {file_id}: DQ shape mismatch "
                    f"(flt: {flt_dq.shape}, msk: {msk_dq.shape})."
                )
                cmf_paths.append(None)
                continue

            # Combine masks using bitwise OR
            combined_dq = np.bitwise_or(flt_dq, msk_dq)

            # Store combined mask in dictionary keyed by filename ID
            mask_dict[file_id] = combined_dq

            if save:
                # Save as a copy of flt with DQ replaced by combined mask
                cmf_name = flt_path.name.replace('_flt', '_cmf')
                cmf_fullpath = (Path(self.params['paths']['output'])
                                / cmf_name)

                with fits.open(flt_file) as cmf_hdul:
                    cmf_hdul['DQ', 1].data = combined_dq
                    cmf_hdul.writeto(cmf_fullpath, overwrite=True)

                cmf_paths.append(str(cmf_fullpath))
                self.logger.info(f"Combined mask saved: {cmf_name}")
            else:
                cmf_paths.append(None)

        # Add CombMaskFile column to a copy of the dataframe
        updated_df = unflattened_df.copy()
        updated_df['CombMaskFile'] = cmf_paths

        # Save updated dataframe to CSV if requested
        if save:
            csv_path = Path(self.params['paths']['csvdir']).resolve()
            csv_name = (self.params['instrument']['filter']
                        + "_cmf.csv")
            updated_df.to_csv(csv_path / csv_name, index=False)

        self.df = updated_df

        self.logger.info(
            f"Combined masks generated for {len(mask_dict)} files."
        )

        return self.df, mask_dict
    
    
    
    def generate_flat(self, input_df, method="mean"):
        
        if 'CombMaskFile' in input_df.columns:
            data_files = input_df['CombMaskFile']
        elif 'FltFile' in input_df.columns:
            data_files = input_df['FltFile']
        else:
            self.logger("Unflattened flat files not found.")
            sys.exit("Input files not found")
        
        sci_data = None
        err_data = None
        
        for i, fl in enumerate(data_files):
            print(fl)
            hdul = fits.open(fl)
            
            m_dat = fits.getdata(fl.replace('cmf', 'msk'), ext=('DQ', 1))
            mask = np.array(m_dat, dtype='bool')
            # mask = np.array(hdul['DQ', 1].data, dtype='bool')
            
            hdul['SCI', 1].data[mask] = np.nan
            hdul['ERR', 1].data[mask] = np.nan
            
            if sci_data is None:
                nx = hdul['SCI', 1].data.shape[0]
                ny = hdul['SCI', 1].data.shape[1]
                sci_data = np.zeros((len(input_df),
                                     hdul['SCI', 1].data.shape[0],
                                     hdul['SCI', 1].data.shape[1]))
            
            if err_data is None:
                err_data = np.zeros((len(input_df),
                                     hdul['ERR', 1].data.shape[0],
                                     hdul['ERR', 1].data.shape[1]))
            
            sci_mval = np.nanmean(hdul['SCI', 1].data[101:900, 101:900])
            err_mval = np.nanmean(hdul['ERR', 1].data[101:900, 101:900])
            
            sci_data[i, :, :] = hdul['SCI', 1].data / sci_mval 
            err_data[i, :, :] = hdul['ERR', 1].data / err_mval
            
            hdul.close()
        
        fits.writeto('quick_output.fits', sci_data, overwrite=True)
        
        with warnings.catch_warnings(action="ignore"):
            sf_mean, sf_median, sf_std = sigma_clipped_stats(sci_data, 
                                                         mask_value=np.nan,
                                                         sigma_lower=3, 
                                                         sigma_upper=3, 
                                                         axis=0)
        
        if method == "median":
            flat = sf_median
        else:
            flat = sf_mean
        
        flat_unc = sf_std / np.sqrt(len(sci_data)) #/ np.nanmedian(flat)
        
        
        ### Set DQ flag for pixels that exhibit AD_Floor.
        xpix, ypix = np.where(flat < 0.0)
        
        if len(xpix) >= 1:
            print(f"\n {len(xpix)} pixels had negative values. These pixels")
            print(" are NaN-ed in the SCI & ERR arrays and AD_Floor flag")
            print(" is set the DQ array.") 
            for x, y in zip(xpix, ypix):
                    flat[x, y] = np.nan
                    flat_unc[x, y] = np.nan
                    mask[x, y] = True
        else:
            print(" All pixels passed the AD_FLOOR test.")
            
        
        ### Get headers to save the falt.
        if self.params['processing']['save']:
            
            hdr = fits.getheader(fl, ext=0)
            
            phdu = fits.PrimaryHDU(header=hdr)
            
            sci_hdu = fits.ImageHDU(data=flat, name='SCI', ver=1)
            err_hdu = fits.ImageHDU(data=flat_unc, name='ERR', ver=1)
            flt_mask = np.zeros_like(sci_hdu.data, dtype=np.int16)
            msk_hdu = fits.ImageHDU(data=flt_mask, name='DQ', ver=1)
            
            flat_hdul = fits.HDUList([phdu, sci_hdu, err_hdu, msk_hdu])
            
            out_path = Path(self.params['paths']['output'])
            out_file = f"HST_WFC3_IR_{self.params['instrument']['filter']}" + \
            "_PFlat.fits"
            
            flat_hdul.writeto(str(out_path / out_file), overwrite=True)
            
        return flat, flat_unc, mask
    
    
    
    def update_dflat(self, pflat):
        
        # divide the outflat with pflat.
        
        # add/update the old dflat with new blobs.
        
        return