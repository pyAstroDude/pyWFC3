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

from wfc3tools import calwf3
from astroquery.mast import Observations

from pywfc3 import utils 

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    wfc3_dum_files = {'F098M': {'pflat': '4ac1921ji_1s_pfl.fits', 
                                'dflat': '4ac1829li_1s_dfl.fits'}, 
                      'F105W': {'pflat': '4ac19225i_1s_pfl.fits', 
                                'dflat': '4ac1816oi_1s_dfl.fits'}, 
                      'F110W': {'pflat': '4ac1921ri_1s_pfl.fits', 
                                'dflat': '4ac18192i_1s_dfl.fits'}, 
                      'F125W': {'pflat': '4ac1921ii_1s_pfl.fits', 
                                'dflat': '4ac1825si_1s_dfl.fits'}, 
                      'F140W': {'pflat': '4ac19224i_1s_pfl.fits', 
                                'dflat': '4ac18187i_1s_dfl.fits'}, 
                      'F160W': {'pflat': '4ac1921li_1s_pfl.fits', 
                                'dflat': '4ac18231i_1s_dfl.fits'}
                      }
    
    def __init__(self, cl_args=None):
        
        if cl_args is not None:
            self.params = self.get_pipeline_params(cl_args)
            
        self.df = None
        
        self._setup_main_logger()
        
        # Set class-level attributes for multiprocessing safety
        self.__class__.params = self.params
        self.__class__.logger = self.logger
        
        
    
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
        
        # 4. Expand tilde and environment variables for all paths
        if 'paths' in params:
            for key, val in params['paths'].items():
                if val and isinstance(val, str):
                    params['paths'][key] = os.path.expandvars(os.path.expanduser(val))
        
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
            
        
    @classmethod
    def _init_worker(cls, params):
        """Initializes class attributes in the pool worker processes."""
        cls.params = params
        cls.logger = logging.getLogger("DFlat_Pipe")
    
    
    
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
                
                # Define and create calwf3 calibration subdirectory early
                calwf3_dir = Path(outpath) / 'calwf3_pipe'
                calwf3_path = utils.get_output_directory(name=calwf3_dir)
                self.logger.info("Setting calwf3 Directory: "
                                 f"\n    {calwf3_path}\n")
                self.params['paths']['calwf3_dir'] = calwf3_path
                
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
                     'DATE-OBS': [],
                     'EXPSTART': []
                     }
        
        self.logger.info("Generating Pandas DataFrame ...")
        self.logger.info(f"Number of files to process: {len(filelist)}")
        self.logger.info("This may take a while if you are on VPN!")
        
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
        
        calwf3_dir = self.params['paths'].get('calwf3_dir')
        masked_path = Path(calwf3_dir) if calwf3_dir else Path(self.params['paths']['output'])
        masked_filename = Path(filename).name.replace('flt', 'msk')
        masked_fullpath = masked_path / masked_filename
        
        file_stats['Source Mask'] = str(masked_fullpath)
        
        if self.params['processing']['save']:
            hdul.writeto(masked_fullpath, overwrite=True)
        
        hdul.close()
        
        return file_stats
    
    
    
    def mask_outliers(self, clean_df, thresholds=None, ncores=1):
        
        self.logger.info("Masking outliers, bias and source pixels.")
        
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
        
    def _set_crds_env(self):
        """Configures CRDS environment variables for STScI servers and local cache."""
        os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
        os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
        os.environ['CRDS_PATH'] = os.path.expandvars('${HOME}/crds_cache')
        
    def update_crds_bestrefs(self, files):
        """Runs the CRDS bestrefs command to download and update best references for the FITS files."""
        # Ensure CRDS environment variables are configured
        self._set_crds_env()
        
        valid_files = [f for f in files if Path(f).is_file()]
        if not valid_files:
            return
            
        self.logger.info(f"Running CRDS bestrefs to download and update headers for {len(valid_files)} files...")
        
        # Chunk size of 200 files balances command line limits and CRDS initialization overhead
        chunk_size = 200
        for i in range(0, len(valid_files), chunk_size):
            chunk = valid_files[i:i + chunk_size]
            self.logger.info(f"Processing CRDS bestrefs chunk {i // chunk_size + 1} ({len(chunk)} files)...")
            
            try:
                cmd = ['crds', 'bestrefs', '--files'] + chunk + ['--sync-references=1', '--update-bestrefs']
                self.logger.info(f"Executing command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.info(f"Chunk {i // chunk_size + 1} processed successfully.")
                if result.stdout:
                    self.logger.info(f"CRDS stdout:\n{result.stdout}")
                if result.stderr:
                    self.logger.info(f"CRDS stderr:\n{result.stderr}")
            except Exception as e:
                self.logger.error(f"CRDS bestrefs failed on chunk {i // chunk_size + 1}: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    self.logger.error(f"CRDS error output:\n{e.stderr}")
    
    def get_raw_data(self, masked_df, max_retries=3):
        
        raw_df = masked_df.copy()
        
        # 1. Generate RAW with dummy strings (Assumes Source Mask is already str)
        raw_df['RAW with dummy'] = raw_df['Source Mask'].str.replace('msk', 
                                                                     'raw', 
                                                                     case=False)
        
        # 2. Get unique files to avoid redundant network/disk checks
        unique_raw_files = raw_df['RAW with dummy'].unique()
        
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
        
        # 5. Run CRDS bestrefs to update headers and download standard references
        self.update_crds_bestrefs(unique_raw_files)
        
        self.df = raw_df
        
        return self.df
                
        
        
    @classmethod
    def update_dummy_in_raw(cls, raw_file):
        """Updates PFLTFILE, DFLTFILE, and ASN_TAB headers in raw files."""
        
        band = fits.getval(raw_file, 'FILTER', ext=0) 
        
        pflat_name = cls.wfc3_dum_files[band]['pflat']
        dflat_name = cls.wfc3_dum_files[band]['dflat']
        
        pflat_value = "nref$"+pflat_name
        dflat_value = "nref$"+dflat_name
            
        fits.setval(raw_file, 'PFLTFILE', value=pflat_value, ext=0)
        fits.setval(raw_file, 'DFLTFILE', value=dflat_value, ext=0)
        fits.setval(raw_file, 'ASN_TAB', value='N/A', ext=0)
        
        return raw_file, pflat_name, dflat_name
    
    
    
    @classmethod
    def _run_single_calw3(cls, args): 
        
        unflattened_flt = {'RAW with dummy': None, 'Unflattened FLT': None, 'Unflattened IMA': None}
        
        raw_file, pflat, dflat = cls.update_dummy_in_raw(args)
        
        nref_dir = os.environ.get('nref')
        if nref_dir:
            pflat_path = Path(nref_dir) / pflat
            dflat_path = Path(nref_dir) / dflat
        else:
            pflat_path = Path(pflat)
            dflat_path = Path(dflat)
        
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
            
        # Convert raw_file path to a relative path to respect calwf3's <95-char limit
        rel_raw_file = os.path.relpath(raw_file)
        
        try:
            flt_path = raw_file.replace('_raw', '_flt')
            ima_path = raw_file.replace('_raw', '_ima')
            
            if not Path(flt_path).is_file() and not Path(ima_path).is_file():
                calwf3(rel_raw_file, save_tmp=True, 
                       verbose=True, log_func=cls.logger.info)
            else:
                file_found_msg = f"""Found {Path(flt_path).name} or 
                    {Path(ima_path).name}. Check flat file keywords in the 
                    header to ensure dummy flat was used in the reduction."""
                cls.logger.info(file_found_msg)
        
            unflattened_flt['RAW with dummy'] = raw_file
            unflattened_flt['Unflattened FLT'] = flt_path \
                if Path(flt_path).is_file() else None
            unflattened_flt['Unflattened IMA'] = ima_path \
                if Path(ima_path).is_file() else None
        except Exception as e:
            cls.logger.error(f"calwf3 execution failed for {raw_file}: {e}")
            
            unflattened_flt['RAW with dummy'] = raw_file
            unflattened_flt['Unflattened FLT'] = None
            unflattened_flt['Unflattened IMA'] = None
            
        return unflattened_flt
    
    
    
    def setup_calibration_env(self):
        """Sets up calibration environment variables (CRDS, iref, nref) and verifies that the reference directories exist."""
        # 1. CRDS environment setup
        self._set_crds_env()
        
        # 2. Process and set iref
        iref_path = self.params['paths']['iref']
        if iref_path:
            abs_path = str(Path(iref_path).resolve())
            if len(abs_path) < 95:
                iref_path = abs_path
            else:
                iref_path = os.path.relpath(iref_path)
            if not iref_path.endswith('/'):
                iref_path += '/'
        os.environ['iref'] = iref_path
        
        # 3. Process and set nref
        new_ref_path = utils.get_output_directory(self.params['paths']['nref'])
        if new_ref_path:
            abs_path = str(Path(new_ref_path).resolve())
            if len(abs_path) < 95:
                new_ref_path = abs_path
            else:
                new_ref_path = os.path.relpath(new_ref_path)
            if not new_ref_path.endswith('/'):
                new_ref_path += '/'
        os.environ['nref'] = new_ref_path

        # 4. Verify directory and reference file existence
        check_iref = Path(self.params['paths']['iref']).resolve()
        check_nref = Path(utils.get_output_directory(self.params['paths']['nref'])).resolve()
        
        self.logger.info("=== Calibration Reference Files Check ===")
        self.logger.info(f"iref path: {check_iref} (exists: {check_iref.is_dir()})")
        self.logger.info(f"nref path: {check_nref} (exists: {check_nref.is_dir()})")
        
        if not check_iref.is_dir():
            err_msg = f"CRITICAL ERROR: The iref directory '{check_iref}' does not exist!"
            self.logger.error(err_msg)
            print(err_msg, file=sys.stderr)
            raise FileNotFoundError(err_msg)
            
        if not check_nref.is_dir():
            err_msg = f"CRITICAL ERROR: The nref directory '{check_nref}' does not exist!"
            self.logger.error(err_msg)
            print(err_msg, file=sys.stderr)
            raise FileNotFoundError(err_msg)
        
        band = self.params['instrument']['filter']
        dummy_pflat = self.wfc3_dum_files[band]['pflat']
        dummy_dflat = self.wfc3_dum_files[band]['dflat']
        
        real_pflat = dummy_pflat.replace('_1s_pfl.fits', '_pfl.fits')
        real_dflat = dummy_dflat.replace('_1s_dfl.fits', '_dfl.fits')
        
        # Check dummy files in nref
        dp_path = check_nref / dummy_pflat
        dd_path = check_nref / dummy_dflat
        self.logger.info(f"Dummy P-flat: {dp_path} (exists: {dp_path.is_file()})")
        self.logger.info(f"Dummy D-flat: {dd_path} (exists: {dd_path.is_file()})")
        
        # Check real files in iref
        rp_path = check_iref / real_pflat
        rd_path = check_iref / real_dflat
        self.logger.info(f"Real P-flat: {rp_path} (exists: {rp_path.is_file()})")
        self.logger.info(f"Real D-flat: {rd_path} (exists: {rd_path.is_file()})")
        self.logger.info("=========================================")

        return check_iref, check_nref
    
    
    def run_calw3_pipe(self, input_df, outdir=None, ncores=1):
        
        if outdir is None:
            outpath = self.params['paths']['output']
        else:
            outpath = utils.get_output_directory(name=outdir)
            self.params['paths']['output'] = outpath
            
        # Define and create calwf3 calibration subdirectory
        calwf3_outdir = Path(outpath) / 'calwf3_pipe'
        calwf3_outdir.mkdir(parents=True, exist_ok=True)
            
        # 1. Get unique parent paths from the RAW with dummy column
        unique_parents = {Path(f).resolve().parent for f in input_df['RAW with dummy']}
        
        # 2. Safety check: Ensure there is only one parent path
        if len(unique_parents) > 1:
            self.logger.warning("Multiple source directories detected: "
                                f"{unique_parents}")
        
        # Get the existing parent (using the first one found)
        current_parent = next(iter(unique_parents))
        
        # 3. Verify that raw files are located under calwf3_outdir
        if current_parent != calwf3_outdir.resolve():
            self.logger.warning(
                f"Warning: RAW with dummy parent path {current_parent} "
                f"does not match expected calwf3_pipe subdirectory {calwf3_outdir.resolve()}. "
                "Reconstructing paths."
            )
            input_df['RAW with dummy'] = input_df['RAW with dummy'].apply(
                lambda x: str(calwf3_outdir / Path(x).name)
            )
        
        # 4. Get dummy flats (set up calibration environment and check path existence)
        self.setup_calibration_env()

        worker_args = [file for file in input_df['RAW with dummy']]
        
        with multiprocessing.Pool(processes=ncores, 
                                  initializer=self.__class__._init_worker, 
                                  initargs=(self.params,)
                                  ) as pool:
            unflt_data = pool.map(self._run_single_calw3, worker_args)
        
        unflt_df = pd.DataFrame(list(unflt_data))
        
        merged_df = pd.merge(input_df, unflt_df, on='RAW with dummy', how='left')
        
        if self.params['processing']['save']:
            csv_path = Path(self.params['paths']['csvdir']).resolve()
            csv_name = self.params['instrument']['filter'] + "_unflt.csv"
            merged_df.to_csv(csv_path / csv_name, index=False)
        
        return merged_df
    
    
    
    def update_mask(self, unflattened_df, save=None):
        """Combine DQ extensions from Unflattened FLT and Source Mask into a
        single combined mask using bitwise OR.

        Parameters
        ----------
        unflattened_df : pd.DataFrame
            Output from run_calw3_pipe. Must contain 'Unflattened FLT' and
            'Source Mask' columns.
        save : bool, optional
            If True, save the combined mask as a copy of the flt file
            with the DQ extension replaced. Defaults to
            self.params['processing']['save'].

        Returns
        -------
        tuple of (pd.DataFrame, dict)
            updated_df : DataFrame with 'Combined Mask' column added
                (paths to *_cmf.fits files if save=True, else None).
            mask_dict : Dictionary mapping filename IDs
                (e.g. 'idjb10xvq_flt.fits') to combined DQ arrays.
        """

        if save is None:
            save = self.params['processing']['save']

        # Get maskbits from parameters and calculate bitmask
        maskbits = self.params.get('processing', {}).get('maskbits', None)
        bitmask = None
        if maskbits is not None:
            if isinstance(maskbits, (int, float)):
                bitmask = int(maskbits)
            elif isinstance(maskbits, list):
                bitmask = 0
                for bit in maskbits:
                    bitmask |= int(bit)
            elif isinstance(maskbits, str):
                if ',' in maskbits:
                    bitmask = 0
                    for bit in maskbits.split(','):
                        bitmask |= int(bit.strip())
                else:
                    bitmask = int(maskbits)

        if bitmask is not None:
            self.logger.info(f"Filtering DQ array using maskbits (bitmask sum: {bitmask})")

        mask_dict = {}
        cmf_paths = []

        for idx, row in unflattened_df.iterrows():
            flt_file = row.get('Unflattened FLT')
            msk_file = row.get('Source Mask')

            # Skip rows where calwf3 failed (Unflattened FLT is None)
            if flt_file is None or pd.isna(flt_file):
                self.logger.warning(
                    f"Skipping row {idx}: Unflattened FLT is None "
                    f"(Source Mask: {msk_file}). calwf3 may have failed."
                )
                cmf_paths.append(None)
                continue

            # Skip rows where Source Mask is also missing
            if msk_file is None or pd.isna(msk_file):
                self.logger.warning(
                    f"Skipping row {idx}: Source Mask is None "
                    f"(Unflattened FLT: {flt_file})."
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

            # Combine masks using bitwise OR (filtering flt_dq if bitmask is specified)
            if bitmask is not None:
                filtered_flt_dq = np.bitwise_and(flt_dq, bitmask)
                combined_dq = np.bitwise_or(filtered_flt_dq, msk_dq)
            else:
                combined_dq = np.bitwise_or(flt_dq, msk_dq)

            # Store combined mask in dictionary keyed by filename ID
            mask_dict[file_id] = combined_dq

            if save:
                # Save as a copy of flt with DQ replaced by combined mask
                cmf_name = flt_path.name.replace('_flt', '_cmf')
                calwf3_dir = self.params['paths'].get('calwf3_dir')
                cmf_fullpath = (Path(calwf3_dir) if calwf3_dir else Path(self.params['paths']['output'])) / cmf_name

                with fits.open(flt_file) as cmf_hdul:
                    cmf_hdul['DQ', 1].data = combined_dq
                    cmf_hdul.writeto(cmf_fullpath, overwrite=True)

                cmf_paths.append(str(cmf_fullpath))
                self.logger.info(f"Combined mask saved: {cmf_name}")
            else:
                cmf_paths.append(None)

        # Add Combined Mask column to a copy of the dataframe
        updated_df = unflattened_df.copy()
        updated_df['Combined Mask'] = cmf_paths

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
    
    
    def get_blob_info(self):
        """Read active blobs from both the external blobfile and the parameter file.

        Returns
        -------
        dict
            A dictionary containing the blobs to process, keyed by blob ID.
        """
        blobs_dict = {}

        # 1. Read from external blobfile if specified and exists
        blob_file = None
        if 'files' in self.params:
            blob_file = self.params['files'].get('blobfile')

        if blob_file:
            blob_path = Path(blob_file)
            # Resolve relative paths with respect to the input directory if needed
            if not blob_path.is_absolute() and 'paths' in self.params:
                inpath = self.params['paths'].get('input')
                if inpath:
                    trial_path = Path(inpath) / blob_path
                    if trial_path.is_file():
                        blob_path = trial_path

            blob_path = blob_path.resolve()
            if blob_path.is_file():
                self.logger.info(f"Reading active blobs from blobfile: {blob_path}")
                try:
                    with open(blob_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped or stripped.startswith('#'):
                                continue
                            parts = stripped.split()
                            # Expect: N, X, Y, RADIUS, COLOR, FLUX, APPEARED
                            if len(parts) >= 7:
                                try:
                                    b_id = int(parts[0])
                                    b_x = float(parts[1])
                                    b_y = float(parts[2])
                                    b_rad = float(parts[3])
                                    b_app = float(parts[6])

                                    blobs_dict[b_id] = {
                                        'id': b_id,
                                        'x': b_x,
                                        'y': b_y,
                                        'radius': b_rad,
                                        'appeared': b_app
                                    }
                                except ValueError as ve:
                                    self.logger.warning(
                                        f"Skipping line due to parse error: '{line.strip()}' - {ve}"
                                    )
                except Exception as e:
                    self.logger.error(f"Error reading blobfile {blob_path}: {e}")
            else:
                self.logger.warning(f"Blobfile not found: {blob_path}")

        # 2. Read from blobs keyword in parameters (YAML file)
        param_blobs = None
        if 'blobs' in self.params:
            param_blobs = self.params['blobs']
        elif 'processing' in self.params and 'blobs' in self.params['processing']:
            param_blobs = self.params['processing']['blobs']

        if param_blobs:
            self.logger.info("Reading blobs from parameter configuration")
            if not isinstance(param_blobs, list):
                self.logger.error("The 'blobs' parameter must be a list in the configuration.")
            else:
                for b_entry in param_blobs:
                    if not isinstance(b_entry, dict):
                        self.logger.warning(f"Skipping non-dict blob entry: {b_entry}")
                        continue

                    # Case-insensitive mapping of keys
                    lower_entry = {k.lower(): v for k, v in b_entry.items()}

                    id_val = lower_entry.get('id') or lower_entry.get('n')
                    x_val = lower_entry.get('x')
                    y_val = lower_entry.get('y')
                    rad_val = lower_entry.get('radius')
                    app_val = lower_entry.get('appeared')

                    # Validate required fields
                    missing_fields = []
                    if id_val is None:
                        missing_fields.append('id/N')
                    if x_val is None:
                        missing_fields.append('x')
                    if y_val is None:
                        missing_fields.append('y')
                    if rad_val is None:
                        missing_fields.append('radius')
                    if app_val is None:
                        missing_fields.append('appeared')

                    if missing_fields:
                        self.logger.error(
                            f"Blob entry {b_entry} is missing required keys: {', '.join(missing_fields)}"
                        )
                        continue

                    try:
                        b_id = int(id_val)
                        blobs_dict[b_id] = {
                            'id': b_id,
                            'x': float(x_val),
                            'y': float(y_val),
                            'radius': float(rad_val),
                            'appeared': float(app_val)
                        }
                    except ValueError as ve:
                        self.logger.error(
                            f"Could not cast values to numeric types in blob entry {b_entry}: {ve}"
                        )

        self.logger.info(f"Loaded {len(blobs_dict)} blobs to process: {list(blobs_dict.keys())}")
        return blobs_dict
    
    
    def generate_flat(self, input_df, blobs=None, method="mean"):
        """Generate master flat field(s). If active blobs are present, stacks

        are generated for each blob using only exposures observed after the
        blob's appearance date.

        Parameters
        ----------
        input_df : pd.DataFrame
            DataFrame containing paths to files to process.
        blobs : dict, optional
            Dictionary of active blobs keyed by ID. If None, queries get_blob_info().
        method : str, optional
            Stacking method ("mean" or "median"). Default is "mean".

        Returns
        -------
        dict or tuple
            If blobs are processed, returns the updated blobs dictionary containing the
            stacked 'flat', 'flat_unc', and 'mask' arrays for each active blob. Otherwise,
            returns a single (flat, flat_unc, mask) tuple.
        """
        # Determine observation MJDs
        if 'EXPSTART' in input_df.columns:
            obs_mjds = pd.to_numeric(input_df['EXPSTART'])
        else:
            from astropy.time import Time
            obs_mjds = input_df['DATE-OBS'].apply(lambda x: Time(x).mjd)

        # Get active blobs if not explicitly passed
        if blobs is None:
            blobs = self.get_blob_info()

        # If there are no active blobs, generate the standard master flat field
        if not blobs:
            self.logger.info("No active blobs to process. Generating standard master flat.")
            flat, flat_unc, mask = self._stack_flat_subset(input_df, obs_mjds, method=method)
            return flat, flat_unc, mask

        # If there are active blobs, generate time-dependent flats for each
        for b_id, b_info in blobs.items():
            b_app = b_info['appeared']
            # Filter exposures: obs_mjd >= blob_appeared_date
            subset_mask = (obs_mjds >= b_app)
            subset_df = input_df[subset_mask]

            self.logger.info(
                f"Blob {b_id} (appeared: {b_app}): Stacking {len(subset_df)} files "
                f"observed after appearance MJD."
            )

            if len(subset_df) == 0:
                self.logger.warning(
                    f"No files found observed after MJD {b_app} for blob {b_id}. Skipping."
                )
                continue

            flat, flat_unc, mask = self._stack_flat_subset(
                subset_df,
                obs_mjds[subset_mask],
                method=method,
                blob_id=b_id,
                blob_appeared=b_app
            )
            b_info['flat'] = flat
            b_info['flat_unc'] = flat_unc
            b_info['mask'] = mask

        return blobs


    def _stack_flat_subset(self, subset_df, subset_mjds, method="mean", blob_id=None, blob_appeared=None):
        """Helper method to stack a subset of flat files and write the FITS data product.

        Parameters
        ----------
        subset_df : pd.DataFrame
            DataFrame containing paths of files to process.
        subset_mjds : pd.Series
            Observation MJDs corresponding to the subset_df files.
        method : str, optional
            Stacking method ("mean" or "median"). Default is "mean".
        blob_id : int, optional
            ID of the blob, if generating a blob-specific flat.
        blob_appeared : float, optional
            Appearance MJD of the blob, if generating a blob-specific flat.

        Returns
        -------
        tuple
            (flat, flat_unc, mask) for the stacked subset.
        """
        n_files = len(subset_df)
        sci_data = None
        err_data = None
        stacked_files = []

        for i, (idx, row) in enumerate(subset_df.iterrows()):
            # 1. Determine unflattened file path (fl) to load SCI & ERR arrays
            fl = row.get('Unflattened FLT')
            if not fl or pd.isna(fl):
                self.logger.error("No valid Unflattened FLT path found in row.")
                continue

            hdul = fits.open(fl)
            stacked_files.append(fl)

            # 2. Determine mask data (DQ array)
            mask = None
            cmf_file = row.get('Combined Mask')
            msk_file = row.get('Source Mask')

            # Check for combined mask (cmf) first
            if cmf_file and not pd.isna(cmf_file) and Path(cmf_file).is_file():
                m_dat = fits.getdata(cmf_file, ext=('DQ', 1))
                mask = np.array(m_dat, dtype='bool')
            # Fallback to outlier mask (msk)
            elif msk_file and not pd.isna(msk_file) and Path(msk_file).is_file():
                m_dat = fits.getdata(msk_file, ext=('DQ', 1))
                mask = np.array(m_dat, dtype='bool')
            # Fallback to string replacement on disk
            else:
                msk_path = str(fl).replace('_cmf.fits', '_msk.fits').replace('_flt.fits', '_msk.fits')
                if Path(msk_path).is_file():
                    m_dat = fits.getdata(msk_path, ext=('DQ', 1))
                    mask = np.array(m_dat, dtype='bool')
                else:
                    mask = np.zeros_like(hdul['SCI', 1].data, dtype='bool')

            hdul['SCI', 1].data[mask] = np.nan
            hdul['ERR', 1].data[mask] = np.nan

            if sci_data is None:
                nx = hdul['SCI', 1].data.shape[0]
                ny = hdul['SCI', 1].data.shape[1]
                sci_data = np.zeros((n_files, nx, ny))
                err_data = np.zeros((n_files, nx, ny))

            # Normalization region 101:900, 101:900
            sci_mval = np.nanmean(hdul['SCI', 1].data[101:900, 101:900])
            err_mval = np.nanmean(hdul['ERR', 1].data[101:900, 101:900])

            sci_data[i, :, :] = hdul['SCI', 1].data / sci_mval
            err_data[i, :, :] = hdul['ERR', 1].data / err_mval

            hdul.close()

        with warnings.catch_warnings(action="ignore"):
            sf_mean, sf_median, sf_std = sigma_clipped_stats(
                sci_data,
                mask_value=np.nan,
                sigma_lower=3,
                sigma_upper=3,
                axis=0
            )

        if method == "median":
            flat = sf_median
        else:
            flat = sf_mean

        flat_unc = sf_std / np.sqrt(n_files)

        # Set DQ flag for pixels that exhibit AD_Floor
        xpix, ypix = np.where(flat < 0.0)

        if len(xpix) >= 1:
            self.logger.info(f"{len(xpix)} pixels had negative values. These pixels")
            self.logger.info(" are NaN-ed in the SCI & ERR arrays and AD_Floor flag")
            self.logger.info(" is set the DQ array.")
            for x, y in zip(xpix, ypix):
                flat[x, y] = np.nan
                flat_unc[x, y] = np.nan
                mask[x, y] = True
        else:
            self.logger.info(" All pixels passed the AD_FLOOR test.")

        # Check and pad arrays from 1014x1014 to 1024x1024 if needed
        if flat.shape == (1014, 1014):
            self.logger.info("Padding stacked P-flat (flat, flat_unc, mask) from 1014x1014 to 1024x1024 with pixel values of 1.")
            flat = np.pad(flat, pad_width=5, mode='constant', constant_values=1.0)
            flat_unc = np.pad(flat_unc, pad_width=5, mode='constant', constant_values=0.0)
            mask = np.pad(mask, pad_width=5, mode='constant', constant_values=False)

        # Save flat FITS file if requested
        if self.params['processing']['save'] and len(stacked_files) > 0:
            hdr = fits.getheader(stacked_files[0], ext=0)
            phdu = fits.PrimaryHDU(header=hdr)

            # Find the last standard keyword in the copied header to place our section after it
            after_key = None
            for key in reversed(list(phdu.header.keys())):
                if key and key not in ['COMMENT', 'HISTORY', '']:
                    after_key = key
                    break

            if after_key:
                phdu.header.set('NFILES', n_files, 'Number of files stacked', after=after_key)
            else:
                phdu.header.set('NFILES', n_files, 'Number of files stacked')

            phdu.header.set('METHOD', method, 'Stacking method', after='NFILES')
            if blob_id is not None:
                phdu.header.set('BLOB_ID', blob_id, 'Blob ID', after='METHOD')
                phdu.header.set('BLOB_MJD', blob_appeared, 'Blob appearance MJD', after='BLOB_ID')

            # Insert D-FLAT Keywords section banner before NFILES
            phdu.header.insert('NFILES', ('', ''), after=False)
            phdu.header.insert('NFILES', ('', '      / D-FLAT Keywords'), after=False)
            phdu.header.insert('NFILES', ('', ''), after=False)

            # List contributing files under HISTORY cards
            phdu.header.add_history('The following files were used in the stack:')
            for fl_path in stacked_files:
                file_id = Path(fl_path).name.replace('_cmf.fits', '').replace('_flt.fits', '')
                phdu.header.add_history(f'  {file_id}')

            sci_hdu = fits.ImageHDU(data=flat, name='SCI', ver=1)
            err_hdu = fits.ImageHDU(data=flat_unc, name='ERR', ver=1)
            flt_mask = np.zeros_like(sci_hdu.data, dtype=np.int16)
            msk_hdu = fits.ImageHDU(data=flt_mask, name='DQ', ver=1)

            flat_hdul = fits.HDUList([phdu, sci_hdu, err_hdu, msk_hdu])

            # Output paths relative to out_path / 'pflats'
            out_path = Path(self.params['paths']['output']) / 'pflats'
            out_path.mkdir(parents=True, exist_ok=True)

            if blob_id is not None:
                out_file = f"HST_WFC3_IR_{self.params['instrument']['filter']}_{blob_id}_PFlat.fits"
            else:
                out_file = f"HST_WFC3_IR_{self.params['instrument']['filter']}_PFlat.fits"

            flat_hdul.writeto(str(out_path / out_file), overwrite=True)
            self.logger.info(f"Saved stacked flat: {out_path / out_file}")

        return flat, flat_unc, mask
    
    
    def get_current_dflat(self):
        """Locates and loads the current real reference D-flat for the active filter from the iref directory.

        Returns
        -------
        dflat_path : Path
            Path to the real reference D-flat FITS file.
        hdul : astropy.io.fits.HDUList
            Opened HDUList of the D-flat file.
        """
        # 1. Determine active filter
        band = self.params['instrument']['filter']
        
        # 2. Check if a specific dflat file is defined in yaml params
        dflat_filename = self.params.get('instrument', {}).get('dflat', None)
        
        if not dflat_filename:
            # Fall back to deriving it from wfc3_dum_files mapping
            dummy_dflat = self.wfc3_dum_files[band]['dflat']
            # e.g., '4ac18187i_1s_dfl.fits' -> '4ac18187i_dfl.fits'
            dflat_filename = dummy_dflat.replace('_1s_dfl.fits', '_dfl.fits')
            self.logger.info(f"Derived real D-flat filename: {dflat_filename}")
        else:
            self.logger.info(f"Using D-flat filename from config: {dflat_filename}")
            
        # 3. Locate the D-flat in the iref directory
        iref_dir = Path(self.params['paths']['iref']).resolve()
        dflat_path = iref_dir / dflat_filename
        
        if not dflat_path.is_file():
            warn_msg = f"Real D-flat file not found at: {dflat_path}"
            self.logger.warning(warn_msg)
            # Try package data resources as a potential fallback
            ref_path = resources.files('pywfc3.data')
            dref_file = ref_path.joinpath(dflat_filename)
            if dref_file.is_file():
                self.logger.info(f"Found {dflat_filename} in package data resources. Copying to iref directory.")
                iref_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dref_file, dflat_path)
            else:
                raise FileNotFoundError(f"Real D-flat file {dflat_filename} could not be located in iref ({iref_dir}) or package resources.")
        
        if 'instrument' not in self.params:
            self.params['instrument'] = {}
        self.params['instrument']['dflat'] = dflat_filename

        self.logger.info(f"Loading current real D-flat: {dflat_path}")
        hdul = fits.open(dflat_path)
        
        return dflat_path, hdul
    
    # Change this to make_new_dflat
    def get_new_dflat(self, blobs):
        """Divides the current reference D-flat by each individual stacked P-flat (derived from generate_flat).
        Saves each resulting FITS file in an output directory named 'dflats'.

        Parameters
        ----------
        blobs : dict
            Dictionary of active blobs containing stacked 'flat' (SCI) and 'flat_unc' (ERR) arrays.
            Keyed by blob ID.
        
        Returns
        -------
        output_paths : dict
            Dictionary mapping blob ID to the Path of the saved D-flat FITS file.
        """
        if not isinstance(blobs, dict) or not blobs:
            self.logger.warning("No active blobs to process for D-flat update.")
            return {}

        # 1. Get the current reference D-flat
        dflat_path, dflat_hdul = self.get_current_dflat()
        dflat_filename = dflat_path.name
        
        # 2. Extract arrays from the reference D-flat
        dflat_sci = dflat_hdul['SCI', 1].data
        dflat_err = dflat_hdul['ERR', 1].data
        dflat_dq = dflat_hdul['DQ', 1].data
        
        # Determine the active filter name
        filt = self.params['instrument']['filter']
        
        # 3. Setup the output directory
        out_dir = Path(self.params['paths']['output']) / 'dflats'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}

        # 4. Loop through each active blob
        for b_id, b_info in blobs.items():
            if 'flat' not in b_info or 'flat_unc' not in b_info:
                self.logger.warning(f"Blob {b_id} is missing stacked flat arrays. Skipping.")
                continue
                
            pflat_sci = b_info['flat']
            pflat_err = b_info['flat_unc']
            
            # Safe division: ratio_sci = dflat_sci / pflat_sci
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_sci = np.where(pflat_sci != 0, dflat_sci / pflat_sci, 0.0)
                # Safeguard against any unexpected NaN or Inf values
                ratio_sci[np.isnan(ratio_sci) | np.isinf(ratio_sci)] = 0.0
                
                # Safe error propagation
                term_d = np.where(dflat_sci != 0, dflat_err / dflat_sci, 0.0)
                term_p = np.where(pflat_sci != 0, pflat_err / pflat_sci, 0.0)
                ratio_err = ratio_sci * np.sqrt(term_d**2 + term_p**2)
                # Safeguard against any unexpected NaN or Inf values
                ratio_err[np.isnan(ratio_err) | np.isinf(ratio_err)] = 0.0
            
            # 5. Build the new FITS structure based on the reference D-flat
            # Copy Primary HDU and update header
            phdu = fits.PrimaryHDU(header=dflat_hdul[0].header.copy())
            
            # Add HISTORY cards documenting the update/division operation
            phdu.header.add_history("Divided reference D-flat by stacked P-flat.")
            phdu.header.add_history(f"Target Blob ID: {b_id}")
            phdu.header.add_history(f"Filter: {filt}")
            phdu.header.add_history(f"Reference D-flat used: {dflat_filename}")
            
            # Create HDUs for SCI, ERR, and DQ extensions
            sci_hdu = fits.ImageHDU(data=ratio_sci, header=dflat_hdul['SCI', 1].header.copy(), name='SCI', ver=1)
            err_hdu = fits.ImageHDU(data=ratio_err, header=dflat_hdul['ERR', 1].header.copy(), name='ERR', ver=1)
            dq_hdu = fits.ImageHDU(data=dflat_dq.copy(), header=dflat_hdul['DQ', 1].header.copy(), name='DQ', ver=1)
            
            new_hdul = fits.HDUList([phdu, sci_hdu, err_hdu, dq_hdu])
            
            # Construct output filename
            # e.g., HST_WFC3_IR_F140W_141_DFlat.fits
            out_file = out_dir / f"HST_WFC3_IR_{filt}_{b_id}_DFlat.fits"
            
            new_hdul.writeto(str(out_file), overwrite=True)
            self.logger.info(f"Saved divided D-flat for blob {b_id} at: {out_file}")
            
            # Generate DS9 region file for this specific D-flat
            reg_file = out_dir / f"HST_WFC3_IR_{filt}_{b_id}_DFlat.reg"
            try:
                with open(reg_file, 'w', encoding='utf-8') as rf:
                    rf.write("# Region file format: DS9 version 4.1\n")
                    rf.write("global color=green select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1\n")
                    rf.write("image\n")
                    for other_id, other_info in blobs.items():
                        color = "red" if other_id == b_id else "green"
                        rf.write(f"circle({other_info['x'] + 1.0:.1f}, {other_info['y'] + 1.0:.1f}, {other_info['radius']:.1f}) # color={color} text=\"Blob {other_id}\"\n")
                self.logger.info(f"Saved DS9 region file for blob {b_id} at: {reg_file}")
            except Exception as e:
                self.logger.error(f"Failed to write DS9 region file for blob {b_id}: {e}")
                
            output_paths[b_id] = out_file
            
        dflat_hdul.close()
        return output_paths

    def update_dflat_with_blobs(self, blobs, divided_dflat_paths):
        """Updates the current reference D-flat by inserting circular blob regions
        from the divided D-flats. Saves the updated D-flat in the output directory
        with a timestamped filename.

        Parameters
        ----------
        blobs : dict
            Dictionary of active blobs containing keys: 'x', 'y', 'radius', etc.
        divided_dflat_paths : dict
            Dictionary mapping blob ID to the Path of the saved divided D-flat.

        Returns
        -------
        updated_dflat_path : Path
            Path to the saved updated D-flat.
        """
        if not isinstance(blobs, dict) or not blobs:
            self.logger.warning("No active blobs to process for D-flat insertion.")
            return None

        # 1. Get the current reference D-flat
        dflat_path, dflat_hdul = self.get_current_dflat()
        dflat_filename = dflat_path.name

        # Copy the HDUList structure and contents exactly to preserve all extensions
        updated_hdul = fits.HDUList([hdu.copy() for hdu in dflat_hdul])
        
        current_sci = updated_hdul['SCI', 1].data
        current_err = updated_hdul['ERR', 1].data
        current_dq = updated_hdul['DQ', 1].data

        ny, nx = current_sci.shape
        y_grid, x_grid = np.ogrid[:ny, :nx]

        # 2. Iterate through each active blob
        for b_id, b_info in blobs.items():
            if b_id not in divided_dflat_paths:
                self.logger.warning(f"No divided D-flat path found for blob {b_id}. Skipping.")
                continue

            div_path = divided_dflat_paths[b_id]
            if not div_path.is_file():
                self.logger.warning(f"Divided D-flat file not found at: {div_path}. Skipping.")
                continue

            # Extract blob parameters
            x = b_info['x']
            y = b_info['y']
            radius = b_info['radius']

            np_x = x
            np_y = y
            dist_sq = (x_grid - np_x)**2 + (y_grid - np_y)**2
            mask = dist_sq <= radius**2

            # Load the divided D-flat data
            self.logger.info(f"Extracting region for blob {b_id} from {div_path.name}")
            with fits.open(div_path) as div_hdul:
                div_sci = div_hdul['SCI', 1].data
                div_err = div_hdul['ERR', 1].data
                div_dq = div_hdul['DQ', 1].data

                # Insert the extracted region into the current D-flat
                current_sci[mask] = div_sci[mask]
                current_err[mask] = div_err[mask]
                current_dq[mask] = div_dq[mask]

            # Update header history
            history_msg = f"Inserted blob {b_id} at x={x:.2f}, y={y:.2f} with radius={radius:.2f}"
            updated_hdul[0].header.add_history(history_msg)
            self.logger.info(f"Updated primary header history: {history_msg}")

        # 3. Save the updated D-flat with the updated name
        today_str = datetime.now().strftime("%Y%m%d")
        updated_filename = f"{dflat_path.stem}_updated_{today_str}.fits"
        
        out_dir = Path(self.params['paths']['output']) / 'dflats'
        out_dir.mkdir(parents=True, exist_ok=True)
        updated_dflat_path = out_dir / updated_filename

        updated_hdul.writeto(str(updated_dflat_path), overwrite=True)
        self.logger.info(f"Saved final updated D-flat at: {updated_dflat_path}")

        # 4. Generate DS9 region file for the updated D-flat
        reg_file = out_dir / f"{dflat_path.stem}_updated_{today_str}.reg"
        try:
            with open(reg_file, 'w', encoding='utf-8') as rf:
                rf.write("# Region file format: DS9 version 4.1\n")
                rf.write("global color=green select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1\n")
                rf.write("image\n")
                for b_id, b_info in blobs.items():
                    if b_id in divided_dflat_paths:
                        rf.write(f"circle({b_info['x'] + 1.0:.1f}, {b_info['y'] + 1.0:.1f}, {b_info['radius']:.1f}) # color=green text=\"Blob {b_id}\"\n")
            self.logger.info(f"Saved DS9 region file for updated D-flat at: {reg_file}")
        except Exception as e:
            self.logger.error(f"Failed to write DS9 region file for updated D-flat: {e}")

        dflat_hdul.close()
        updated_hdul.close()

        return updated_dflat_path