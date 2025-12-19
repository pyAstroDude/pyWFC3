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
import json
import warnings
import importlib.resources

# from glob import glob
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
            if os.path.exists(paramfile):
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
        
        
        
        