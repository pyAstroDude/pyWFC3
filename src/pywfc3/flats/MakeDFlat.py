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
# import json
# import warnings

from glob import glob

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self):
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
        
    
    
    def check_directory(self, input_dir, data_dir=False):
        """
        This method checks if the input directory exists and if it does
        then updates the approprite class variable. If the data_dir flag 
        is set then the function checks for the existence of the diretory
        as well as the existance of any fits data file. If the directories
        are not found then the class cariables are set to None

        Parameters
        ----------
        input_file : str
            Input path to check.
        
        data_dir : bool
            Boolean flag to check if there is any fits data in the
            input_dir. Deafult is false i.e., no checking for fits 
            data files.

        Returns
        -------
        MakeDFlat.inpath : str
            Returns the inpath variables of MakeDFlat class. If data_dir
            is set then returns the datadir variable of MakeDFlat.

        """
        
        user_directory = os.path.abspath(input_dir)
        
        if data_dir:
            data_directory = os.path.join(user_directory, 'fits')
            if not os.path.isdir(data_directory):
                data_directory = user_directory
            
            if not os.path.isdir(data_directory):
                print("Data directory does not exist. Exiting ......")
            
            self.datadir = data_directory
            input_directory = data_directory

            files_list = glob(os.path.join(data_directory, '*.fits'))
            if len(files_list) == 0:
                print("Did not find any fits data file in the data directory:")
                print(f"{data_directory}")
            
        else:
            input_directory = user_directory
            if not os.path.isdir(input_directory):
                print("Using current directory as input directory.")
                input_directory = os.getcwd()
            self.inpath = input_directory

    
    
    def get_output_directory(self, name=None):
        """
        This method checks the existence of the output directory, either
        at the deafult location './data/proc' or user supplied name variable.
        If the directory does not exists the methodd creates it.

        Parameters
        ----------
        name : str, optional
            Name of the user supplied directory. The default is None. 

        Returns
        -------
        MakeDFlat.outpath : str

        """
        
        if name is None:
            output_directory = os.getcwd() + '/proc'
        else:
            output_directory = os.path.abspath(name)
        
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        
        self.outpath = output_directory
    
    
    
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
        
        ### python dictionary varaible name to hold all the parameters.
        params = {}
        
        return params
    
    
   