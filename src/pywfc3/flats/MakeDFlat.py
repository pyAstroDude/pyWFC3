#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:34:35 2025

@author: sshenoy

This is the class that can be used to run the procedures list in WFC3 ISR 
2021-10 to generate a WFC3 IR D-Flat.
"""
import os
import json
import warnings

class MakeDFlat(object):
    """Class to generate WFC3 IR D-flat"""
    
    def __init__(self):
        self.inpath = None
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
        
    
    
    def check_input_path(self, input):
        """
        

        Parameters
        ----------
        input : str
            Input path to check.

        Returns
        -------
        input_directory : An os.path object point to the location of the 
            input directory.

        """
        
        input_directory = os.getcwd()
        
        return input_directory
    
    
    
    def get_output_directory(self, name=None):
        """
        

        Parameters
        ----------
        name : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        output_directory = os.getcwd() + '/proc'
        
        return output_directory
    
    
    
    def get_params(self, paramfile=None):
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
    
    
   