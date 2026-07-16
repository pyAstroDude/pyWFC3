#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:45:46 2025

@author: sshenoy
"""

import sys
from pathlib import Path

def check_directory(input_dir, data_dir=False):
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
    input_directory : str
        Returns the inpath variables of MakeDFlat class. If data_dir
        is set then returns the datadir variable of MakeDFlat.

    """
    
    user_directory = Path(input_dir).resolve()
    
    if data_dir:
        if user_directory.is_dir():
            data_directory = user_directory
            fits_dir = data_directory / 'fits'
            if fits_dir.is_dir():
                data_directory = fits_dir
            
            files_list = list(data_directory.glob('*.fits'))
            if len(files_list) == 0:
                print(" Did not find any FITS data file in the data directory:")
                print(f"\t{data_directory}".expandtabs(4))
                sys.exit(" Stopping execution.")
        else:
            sys.exit(" Data directory does not exist. Stopping Execution.")

        
        input_directory = data_directory
        
    else:
        if not user_directory.is_dir():
            print(" User Directory not found. Using current directory as ")
            print("    input directory.")
            input_directory = Path.cwd()
        else:
            input_directory = user_directory
    
    
    return input_directory