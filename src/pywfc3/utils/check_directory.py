#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:45:46 2025

@author: sshenoy
"""

import os
from glob import glob

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
    
    user_directory = os.path.abspath(input_dir)
    
    if data_dir:
        data_directory = os.path.join(user_directory, 'fits')
        if not os.path.isdir(data_directory):
            data_directory = user_directory
        
        if not os.path.isdir(data_directory):
            print(" Data directory does not exist. Returning None.")

        files_list = glob(os.path.join(data_directory, '*.fits'))
        if len(files_list) == 0:
            print(" Did not find any fits data file in the data directory:")
            print(f"    {data_directory}")
        
        input_directory = data_directory
        
    else:
        if not os.path.isdir(user_directory):
            print(" User Directory not found. Using current directory as ")
            print("    input directory.")
            input_directory = os.getcwd()
        else:
            input_directory = user_directory
    
    
    return input_directory