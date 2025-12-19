#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:52:01 2025

@author: sshenoy
"""

import os

def get_output_directory(name=None):
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
    
    print(" Created output directory: ")
    print(f"    {output_directory}")
    
    return output_directory
