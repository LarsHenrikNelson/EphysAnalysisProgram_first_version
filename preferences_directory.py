# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:59:41 2022

Last updated on Wed Feb 16 12:33:00 2021

@author: LarsNelson
"""

from pathlib import Path
import os
from utility_classes import YamlWorker


#Find home directory and determine whether program directory exists.
#If program directory does not exist, create the directory.
def startup_function():
    p = Path.home()
    h = 'EphysAnalysisProgram'
    file_name = 'Preferences.yaml'    

    
    if Path(p/h).exists():
        if Path(p/h/file_name).exists():
            pref_dict = YamlWorker.load_yaml(p/h/file_name)
        else:
            pref_dict = {"MiniAnalysisWidget": }
            pass
    else:
        os.mkdir(p/h)