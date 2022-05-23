# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:59:41 2022

Last updated on Wed Feb 16 12:33:00 2021

@author: LarsNelson
"""

from pathlib import Path
import os
from utility_classes import YamlWorker

import pyqtgraph as pg


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
            pref_dict = {"MiniAnalysisWidget": 'test'}
            pass
    else:
        os.mkdir(p/h)

mini_app = color_dict = {
            'Template plot axis': pg.mkColor('w').name(),
            'Template plot background': pg.mkColor('k').name(),
            'Scroll plot axis': pg.mkColor('w').name(),
            'Scroll plot background': pg.mkColor('k').name(),
            'Inspection plot axis': pg.mkColor('w').name(),
            'Inspection plot background': pg.mkColor('k').name(),
            'Mini plot axis': pg.mkColor('w').name(),
            'Mini plot background': pg.mkColor('k').name(),
            'Ave mini plot axis': pg.mkColor('w').name(),
            'Ave mini plot background': pg.mkColor('k').name(),
        }