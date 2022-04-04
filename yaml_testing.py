# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:10:12 2021

@author: LarsNelson
"""

import yaml
import os
import pathlib
from utility_classes import YamlWorker



h = {'self.line_edit': '1000', 'self.high_pass': '', 'self.lowpass': '600'}

test = YamlWorker.load_yaml()

with open('test.yaml', 'w') as file:
    yaml.dump(h, file)
    
with open('test.yaml', 'r') as file:
    test_open = yaml.safe_load(file)
    


if not os.path.isfile('eap_config_file.yaml'):
    mini_analysis_config = {
        ''}
    
    
h = {'test': {'test': '1', 'test_1': 'u'},
     'test_1': {'test': 'l','test_1': 'r'}}

