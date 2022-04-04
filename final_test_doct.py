# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:47:24 2022

@author: LarsNelson
"""

from acq_class import (MiniAnalysis, PostSynapticEvent, CurrentClamp,
                       LoadMiniAnalysis, LoadMini, LFP, oEPSC,
                       LoadCurrentClamp, Acquisition)
from final_analysis_classes import (FinalMiniAnalysis, FinalEvokedCurrent,
                                    FinalCurrentClampAnalysis)
from utility_classes import NumpyEncoder
from load_classes import LoadMiniSaveData, LoadCurrentClampData
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal, integrate,stats, interpolate
import numpy as np
import bottleneck as bn
import pandas as pd
from scipy.stats import linregress
import pickle
from glob import glob
import json
from sklearn.linear_model import LinearRegression
from patsy import cr
# import csaps


#Testing out the mini analysis function on Nov 16, 2021
acq_dict = {}
for i in range(1, 31):
    print(i)
    xy = MiniAnalysis('AD0', i, 10000, 0, 80, 'remez_2', order=101,
        low_pass=600,low_width=600, template=None, rc_check=True,
        rc_check_start=10000, rc_check_end=10300, sensitivity=2.5,
        amp_threshold=7, mini_spacing=5, min_rise_time=0.5, min_decay_time=0.7,
        invert=False)
    xy.analyze()
    acq_dict[str(i)] = xy

test = FinalMiniAnalysis(acq_dict)
k = test.final_df

x = acq_dict['170']
x.save_postsynaptic_events()
with open("Cell_6_AD0_170.json", "w") as write_file:
    json.dump(x.__dict__, write_file, cls=NumpyEncoder)

json_dict = {}
file_list = glob('*.json')
for i in range(len(file_list)):
    with open(file_list[i]) as file:
        data = json.load(file)
        x = LoadMiniAnalysis(data)
        json_dict[str(x.acq_number)] = x

json_dict['172'].postsynaptic_events = [x for _, x in
    sorted(zip(json_dict['172'].final_events, json_dict['172'].postsynaptic_events))]




with open('test', 'wb') as pic:
    pickle.dump(acq_dict, pic, protocol=pickle.HIGHEST_PROTOCOL)

with open('test', 'rb') as pic_o:
    test = pickle.load(pic_o)

j = pd.read_excel('Test.xlsx', sheet_name=None)

test_load = LoadMiniSaveData(j)


#Testing out the LFP and oEPSC function
#August 4, 2021
lfp_acq_dict = {}
oepsc_acq_dict = {}
for i in range(11,44):
    oepsc= oEPSC('AD1', 11, 10000, 0, 1000, 'savgol', order=5, polyorder=3)
    lfp = LFP('AD0', 11, 10000, 0, 1000, 'savgol', order=5, polyorder=3)
    oepsc_acq_dict[str(i)] = oepsc
    lfp_acq_dict[str(i)] = lfp

test = FinalEvokedCurrent(o_acq_dict=oepsc_acq_dict, lfp_acq_dict=lfp_acq_dict)
test.save_data('Test')


#Testing out the current clamp final analysis using Oct 29, 2021
acq_dict = {}
for i in range(83,474):
    print(i)
    acq = CurrentClamp('AD0', i, 10000, baseline_start=0, baseline_end=300)
    acq_dict[str(i)] = acq

test = FinalCurrentClampAnalysis(acq_dict)
test.save_data('test')

for i in test.deltav_df.columns:
    plt.plot(test.deltav_df)



final_obj = LoadCurrentClampData('test.xlsx')

plt.stemplot