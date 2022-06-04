# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:47:24 2022

@author: LarsNelson
"""
# %%
from acq_class import (
    MiniAnalysis,
    PostSynapticEvent,
    CurrentClamp,
    LoadMiniAnalysis,
    LoadMini,
    LFP,
    oEPSC,
    LoadCurrentClamp,
    Acquisition,
)
from final_analysis_classes import (
    FinalMiniAnalysis,
    FinalEvokedCurrent,
    FinalCurrentClampAnalysis,
)
from utilities import load_scanimage_file, load_mat
from utility_classes import NumpyEncoder
from load_classes import LoadMiniSaveData, LoadCurrentClampData
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal, integrate, stats, interpolate
import numpy as np
import bottleneck as bn
import pandas as pd
from scipy.stats import linregress
import pickle
from glob import glob
import json
from sklearn.linear_model import LinearRegression
from patsy import cr
import copy
from pathlib import PurePath


# %%
acq_dict = {}
path = PurePath(r"/Volumes/Backup/Lars Slice Ephys/2021_12_09/AD0_1.mat")
acq_components = load_scanimage_file(path)

#%%
xy = MiniAnalysis(
    acq_components=acq_components,
    sample_rate=10000,
    baseline_start=0,
    baseline_end=80,
    filter_type="fir_zero_2",
    order=201,
    low_pass=600,
    low_width=600,
    window="hann",
    template=None,
    rc_check=True,
    rc_check_start=10000,
    rc_check_end=10300,
    sensitivity=2.5,
    amp_threshold=7,
    mini_spacing=5,
    min_rise_time=0.5,
    min_decay_time=0.7,
    invert=False,
    curve_fit_decay=True,
    decon_type="weiner",
)
xy.analyze()

#%%
kernel = np.hstack((xy.template, np.zeros(len(xy.final_array) - len(xy.template))))

#%%
convolved = signal.convolve(xy.final_array, xy.template, mode="same")

#%%
correlated = signal.correlate(xy.final_array, xy.template[30:], mode="same")

#%%
mu, std = stats.norm.fit(xy.final_decon_array)
peaks, _ = signal.find_peaks(xy.final_decon_array, height=2.25 * abs(std))
events = peaks

#%%
z = bn.move_mean(xy.final_array, window=5, min_count=1) ** 2
mu, std = stats.norm.fit(xy.final_decon_array)
peaks, _ = signal.find_peaks(xy.final_decon_array, height=2.5 * abs(std))
events = peaks


#%%
fig, ax = plt.subplots()
ax.plot(xy.x_array, xy.final_decon_array, "white")
ax.plot(peaks / 10, xy.final_decon_array[peaks], "mo")
ax.axhline(2.5 * abs(std))
ax.set_facecolor("black")
ax.tick_params(axis="both", colors="white", labelsize=12)
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(axis="both", labelsize=18)
ax.spines[["top", "left", "right", "bottom"]].set_color("white")
ax.set_xlabel("Time", fontsize=20, fontweight="bold")
ax.set_ylabel("Amplitude (AU)", fontsize=20, fontweight="bold")
l = ax.figure.subplotpars.left
r = ax.figure.subplotpars.right
t = ax.figure.subplotpars.top
b = ax.figure.subplotpars.bottom
figw = float(6) / (r - l)
figh = float(4) / (t - b)
ax.figure.set_size_inches(figw, figh)


#%%
fig.savefig("/Volumes/Backup/Lab meeting 2022_06_07/weiner_decon", bbox_inches="tight")

#%%
x = copy.deepcopy(acq_dict["1"])

test = FinalMiniAnalysis(acq_dict)
k = test.final_df

x = acq_dict["170"]
x.save_postsynaptic_events()
with open("Cell_6_AD0_170.json", "w") as write_file:
    json.dump(x.__dict__, write_file, cls=NumpyEncoder)

json_dict = {}
file_list = glob("*.json")
for i in range(len(file_list)):
    with open(file_list[i]) as file:
        data = json.load(file)
        x = LoadMiniAnalysis(data)
        json_dict[str(x.acq_number)] = x

json_dict["172"].postsynaptic_events = [
    x
    for _, x in sorted(
        zip(json_dict["172"].final_events, json_dict["172"].postsynaptic_events)
    )
]


with open("test", "wb") as pic:
    pickle.dump(acq_dict, pic, protocol=pickle.HIGHEST_PROTOCOL)

with open("test", "rb") as pic_o:
    test = pickle.load(pic_o)

j = pd.read_excel("Test.xlsx", sheet_name=None)

test_load = LoadMiniSaveData(j)

#%%
# Testing out the LFP and oEPSC function
# August 4, 2021
lfp_acq_dict = {}
oepsc_acq_dict = {}
for i in range(11, 44):
    path = PurePath(f"/Volumes/Backup/Lars Slice Ephys/2021_11_21/AD0_{i}.mat")
    acq_components = load_scanimage_file(path)
    oepsc = oEPSC(
        acq_components=acq_components,
        sample_rate=10000,
        baseline_start=0,
        baseline_end=1000,
        filter_type="savgol",
        order=5,
        polyorder=3,
    )
    oepsc_acq_dict[str(i)] = oepsc
for i in range(11, 44):
    path = PurePath(f"/Volumes/Backup/Lars Slice Ephys/2021_11_21/AD1_{i}.mat")
    acq_components = load_scanimage_file(path)
    lfp = LFP(
        acq_components=acq_components,
        sample_rate=10000,
        baseline_start=0,
        baseline_end=1000,
        filter_type="savgol",
        order=5,
        polyorder=3,
    )
    lfp_acq_dict[str(i)] = lfp

#%%
test = FinalEvokedCurrent(o_acq_dict=oepsc_acq_dict, lfp_acq_dict=lfp_acq_dict)
# test.save_data('Test')

#%%
# Testing out the current clamp final analysis using Oct 29, 2021
acq_dict = {}
for i in range(1, 474):
    path = PurePath(f"D:\Lars Slice Ephys/2021_10_06/AD0_{i}.mat")
    acq_components = load_scanimage_file(path)
    print(i)
    acq = CurrentClamp(
        acq_components=acq_components,
        sample_rate=10000,
        baseline_start=0,
        baseline_end=300,
        pulse_start=300,
        pulse_end=1000,
        ramp_start=300,
        ramp_end=4000,
    )
    acq.analyze()
    acq_dict[str(i)] = acq

#%%
test = FinalCurrentClampAnalysis(acq_dict)


#%%
for i in test.deltav_df.columns:
    plt.plot(test.deltav_df)


final_obj = LoadCurrentClampData("test.xlsx")

plt.stemplot
# %%
i = 291
path = PurePath(f"D:\Lars Slice Ephys/2021_10_06/AD0_{i}.mat")
acq_components = load_scanimage_file(path)
acq = CurrentClamp(
    acq_components=acq_components,
    sample_rate=10000,
    baseline_start=0,
    baseline_end=300,
    pulse_start=300,
    pulse_end=1000,
    ramp_start=300,
    ramp_end=4000,
)
acq.analyze()

# %%
i = 235
path = PurePath(f"D:\Lars Slice Ephys/2021_10_06/AD0_{i}.mat")
acq_components = load_scanimage_file(path)
acq2 = CurrentClamp(
    acq_components=acq_components,
    sample_rate=10000,
    baseline_start=0,
    baseline_end=300,
    pulse_start=300,
    pulse_end=1000,
    ramp_start=300,
    ramp_end=4000,
)
acq2.analyze()

#%%
fig, ax = plt.subplots()
ax.plot(acq.spike_x_array(), acq.first_ap, color="white")
ax.plot(acq.spike_width_x(), acq.spike_width_y(), color="magenta")
ax.plot(acq.plot_rheo_x(), acq.spike_threshold, color="green", marker="o")
# ax.plot(acq2.x_array[acq.pulse_start:3035], np.gradient(acq2.array)[acq.pulse_start:3035], "magenta")
ax.set_facecolor("black")
ax.tick_params(axis="both", colors="white", labelsize=12)
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(axis="both", labelsize=18)
ax.spines[["top", "left", "right", "bottom"]].set_color("white")
ax.set_xlabel("Time", fontsize=20, fontweight="bold")
ax.set_ylabel("Amplitude (mV/ms)", fontsize=20, fontweight="bold")
l = ax.figure.subplotpars.left
r = ax.figure.subplotpars.right
t = ax.figure.subplotpars.top
b = ax.figure.subplotpars.bottom
figw = float(6) / (r - l)
figh = float(4) / (t - b)
ax.figure.set_size_inches(figw, figh)


#%%
fig.savefig("D:\Lab meeting 2022_06_07/cc_inter", bbox_inches="tight")


# %%
i = 292
path = PurePath(f"D:\Lars Slice Ephys/2021_10_06/AD0_{i}.mat")
acq_components = load_scanimage_file(path)
ak = Acquisition(
    acq_components=acq_components,
    sample_rate=10000,
    baseline_start=0,
    baseline_end=300,
)

# %%

