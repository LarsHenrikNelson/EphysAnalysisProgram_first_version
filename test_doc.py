# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:25:15 2021

@author: LarsNelson
"""
# %%
from acq_class import (MiniAnalysis, PostSynapticEvent, CurrentClamp,
                       LoadMiniAnalysis, LoadMini, LFP, oEPSC,
                       LoadCurrentClamp, Acquisition)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal, integrate
import numpy as np
import bottleneck as bn
import pandas as pd
from pathlib import PurePath
from utilities import load_scanimage_file

# %%
path_test = PurePath(r'D:\Lars Slice Ephys\2021_11_18\AD0_1')
acq_test = load_scanimage_file(path_test)

# %%
xy = MiniAnalysis(path, 10000, 0, 80, 'remez_2', order=201,
     low_pass=600,low_width=600, template=None, rc_check=True,
     rc_check_start=10000, rc_check_end=10300, sensitivity=2.5,
     amp_threshold=6, mini_spacing=2, min_rise_time=0.5, min_decay_time=0.8,
     invert=False, curve_fit_decay=False)
xy.analyze()
len(xy.postsynaptic_events)

# %%
#Find baseline improved version
me1 = xy.postsynaptic_events[10]


peaks_1 = signal.argrelextrema(me1.event_array, comparator = np.less,
                  order=int(3*me1.s_r_c))[0]
peaks_1 = peaks_1[peaks_1 > 1*me1.s_r_c]
peak_1 = peaks_1[0]
peaks_2 = signal.argrelextrema(me1.event_array[:peak_1],
                  comparator = np.less, order=int(.5*me1.s_r_c))[0]
peaks_2 = peaks_2[peaks_2 > peak_1-4*me1.s_r_c]
peaks_3 = peaks_2[me1.event_array[peaks_2] < 0.9*me1.event_array[peak_1]]
peak_3 = peaks_3[0]

plt.plot(me1.event_array)
plt.plot(int(peaks_3[0]), me1.event_array[int(peaks_3[0])], 'x')
plt.plot(int(peaks_1[0]), me1.event_array[int(peaks_1[0])], 'x')

def db_exp(x, a_fast, tau1, a_slow, tau2):
    y = (a_fast * np.exp(-x/tau1)) + (a_slow * np.exp(-x/tau2))
    return y


def decay_exp(x, amp, tau, c):
    decay = amp * np.exp(-x/tau) + c
    return decay


def neg_db_exp(t, a_fast, tau_fast, a_slow, tau_slow):
    y = a_fast * np.exp(-t/tau_fast) + a_slow * np.exp(-t/tau_slow)
    return y

xh = Acquisition('AD1', 42, 10000, 0, 1000, 'savgol', order=5, polyorder=3)

oepsc_acq_dict = {}
for i in range(1,81):
    xh = oEPSC('AD1', i, 10000, 0, 1000, 'savgol', order=5, polyorder=3,
               pulse_start=1000, n_window_start=1001)
    oepsc_acq_dict[str(i)] = xh

o_raw_list = pd.DataFrame([oepsc_acq_dict[i].create_dict() for i in \
                             oepsc_acq_dict.keys()])








rms = np.sqrt(np.mean(np.square(
            xh.filtered_array[xh.baseline_start:xh.baseline_end])))
index = np.argmax(abs(xh.filtered_array[xh.peak_x*10:])
                < 2*xh.rms) + xh.peak_x
integrated_signal = integrate.trapz(xh.filtered_array[xh.pulse_start:index],
                        xh.x_array[xh.pulse_start:index])
plt.plot(abs(xh.filtered_array[xh.pulse_start:]))
plt.plot(index, xh.filtered_arra[index], 'x')

est_tau_y = xh.peak_y * (1 / np.exp(1))
decay_y = xh.filtered_array[xh.peak_x:index]
decay_x = np.arange(decay_y.shape[0])
est_tau_x = np.interp(est_tau_y, decay_y, decay_x)
init_param = np.array([xh.peak_y, est_tau_x, 1, 0])
popt, pcov = curve_fit(neg_db_exp, decay_x, decay_y, method='lm')
a_fast, tau_fast, a_slow, tau_slow = popt
# plot_decay_y = decay_exp(decay_x, fit_amp, fit_tau)
plot_decay_y = neg_db_exp(decay_x, a_fast, tau_fast, a_slow, tau_slow)


plt.plot(decay_x, decay_y)
plt.plot(decay_x, plot_decay_y)



plt.plot(xh.filtered_array)
plt.plot(est_tau_x, est_tau_y, 'x')



plt.ion()
plt.plot(xh.array)

acq_dict = {}
for i in range(1, 5):
    print(i)
    xy = MiniAnalysis('AD0', 1, 10000, 0, 80, 'remez_2', order=101, low_pass=600,
        low_width=600, template=None, rc_check=True,
        rc_check_start=10000, rc_check_end=10300, sensitivity=2.5,
        amp_threshold=7, mini_spacing=7.5, min_rise_time=1,
        invert=False)
    acq_dict[str(i)] = xy
    
test = acq_dict['1']

energy = bn.move_mean(np.square(xh.filtered_array), 10, 1)
nonlinear_energy = np.gradient(xh.filtered_array) * xh.filtered_array


xh = CurrentClamp('AD0', 301, 10000, 0, 300)
xh = CurrentClamp('AD0', 83, 10000, 0, 300)
diff1 = np.gradient(xh.array)
diff2 = np.gradient(diff1, xh.array)

fig, ax = plt.subplots()
ax.plot(diff1)





#Kernel density estimation testing
y = raw_df['IEIs'].dropna().to_numpy()[:, np.newaxis]
x = np.arange(y.shape[0])[:, np.newaxis]
kde = KernelDensity(kernel='gaussian').fit(y)
bandwidth = np.arange(0.05, 3, .05)
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(y)
kde = grid.best_estimator_
logprob = kde.score_samples(x)

plt.fill_between(np.arange(y.shape[0]), np.exp(logprob), alpha=0.5)
plt.xlim(np.min(raw_df['IEIs']), np.max(raw_df['IEIs']))
plt.plot(x[:,0], np.exp(logprob))
plt.plot(raw_df['IEIs'], [0]*len(raw_df['IEIs']), '|', color='black')

raw_df

sns.histplot(data=raw_df, x='Amplitudes', kde=True)



plt.plot(np.arange(len(average))/10, average)
plt.plot(decay_x, fit_decay_y)



# %%
