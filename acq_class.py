# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:25:54 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: LarsNelson
"""

import bottleneck as bn
import numpy as np
from scipy import signal, stats, integrate
from scipy.fft import fft, ifft
from scipy.stats import linregress
from scipy.optimize import curve_fit

from filtering_functions import (median_filter,
                                 remez_1,
                                 remez_2,
                                 fir_zero_1,
                                 fir_zero_2,
                                 bessel)
from utilities import return_acq_components


class Acquisition:
    '''
    This is the base class for acquisitions. It returns the array from a
    matfile and filters the array.
    
    To remove DC from the signal, signal is baselined to the mean of the 
    chosen baseline of the array. A highpass filter is usually not needed to
    for offline analysis because the signal can baselined using the mean.
    '''
    
    
    def __init__(self, acq_components, sample_rate, baseline_start,
                 baseline_end, filter_type='None', order=None, high_pass=None,
                 high_width=None, low_pass=None, low_width=None, window=None,
                 polyorder=None):
        self.sample_rate = sample_rate
        self.name = acq_components[0]
        self.acq_number = acq_components[1]
        self.array = acq_components[2]
        self.epoch = acq_components[3]
        self.pulse_pattern = acq_components[4]
        self.ramp = acq_components[5]
        self.pulse_amp = acq_components[6]
        self.time_stamp = acq_components[7]
        self.s_r_c = sample_rate/1000
        self.x_array = np.arange(len(self.array))/(sample_rate/1000)
        self.baseline_start = int(baseline_start*(sample_rate/1000))
        self.baseline_end = int(baseline_end*(sample_rate/1000))
        self.filter_type = filter_type
        self.order = order
        self.high_pass = high_pass
        self.high_width = high_width
        self.low_pass = low_pass
        self.low_width = low_width
        self.window = window
        self.polyorder=polyorder
        self.baselined_array = (self.array
            - np.mean(self.array [self.baseline_start:self.baseline_end]))
    
        
    def filter_array(self):
        '''
        This funtion filters the array of data, with several different types
        of filters.
        
        median: is a filter that return the median for a specified window
        size. Needs an odd numbered window.
        
        bessel: modeled after the traditional analog minimum phase filter.
        Needs to have order, sample rate, high pass and low pass settings.
        
        fir_zero_1: zero phase phase filter that filter backwards and forwards
        to achieve zero phase. The magnitude of the filter is sqaured due to
        the backwards and forwards filtering. Needs sample rate, order, high
        pass, width of the cutoff region, low pass, width of the low pass
        region and the type of window. Windows that are currently supported
        are hann, hamming, nutall, flattop, blackman. Currently does not
        support a kaiser filter.
        
        fir_zero_2: An almost zero phase filter that filters only in the
        forward direction. The zero phase filtering only holds true for odd
        numbered orders. The zero phase filtering is achieved by adding a set
        amount of values ((order-1)/2 = 5) equal to the last value of the array to the
        ending of the array. After the signal has been filtered, the same
        same number of values are removed from the beginning of the array thus
        yielding a zero phase filter.
        
        remez_1: A zero phase FIR filter that does not rely on windowing. The
        magnitude of the filter is squared since it filters forward and
        backwards. Uses the same arguments as fir_zero_1/2, but does not need
        a window type.
        
        remez_2: An almost zero phase filter similar fir_zero_2 except that
        it does not need a window type.
        
        savgol: This a windowed polynomial filter called the Savitsky-Golay
        filter. It fits a polynomial of specified number to a specified
        window. Please note the order needs to be larger than the polyorder.
        
        none: No filtering other than baselining the array.
        
        subtractive: This filter is more experimental. Essentially you filter
        the array to create an array of frequency that you do not want then
        subtract that from the unfiltered array to create a filtered array
        based on subtraction. Pretty esoteric and is more for learning
        purposes.
        '''
        
        
        if self.filter_type == 'median':
            self.filtered_array = median_filter(self.baselined_array,
                                                self.order)
        elif self.filter_type == 'bessel':
            self.filtered_array = bessel(self.baselined_array,
                                         self.order,
                                         self.sample_rate,
                                         self.high_pass,
                                         self.low_pass)
        elif self.filter_type == 'fir_zero_1':
            self.filtered_array = fir_zero_1(self.baselined_array,
                                             self.sample_rate,
                                             self.order,
                                             self.high_pass,
                                             self.high_width,
                                             self.low_pass,
                                             self.low_width,
                                             self.window)
        elif self.filter_type == 'fir_zero_2':
            self.filtered_array = fir_zero_2(self.baselined_array,
                                             self.sample_rate,
                                             self.order,
                                             self.high_pass,
                                             self.high_width,
                                             self.low_pass,
                                             self.low_width,
                                             self.window)
        elif self.filter_type == 'remez_1':
            self.filtered_array = remez_1(self.baselined_array,
                                          self.sample_rate,
                                          self.order,
                                          self.high_pass,
                                          self.high_width,
                                          self.low_pass,
                                          self.low_width)
        elif self.filter_type == 'remez_2':
            self.filtered_array = remez_2(self.baselined_array,
                                          self.sample_rate,
                                          self.order,
                                          self.high_pass,
                                          self.high_width,
                                          self.low_pass,
                                          self.low_width)
        elif self.filter_type == 'savgol':
            self.filtered_array = signal.savgol_filter(self.baselined_array,
                                                       self.order, 
                                                       self.polyorder,
                                                       mode = 'nearest')
        elif self.filter_type == 'None':
            self.filtered_array = self.baselined_array.copy()
        elif self.filter_type == 'subtractive':
            self.spike_array = fir_zero_2(self.baselined_array,
                                          self.sample_rate,
                                          self.order,
                                          self.high_pass,
                                          self.high_width,
                                          self.low_pass,
                                          self.low_width,
                                          self.window)
            self.filtered_array = self.baselined_array - self.spike_array
        return self.filtered_array


    def s_exp(self, x_array, amp, tau):
        decay = amp * np.exp((-x_array)/tau)
        return decay
    
    
    def db_exp(self, x_array, a_fast, tau1, a_slow, tau2):
        y = (a_fast * np.exp((-x_array)/tau1)) + (a_slow * np.exp((-x_array)/tau2))
        return y
    

class LFP(Acquisition):
    '''
    This class creates a LFP acquisition. This class subclasses the
    Acquisition class and takes input specific for LFP analysis.
    '''
    
    
    def __init__(self, filter_type='None', order=None, high_pass=None,
                 high_width=None, low_pass=None, low_width=None, window=None,
                 polyorder=None, pulse_start=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_start = int(pulse_start*self.s_r_c)
        self.fp_x = np.nan
        self.fp_y = np.nan
        self.fv_x = np.nan
        self.fv_y = np.nan
        self.max_x = np.nan
        self.max_y = np.nan
        self.slope_y = np.nan
        self.slope_x = np.nan
        self.b = np.nan
        self.slope = np.nan
        self.regression_line = np.nan
        self.filter_array()
        self.analyze_lfp()
        
    
    def field_potential(self):
        '''
        This function finds the field potential based on the largest value in
        the array.

        Returns
        -------
        None.

        '''
        #The end value was chosed based on experimental data.
        w_end = self.pulse_start + 200
        
        #The start value was chosen based on experimental data.
        w_start = self.pulse_start + 44
        
        if (abs(np.max(self.filtered_array[w_start:w_end]))
            < abs(np.min(self.filtered_array[w_start:w_end]))):
            self.fp_y = np.min(self.filtered_array[w_start:w_end])
            self.fp_x = np.argmin(self.filtered_array[w_start:w_end])+w_start
        else:
            self.fp_y = np.nan
            self.fp_x = np.nan
    
    
    def fiber_volley(self):
        '''
        This function finds the fiber volley based on the position of the
        field potential. The window for finding the fiber volley is based on
        experimental data.

        Returns
        -------
        None.

        '''
        w_start = (self.pulse_start + 9)
        if self.fp_x < (self.pulse_start + 74):
            w_end = self.fp_x-20
        else: 
            w_end = self.fp_x-40
            if w_end > (self.pulse_start + 49): 
                w_end = (self.pulse_start + 49)
        if self.fp_x is np.nan or None:
            self.fv_y = np.nan
            self.fv_x = np.nan
        else:
            self.fv_y = np.min(self.filtered_array[w_start:w_end])
            self.fv_x = (np.argmin(self.filtered_array[w_start:w_end])
                         +w_start
                          )


    def find_slope_array(self, w_end):
        '''
        This function returns the array for slope of the field potential onset
        based upon the 10-90% values of the field potential rise. The field
        potential x coordinate needs to be passed to this function.

        Returns
        -------
        None.

        '''
        start = self.pulse_start
        # baseline_start = start - int(15.1*self.s_r_c)
        # analysis_end = start + int(20.0*self.s_r_c)
        x_values = np.arange(0, len(self.filtered_array)-1)
        if w_end is not np.nan:
        # if (abs(np.max(self.filtered_array[baseline_start:analysis_end]))
        #     < abs(np.min(self.filtered_array[baseline_start:analysis_end]))):
        #     w_end = (np.argmin(self.filtered_array[(start+ 44):analysis_end])
        #              + (start+ 44)
        #               )
            if w_end < (start + 74):
                x = w_end-19
            else: 
                x = w_end-39
                if x > (start+49):
                    x = (start+49)
            w_start = (np.argmin(self.filtered_array[(start + 9):x])
                       +(start + 9)
                        )
            if w_end < w_start:
                self.slope_y = [np.nan]
                self.slope_x = [np.nan]
                self.max_x= [np.nan]
                self.max_y = [np.nan]
            else:
                self.max_x = (np.argmax(self.filtered_array[w_start:w_end])
                    + w_start
                     )
                self.max_y = np.max(self.filtered_array[w_start:w_end])
                y_array_subset = self.filtered_array[self.max_x:w_end]
                x_array_subset = x_values[self.max_x:w_end]
                self.slope_y = y_array_subset[int(len(y_array_subset)
                    * .1):int(len(y_array_subset) * .9)
                     ]
                self.slope_x = x_array_subset[int(len(x_array_subset)
                    * .1):int(len(x_array_subset) * .9)
                     ]
        else:
            self.max_x = [np.nan]
            self.max_y = [np.nan]
            self.slope_y = [np.nan]
            self.slope_x = [np.nan]
    
    
    def regression(self):
        '''
        This function runs a regression on the slope array created by the
        find_slop_array function.

        Returns
        -------
        None.

        '''
        # if np.isnan(self.slope_y):
        #     self.b = np.nan
        #     self.slope = np.nan
        #     self.reg_line = np.nan
        if len(self.slope_y) > 5:
            reg = linregress(self.slope_x, self.slope_y)
            self.slope = reg[0]
            self.reg_line = [(self.slope*i) + reg[1] for i in self.slope_x]
            
            # x_constant = sm.add_constant(self.slope_x)
            # model_1 = sm.RLM(self.slope_y, x_constant,
            #                  M=sm.robust.norms.TrimmedMean())
            # results_1 = model_1.fit()
            # self.b, self.slope = results_1.params
            # self.regression_line = [
            #     (self.slope*i) + self.b for i in self.slope_x]
        else:
            self.b = np.nan
            self.slope = np.nan
            self.reg_line = np.nan
    

    def analyze_lfp(self):
        '''
        This function runs all the other functions in one place. This makes
        it easy to troubleshoot.
        '''
        self.field_potential()
        if self.fp_x is np.nan:
            pass
        else:
            self.fiber_volley()
            self.find_slope_array(self.fp_x)
            self.regression()


    def change_fv(self, x, y):
        self.fv_x = x
        self.fv_y = y
     
        
    def change_fp(self, x, y):
         self.fp_x = x
         self.fp_y = y
     
     
    def plot_elements_x(self):
        return [self.fv_x/self.s_r_c,
                self.fp_x/self.s_r_c
                ]
    
    
    def plot_elements_y(self):
        return [self.fv_y, self.fp_y]


    def create_dict(self):
        '''
        This function returns a dictionary of all the values you will need to
        analyze the lfp. The dictionary is not an attribute of the class to
        keep the size of the created object small.

        Returns
        -------
        lfp_dict : TYPE
            DESCRIPTION.

        '''
        lfp_dict = {'fv_amp': self.fv_y,
                    'fv_time': self.fv_x/self.s_r_c,
                    'fp_amp': self.fp_y,
                    'fp_time': self.fp_x/self.s_r_c,
                    'fp_slope': self.slope,
                    'Epoch': self.epoch,
                    'Acq number': self.acq_number
                    }
        return lfp_dict


class oEPSC(Acquisition):
    def __init__(self, pulse_start=1000, n_window_start=1001,
                 n_window_end=1050, p_window_start=1045,
                 p_window_end=1055, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_start = int(pulse_start*self.s_r_c)
        self.n_window_start = int(n_window_start*self.s_r_c)
        self.n_window_end = int(n_window_end*self.s_r_c)
        self.p_window_start = int(p_window_start*self.s_r_c)
        self.p_window_end = int(p_window_end*self.s_r_c)
        self.filter_array()
        self.baseline_mean = np.mean(
            self.filtered_array[self.baseline_start:self.baseline_end]
            )
        self.analyze_oepsc()
        
    
    def peak_direction(self):
        if abs(max(self.filtered_array)) > abs(min(self.filtered_array)):
            self.peak_direction = 'positive'
            self.peak_direction = 'positive'
        else:
            self.peak_direction = 'negative'
    
    
    def find_amplitude(self):
        if self.peak_direction == 'positive':
            self.peak_y = np.max(
                self.filtered_array[self.p_window_start:self.p_window_end])
            self.peak_x = ((np.argmax(
                self.filtered_array[self.p_window_start:self.p_window_end])
                + self.p_window_start) / self.s_r_c
                )
        elif self.peak_direction == 'negative':
            self.peak_y = np.min(
                self.filtered_array[self.n_window_start:self.n_window_end])
            self.peak_x = ((np.argmin(
                self.filtered_array[self.n_window_start:self.n_window_end])
                + self.n_window_start) / self.s_r_c
                )
        
    
    def analyze_oepsc(self):
        self.peak_direction()
        self.find_amplitude()
        self.find_charge_transfer()
        
        
    def find_charge_transfer(self):
        self.rms = np.sqrt(np.mean(np.square(
            self.filtered_array[self.baseline_start:self.baseline_end])))
        try:
            index = (np.argmax(abs(self.filtered_array[int(
                self.peak_x*self.s_r_c):])
                < 2*self.rms) + int(self.peak_x*self.s_r_c)
                )
            self.charge_transfer = integrate.trapz(
                self.filtered_array[self.pulse_start:index], 
                self.x_array[self.pulse_start:index]
                )
        except:
            self.charge_transfer = np.nan
        
    
    def plot_y(self):
        return self.filtered_array[self.baseline_start:]
    
    
    def plot_x(self):
        return self.x_array[self.baseline_start:]
    
    def plot_peak_x(self):
        return [self.peak_x]
        
    def plot_peak_y(self):
        return [self.peak_y]
    
    def create_dict(self):
        oepsc_dict = {'Amplitude': abs(self.peak_y),
                      'Charge_transfer': self.charge_transfer,
                      'Epoch': self.epoch,
                      'Acq number': self.acq_number,
                      'Peak direction': self.peak_direction
                      }
        return oepsc_dict
    

class CurrentClamp(Acquisition):
    def __init__(self, sample_rate, baseline_start,
                 baseline_end, filter_type='None', order=None, high_pass=None,
                 high_width=None, low_pass=None, low_width=None, window=None, 
                 polyorder=None, pulse_start=300, pulse_end=1000, 
                 ramp_start=300, ramp_end=4000, threshold=-15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_start = int(pulse_start*self.s_r_c)
        self.pulse_end = int(pulse_end*self.s_r_c)
        self.ramp_start = int(ramp_start*self.s_r_c)
        self.ramp_end = int(ramp_end*self.s_r_c)
        self.x_array = np.arange(len(
            self.array))/(self.s_r_c)
        self.threshold=threshold
        self.get_delta_v()
        self.find_spike_parameters()
        self.first_spike_parameters()
        self.plot_delta_v()
        self.get_ramp_rheo()
        self.find_AHP_peak()
        self.spike_adaptation()
        self.calculate_sfa_local_var()
        self.calculate_sfa_divisor()
    
    
    def get_delta_v(self):
        '''
        This function finds the delta-v for a pulse. It simply takes the mean
        value from the pulse start to end for pulses without spikes. For
        pulses with spikes it takes the mode of the moving mean.
        
        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if self.ramp == '0':    
            self.baseline_mean = np.mean(
                self.array[self.baseline_start:self.baseline_end])
            max_value = np.max(self.array[self.pulse_start:self.pulse_end])
            self.start = int(self.pulse_start
                + ((self.pulse_end - self.pulse_start)/2))
            if max_value < self.threshold:
                self.delta_v = (np.mean(self.array[self.start:self.pulse_end])
                    - self.baseline_mean)
            else:
                m = stats.mode(bn.move_mean(
                    self.array[self.start:self.pulse_end], window=1000,
                    min_count=1))
                self.delta_v = m[0][0] - self.baseline_mean
        elif self.ramp == '1':
            self.delta_v = np.nan
            self.baseline_mean = np.mean(
                self.array[self.baseline_start:self.baseline_end])
        return self.delta_v
    
    
    def find_spike_parameters(self):
        '''
        This function returns the spike parameters of a pulse or ramp that 
        spikes. A separate function characterizes the first spike in a train
        of spikes. This function is to determine whether spikes exist.

        Returns
        -------
        self.rheo_x
            The x position in an array of the spike threshold. Also used to 
            calculate the rheobase for a ramp.
        self.spike_threshold
            The threshold at which the first spike occurs.
        self.peaks
            The x_position of the action potential peaks. This is used to
            calculate the self.hertz_exact and self.iei
        '''
        #Find the peaks of the spikes. The prominence is set to avoid picking
        #peaks that are just noise.
        if self.ramp == '0':
            self.peaks, _ = signal.find_peaks(self.array[:self.pulse_end], 
                height = self.threshold, prominence=10)
        elif self.ramp == '1':
            self.peaks, _ = signal.find_peaks(self.array[:self.ramp_end], 
                height = self.threshold, prominence=10)
        if len(self.peaks) == 0:
            #If there are no peaks fill in values with np.nan. This helps with
            #analysis further down the line as nan values are fairly easy to
            #work with.
            self.peaks = [np.nan]
            self.spike_threshold = np.nan
            self.rheo_x = np.nan
            self.hertz_exact = np.nan
            self.iei = [np.nan]
            self.iei_mean = np.nan
            self.ap_v = np.nan
        else:
            #Differentiate the array to find the peak dv/dt.
            dv = np.gradient(self.array)
            dt = np.gradient(np.arange(len(self.x_array))/10)
            peak_dv, _ = signal.find_peaks(dv, height = 6)
            
            #Pull out the index of the first peak and find the peak velocity.
            self.ap_v = (dv/dt)[peak_dv[0]]
            
            #Calculate this early so that it does not need to be calculated
            #a second time.
            baseline_std = np.std(
                dv[self.baseline_start:self.baseline_end])
            
            #Calculate the IEI and correction for sample rate
            if len(self.peaks) > 1:
                self.iei = np.diff(self.peaks)/self.s_r_c
                self.iei_mean = self.iei.mean()
            else:
                self.iei = [np.nan]
                self.iei_mean = np.nan
    
            #While many papers use a single threshold to find the threshold
            #potential this does not work if you want to analyze both
            #interneurons and other neuron types. I have created a shifting
            #threshold based on where the maximum velocity occurs of the first
            #spike occurs. 
            if self.ramp == '0':
                if (peak_dv[0] < self.pulse_start + 200 
                    and peak_dv[0] > self.pulse_start + 30):
                    multiplier = 20
                elif peak_dv[0] <= self.pulse_start + 30:
                    multiplier = 22
                elif peak_dv[0] >= self.pulse_start + 200:
                    multiplier = 6
                    
                #This takes the last value of an array of values that are 
                #less than the threshold. It was the most robust way to find
                #the spike threshold time.
                self.rheo_x = (np.argwhere(dv[self.pulse_start:peak_dv[0]]
                    < (multiplier*baseline_std))
                    + self.pulse_start)[-1][0]
                
                #Find the spike_threshold using the timing found above
                self.spike_threshold = self.array[self.rheo_x]
                
                #Calculates the exact hertz by dividing the number peaks by
                #length of the pulse. If there is only one spike the hertz
                #returns an impossibly fast number is you take only divide by
                #the start of the spike_threshold to the end of the pulse.
                self.hertz_exact = len(self.peaks)/((self.pulse_end
                    - self.pulse_start)/self.sample_rate)
            
            elif self.ramp == '1':
                #This takes the last value of an array of values that are 
                #less than the threshold. It was the most robust way to find
                #the spike threshold time.
                self.rheo_x = (np.argwhere(dv[self.pulse_start:peak_dv[0]]
                    < (8*baseline_std))
                    + self.pulse_start)[-1][0]
                self.spike_threshold = self.array[self.rheo_x]
                self.hertz_exact = len(self.peaks)/((self.ramp_end
                    - self.rheo_x)/self.sample_rate)   
        # return (self.rheo_x, self.spike_threshold, self.hertz_exact,
        #         self.peaks, self.spike_width, self.spike_width_ave, self.ap_v,
        #         self.iei)
    
    
    def first_spike_parameters(self):
        '''
        This function analyzes the parameter of the first action potential in
        a pulse that contains at least one action potential.

        Returns
        -------
        None.

        '''
        if self.peaks[0] is np.nan:
            #If there are no peaks fill in values with np.nan. This helps with
            #analysis further down the line as nan values are fairly easy to
            #work with.
            self.spike_width = np.nan
            self.spike_width_ave = np.nan
            self.first_ap = [np.nan]
            self.indices = np.nan
            self.peak_volt = np.nan
        else:
            self.peak_volt = self.array[self.peaks[0]]
            if self.ramp == '0':
                #To extract the first action potential and to find the
                #half-width of the spike you have create an array whose value
                #is the spike threshold wherever the value drops below the
                #spike threshold. This is used because of how scipy.find_peaks
                #works and was a robust way to find the first
                #action_potential.
                masked_array = self.array.copy()
                mask = np.array(self.array > self.spike_threshold)
                
                #First using a mask to find the indices of each action
                #potential.
                self.indices = np.nonzero(mask[1:] != mask[:-1])[0]
                if len(self.indices)>2:
                    self.indices = self.indices[self.indices>=self.rheo_x]
                    ap_index = [self.indices[0] 
                        - int(5*self.s_r_c),
                        self.indices[2]]
                else:
                    ap_index = [self.indices[0]
                        - int(5*self.s_r_c), self.pulse_end]
                    
                #Extract the first action potential based on the ap_index.
                self.first_ap = np.split(self.array, ap_index)[1]
                
                #Create the masked array using the mask found earlier to find
                #The pulse half-width.
                masked_array[~mask] = self.spike_threshold
                self.widths = signal.peak_widths(
                    masked_array[:self.pulse_end], self.peaks,
                    rel_height=0.5)[0]
                self.spike_width = self.widths[0]/self.s_r_c
            
            elif self.ramp == '1':
                #To extract the first action potential and to find the
                #half-width of the spike you have create an array whose value
                #is the spike threshold wherever the value drops below the
                #spike threshold. This is used because of how scipy.find_peaks
                #works and was a robust way to find the first action_potential.
                masked_array = self.array.copy()
                
                #First using a mask to find the indices of each action
                #potential. The index pulls out the action potential fairly
                #close to the spike so the first index is set to 5 ms before
                #the returned index.
                mask = np.array(self.array > self.spike_threshold)
                self.indices = np.nonzero(mask[1:] != mask[:-1])[0]
                if len(self.indices)>2:
                    self.indices = self.indices[self.indices>=self.rheo_x]
                    ap_index = [self.indices[0]
                        - int(5*self.s_r_c), self.indices[2]]
                else:
                    ap_index = [self.indices[0]
                        - int(5*self.s_r_c), self.ramp_end]
               
                #Extract the first action potential based on the ap_index.
                self.first_ap = np.split(self.array, ap_index)[1]
                
                #Create the masked array using the mask found earlier to find
                #The pulse half-width.
                masked_array[~mask] = self.spike_threshold
                self.widths = signal.peak_widths(masked_array[:self.ramp_end],
                    self.peaks, rel_height=0.5)[0]
                self.spike_width = self.widths[0]/self.s_r_c
                
                
    
    
    def spike_adaptation(self):
        '''
        This function calculates the spike frequency adaptation. A positive
        number means that the spikes are speeding up and a negative number
        means that spikes are slowing down. This function was inspired by the
        Allen Brain Institutes IPFX analysis program
        https://github.com/AllenInstitute/ipfx/tree/
        db47e379f7f9bfac455cf2301def0319291ad361
        '''
        
        if len(self.iei) <= 1:
            self.spike_adapt = np.nan
        else:
            # self.iei = self.iei.astype(float)
            if np.allclose((self.iei[1:] + self.iei[:-1]), 0.):
                self.spike_adapt = np.nan
            norm_diffs = (self.iei[1:] - self.iei[:-1]) / (self.iei[1:]
                                                           + self.iei[:-1])
            norm_diffs[(self.iei[1:] == 0) & (self.iei[:-1] == 0)] = 0.
            self.spike_adapt = np.nanmean(norm_diffs)
            return self.spike_adapt

    
    def plot_delta_v(self):
        '''
        This function creates the elements to plot the delta-v as a vertical
        line in the middle of the pulse. The elements are corrected so that
        they will plot in milliseconds.
        '''
        if self.ramp == '0':
            x = int(((self.pulse_end - self.pulse_start)/2)
                         + self.pulse_start)/self.s_r_c
            voltage_response = self.delta_v + self.baseline_mean
            self.plot_x = [x, x]
            self.plot_y = [self.baseline_mean, voltage_response]
        elif self.ramp == '1':
            self.plot_x = np.nan
            self.plot_y = np.nan
    
    
    def get_ramp_rheo(self):
        '''
        This function gets the ramp rheobase. The ramp pulse is recreated
        based on the values in the matfile and then the current is extracted.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if self.ramp == '1':
            if self.rheo_x is np.nan:
                self.ramp_rheo = np.nan
            else:
                #Create the ramp current values.
                ramp_values = np.linspace(0, int(self.pulse_amp), 
                                          num = self.ramp_end - self.ramp_start)
                
                #Create an array of zeros where the ramp will be placed.
                ramp_array = np.zeros(len(self.array))
                
                #Insert the ramp into the array of zeros.
                ramp_array[self.ramp_start:self.ramp_end] = ramp_values
                
                #Extract the ramp rheobase.
                self.ramp_rheo = ramp_array[self.rheo_x]
        else: 
            self.ramp_rheo = np.nan
        return self.ramp_rheo
    
    
    def calculate_sfa_local_var(self):
        '''
        The idea for the function was initially inspired by a program called
        Easy Electropysiology (https://github.com/easy-electrophysiology).
        
        This function calculates the local variance in spike frequency
        accomadation that was drawn from the paper:
        Shinomoto, Shima and Tanji. (2003). Differences in Spiking Patterns
        Among Cortical Neurons. Neural Computation, 15, 2823-2842.

        Returns
        -------
        None.

        '''
        if len(self.iei) < 2 or self.iei is np.nan:
            self.local_var = np.nan
        else:
            isi_shift = self.iei[1:]
            isi_cut = self.iei[:-1]
            n_minus_1 = len(isi_cut)
            self.local_var = (np.sum((3*(isi_cut - isi_shift)**2)
                              /(isi_cut + isi_shift)**2)/
                              n_minus_1)
    
    
    def calculate_sfa_divisor(self):
        '''
        The idea for the function was initially inspired by a program called
        Easy Electropysiology (https://github.com/easy-electrophysiology).
        '''
        self.sfa_divisor = self.iei[0]/self.iei[-1]
        
        
    def find_AHP_peak(self):
        '''
        Rather than divide the afterhyperpolarization potential into different
        segments it seems best to pull out the peak of the AHP and its timing
        compared to the the first spike or spike threshold. It seems to me to
        be less arbitrary.
        '''
        if self.peaks[0] is not np.nan:
            self.ahp = np.min(self.array[self.peaks[0]:])
        else:
            self.ahp = np.nan
    
    
    def create_dict(self):
        '''
        This create a dictionary of all the values created by the class. This
        makes it very easy to concentatenate the data from multiple
        acquisitions together.

        Returns
        -------
        None.

        '''
        current_clamp_dict = {
            'Acquisition': self.acq_number,
            'Pulse_pattern': self.pulse_pattern,
            'Pulse_amp': self.pulse_amp,
            'Ramp': self.ramp,
            'Epoch': self.epoch,
            'Baseline': self.baseline_mean,
            'Delta_v': self.delta_v, 
            'Spike_threshold': self.spike_threshold,
            'Spike_peak_volt': self.peak_volt,
            'Hertz': self.hertz_exact,
            'Spike_iei': self.iei_mean,
            'Spike_width': self.spike_width,
            'Max_AP_vel': self.ap_v,
            'Spike_freq_adapt': self.spike_adapt,
            'Local_sfa': self.local_var,
            'Divisor_sfa': self.sfa_divisor,
            'Peak_AHP': self.ahp,
            'Ramp_rheobase': self.ramp_rheo}
        return current_clamp_dict
            

class MiniAnalysis(Acquisition):
    def __init__(self, template=None, rc_check=False, rc_check_start=None, 
                 rc_check_end=None, sensitivity=3, amp_threshold=7,
                 mini_spacing=7.5, min_rise_time=1, min_decay_time= 2,
                 invert=False, decon_type='weiner', curve_fit_decay=False,
                 curve_fit_type='db_exp', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc_check = rc_check
        self.rc_check_start = int(rc_check_start
                                       * self.s_r_c)
        self.rc_check_end = int(rc_check_end*self.s_r_c)
        self.sensitivity = sensitivity
        self.amp_threshold = amp_threshold
        self.mini_spacing = int(mini_spacing * self.s_r_c)
        self.min_rise_time = min_rise_time
        self.min_decay_time = min_decay_time
        self.invert = invert
        self.curve_fit_decay = curve_fit_decay
        self.decon_type = decon_type
        self.curve_fit_type = curve_fit_type
        self.create_template(template)
    
        
    def analyze(self):
        self.filter_array()
        self.create_mespc_array()
        self.set_sign()
        self.wiener_deconvolution()
        self.wiener_filt()
        self.create_events()

    
    def tm_psp(self, amplitude, tau_1, tau_2, risepower, t_psc, spacer=0):
        template = np.zeros(len(t_psc)+spacer)
        offset = len(template)-len(t_psc)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        y = amplitude/Aprime*((1-(np.exp(-t_psc/tau_1)))**risepower
                              * np.exp((-t_psc/tau_2)))
        template[offset:] = y
        return template
    
    
    def create_template(self, template):
        if template is None:
            tau_1 = 3
            tau_2 = 50
            amplitude = -20
            risepower = 0.5
            t_psc = np.arange(0, 300)
            spacer = 20
            self.template = self.tm_psp(amplitude, tau_1, tau_2, risepower, t_psc,
                        spacer=spacer)
        else:
            self.template = template
    
    
    def create_mespc_array(self):
        if self.rc_check is False:
            self.final_array = np.copy(self.filtered_array)
        elif self.rc_check is True:
            if self.rc_check_end == len(self.filtered_array):
                self.final_array = np.copy(
                    self.filtered_array[:self.rc_check_start])
                self.rc_check_array = np.copy(self.array[self.rc_check_start:])
            else:
                self.final_array = np.copy(
                    self.filtered_array[self.rc_check_end:])
                self.rc_check_array = np.copy(self.array[self.rc_check_end:])
        self.x_array = (np.arange(len(self.final_array))
                    / (self.s_r_c))
        return self.final_array, self.x_array
    
    
    def set_sign(self):
        if not self.invert:
            self.final_array = self.final_array * 1
        else:
            self.final_array = self.final_array * -1
    
    
    def wiener_deconvolution(self, lambd=4):
        '''
        The Wiener deconvolution equation can be found on GitHub from pbmanis
        and danstowell. The basic idea behind this function is deconvolution
        or divsion in the frequency domain. I have found that changing lambd
        from 2-10 does not seem to affect the performance of the Wiener
        equation. The fft deconvolution type is the most simple and default
        choice.
        
        Parameters
        ----------
        array : Filtered signal in a numpy array form. There are edge effects if
            an unfiltered signal is used.
        kernel : A representative PSC or PSP. Can be an averaged or synthetic
            template.
        lambd : Signal to noise ratio. A SNR anywhere from 1 to 10 seems to work
            without determining the exact noise level.
            
        Returns
        -------
        deconvolved_array: numpy array
            Time domain deconvolved signal that is returned for filtering.
            
        '''
        
        kernel = np.hstack((self.template, np.zeros(len(self.final_array)
                                                   -len(self.template))))
        H = fft(kernel)
        if self.decon_type == 'fft':
            self.deconvolved_array = np.real(ifft(fft(self.final_array)/H))
        else:
            self.deconvolved_array = np.real(ifft(fft(self.final_array)
                                                  * np.conj(H)/(H*np.conj(H)
                                                  + lambd**2)))
        # return self.deconvolved_array
    
    
    def wiener_filt(self):
        '''
        This function takes the deconvolved array, filters it and finds the
        peaks of the which is where mini events are located.
        
        Parameters
        ----------
        array : Filtered signal in a numpy array form.There are edge effects if
            an unfiltered signal is used.
        template : A representative PSC or PSP. Can be an averaged or synthetic
            template. The template works best when there is a small array of 
            before the mini onset.
    
        Returns
        -------
        peaks : PSC or PSP peaks.
        y1 : Wiener deconvolved signal.
    
        '''
        baselined_decon_array = (self.deconvolved_array
                                 - np.mean(self.deconvolved_array[0:800]))
        filt = signal.firwin2(201, freq=[0, 300, 600, self.sample_rate/2],
                              gain=[1, 1, 0, 0], window='hann',
                              fs=self.sample_rate)
        y = signal.filtfilt(filt, 1.0, baselined_decon_array)
        self.final_decon_array = signal.detrend(y, type='linear')
        mu, std = stats.norm.fit(self.final_decon_array)
        peaks, _ = signal.find_peaks(self.final_decon_array, 
                                          height=self.sensitivity * abs(std))
        self.events = peaks.tolist()
        return self.events, self.final_decon_array


    def create_events(self):
        '''
        This functions creates the events based on the list of peaks found
        from the deconvolution. Events less than 20 ms before the end of
        the acquisitions are not counted. Events get screened out based on the
        amplitude, min_rise_time, and min_decay_time passed by the
        experimenter. 

        Returns
        -------
        None.

        '''
        self.postsynaptic_events = []
        self.final_events = []
        event_number = 0
        event_time = []
        if len(self.events) == 0:
            pass
        elif len(self.events) > 1:
            for count, peak in enumerate(self.events):
                if (len(self.final_array)-peak
                    < 20*self.s_r_c):
                    pass
                else:
                    event = PostSynapticEvent(self.acq_number, peak, 
                                              self.final_array,
                                              self.sample_rate,
                                              self.curve_fit_decay,
                                              self.curve_fit_type)
                    if (event.event_peak_x is np.nan
                        or event.event_peak_x in event_time):
                        pass
                    else:
                        if count > 0:
                            if (self.events[count] - self.events[count-1] 
                                < self.mini_spacing):
                                pass
                            else:
                                if (event.amplitude > self.amp_threshold
                                    and event.rise_time > self.min_rise_time
                                    and event.final_tau_x > self.min_decay_time):
                                    self.postsynaptic_events += [event]
                                    self.final_events += [event.event_peak_x]
                                    event_time += [event.event_peak_x]
                                    event_number +=1
                                else:
                                    pass
                        else:
                            if event.amplitude > self.amp_threshold:
                                    self.postsynaptic_events += [event]
                                    self.final_events += [event.event_peak_x]
                                    event_time += [event.event_peak_x]
                                    event_number +=1
                            else:
                                pass
        else:
            peak = self.events[0]
            event = PostSynapticEvent(self.acq_number, peak, 
                                      self.final_array,
                                      self.sample_rate,
                                      self.curve_fit_decay,
                                      self.curve_fit_type)
            event_time += [event.event_peak_x]
            if (event.event_peak_x is np.nan
                or event.event_peak_x in event_time):
                pass
            else:
                if (event.amplitude > self.amp_threshold
                    and event.rise_time > self.min_rise_time
                    and event.final_tau_x > self.min_decay_time):
                        self.postsynaptic_events += [event]
                        self.final_events += [event.event_peak_x]
                        event_time += [event.event_peak_x]
                        event_number +=1
                else:
                    pass
    
    
    def create_new_mini(self, x):
        event = PostSynapticEvent(self.acq_number, x, self.final_array,
                                  self.sample_rate, self.curve_fit_decay,
                                  self.curve_fit_type)
        self.final_events += [event.event_peak_x]
        self.postsynaptic_events += [event]
    
    
    def final_acq_data(self):
        '''
        Creates the final data using list comprehension by looping over each
        of the minis in contained in the postsynaptic event list.
        '''
        #Sort postsynaptic events before calculating the final results. This
        #is because of how the user interface works and facilates commandline
        #usage of the program. Essentially it is easier to just add new minis
        #to the end of the postsynaptic event list. This prevents a bunch of
        #issues since you cannot modify the position of plot elements in the 
        #pyqtgraph data items list.
        self.postsynaptic_events.sort(key=lambda x: x.event_peak_x)
        self.final_events.sort()
        self.acq_number_list = [i.acq_number for i in
                    self.postsynaptic_events]
        self.amplitudes = [i.amplitude for i in 
                    self.postsynaptic_events]
        self.taus = [i.final_tau_x for i in 
                    self.postsynaptic_events]
        self.event_times = [i.event_peak_x
                    /self.s_r_c for i in 
                    self.postsynaptic_events]
        self.time_stamp_events = [self.time_stamp for i in 
                    self.postsynaptic_events]
        self.rise_times = [i.rise_time for i in
                           self.postsynaptic_events]
        self.rise_rates = [i.rise_rate for i in
                           self.postsynaptic_events]
        self.event_arrays = [i.event_array-i.event_start_y
                             for i in self.postsynaptic_events]
        self.peak_align_values = [i.event_peak_x - i.array_start for i in
                           self.postsynaptic_events]
        self.iei = np.append(np.diff(self.event_times), np.nan) 
        self.freq = (len(self.amplitudes)
                     /(len(self.final_array)/self.sample_rate))
    
    
    def save_postsynaptic_events(self):
        '''
        This helper function is called when you want to save the file. This
        makes the size of the file smaller so it is of more managable size.
        All the data that is need to recreate the minis is saved.

        Returns
        -------
        None.

        '''
        self.saved_events_dict = []
        self.final_decon_array = 'saved'
        self.deconvolved_array = 'saved'
        self.filtered_array = 'saved'
        self.events = 'saved'
        self.x_array = 'saved'
        self.event_arrays = 'saved'
        for i in self.postsynaptic_events:
            i.x_array = 'saved'
            i.event_array = 'saved'
            self.saved_events_dict += [i.__dict__]
        self.postsynaptic_events = 'saved'
    
    
class PostSynapticEvent():
    '''
    This class creates the mini event.
    '''
    
    def __init__(self, acq_number, event_pos, y_array, sample_rate,
                 curve_fit_decay=False, curve_fit_type='db_exp'):
        self.acq_number = acq_number
        self.event_pos = int(event_pos)
        self.sample_rate = sample_rate
        self.s_r_c = sample_rate/1000
        self.curve_fit_decay = curve_fit_decay
        self.curve_fit_type = curve_fit_type
        self.fit_tau = np.nan
        self.create_event_array(y_array)
        self.find_peak()
        self.find_event_parameters(y_array)
        self.peak_align_value = self.event_peak_x - self.array_start
        
    
    def create_event_array(self, y_array):
        self.array_start = int(self.event_pos 
                                    - (2*self.s_r_c))
        end = int(self.event_pos+(30 * self.s_r_c))
        if end > len(y_array) - 1:
            self.array_end = len(y_array) - 1
        else:
            self.array_end = end
        self.event_array = y_array[self.array_start:self.array_end]
        self.x_array = np.arange(self.array_start, self.array_end)
    
    
    def find_peak(self):
        peaks_1 = signal.argrelextrema(self.event_array, comparator = np.less,
                          order=int(3*self.s_r_c))[0]
        peaks_1 = peaks_1[peaks_1 > 1*self.s_r_c]
        if len(peaks_1) == 0:
            self.event_peak_x = np.nan
            self.event_peak_y = np.nan
        else:
            peak_1 = peaks_1[0]
            peaks_2 = signal.argrelextrema(self.event_array[:peak_1],
                              comparator = np.less,
                              order=int(.4*self.s_r_c))[0]
            peaks_2 = peaks_2[peaks_2 > peak_1-4*self.s_r_c]
            if len(peaks_2) == 0:
                final_peak = peak_1
            else:
                peaks_3 = peaks_2[self.event_array[peaks_2]
                                  < 0.85*self.event_array[peak_1]]
                if len(peaks_3) == 0:
                    final_peak = peak_1
                else:
                    final_peak = peaks_3[0]
            self.event_peak_x = self.x_array[int(final_peak)]
            self.event_peak_y = self.event_array[int(final_peak)]
     
    
    def find_alt_baseline(self):
        baselined_array = self.event_array - np.mean(
            self.event_array[:int(1*self.s_r_c)])
        masked_array = baselined_array.copy()
        mask = np.argwhere(baselined_array <= 0)
        masked_array[mask] = 0 
        peaks = signal.argrelmax(masked_array[0:int(
            self.event_peak_x - self.array_start)], order=2)
        if len(peaks[0]) >  0:
            self.event_start_x = self.x_array[peaks[0][-1]]
            self.event_start_y = self.event_array[peaks[0][-1]]
        else:
            event_start = np.argmax(
                masked_array[0:int(self.event_peak_x - self.array_start)])
            self.event_start_x = self.x_array[event_start]
            self.event_start_y = self.event_array[event_start]
        self.event_baseline = self.event_start_y
    
    
    def find_baseline(self):
        '''
         This functions finds the baseline of an event. The biggest issue with
         most methods that find the baseline is that they assume the baseline
         does not deviate from zero, however this is often not true is real
         life. This methods combines a slope finding method with a peak
         finding method.

        Returns
        -------
        None.

        '''
        baselined_array = self.event_array - np.mean(
            self.event_array[:int(1*self.s_r_c)])
        peak = self.event_peak_x - self.array_start
        search_start = np.argwhere(baselined_array[:peak]
                        > 0.5 * self.event_peak_y).flatten()
        if search_start.size > 0:
            slope = ((self.event_array[search_start[-1]] - self.event_peak_y)
                     /(peak - search_start[-1]))
            new_slope = slope + 1
            i = search_start[-1]
            while new_slope > slope:
                slope = (self.event_array[i] - self.event_peak_y)/(peak - i)
                i -= 1
                new_slope = (self.event_array[i]
                             - self.event_peak_y)/(peak - i)
            baseline_start = signal.argrelmax(
                baselined_array[int(i-1*self.s_r_c):i + 2], order=2)[0]
            if baseline_start.size > 0:
                temp = int(baseline_start[-1] + (i- 1*self.s_r_c))
                self.event_start_x = self.x_array[temp]
                self.event_start_y = self.event_array[temp]
            else:
                temp = int(baseline_start.size/2 + (i- 1*self.s_r_c))
                self.event_start_x = self.x_array[temp]
                self.event_start_y = self.event_array[temp]
        else:
            self.find_alt_baseline()
        
    
    def calc_event_amplitude(self, y_array):
        self.amplitude = abs(self.event_peak_y - self.event_start_y)
    
    
    def calc_event_rise_time(self):
        '''
        This function calculates the rise rate (10-90%) and the rise time
        (end of baseline to peak).

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        end = self.event_peak_x - self.array_start
        start = self.event_start_x - self.array_start
        rise_array = self.event_array[start:end]
        rise_y = rise_array[int(len(
            rise_array) * 0.1):int(len(rise_array) * 0.9)]
        rise_x = (np.arange(int(len(rise_array) * 0.1),
                  int(len(rise_array) * 0.9))
                  + self.event_start_x)/self.s_r_c
        self.rise_time = ((self.event_peak_x - self.event_start_x)
                          /self.s_r_c)
        if len(rise_y) > 3:
            self.rise_rate = abs(linregress(rise_x, rise_y)[0])
        else:
            self.rise_rate = np.nan
        return self.rise_time, self.rise_rate
    
    
    def est_decay(self):
        baselined_event = self.event_array - self.event_start_y
        return_to_baseline = int((np.argmax(
            baselined_event[self.event_peak_x-self.array_start:]
            >= (self.event_peak_y-self.event_start_y)*.25))
            + (self.event_peak_x-self.array_start))
        decay_y = self.event_array[self.event_peak_x 
                          -self.array_start:return_to_baseline]
        if decay_y.size > 0:
            self.est_tau_y = (((self.event_peak_y - self.event_start_y)
                          * (1 / np.exp(1))) + self.event_start_y)
            decay_x = self.x_array[self.event_peak_x
                              - self.array_start:return_to_baseline]
            self.est_tau_x = np.interp(self.est_tau_y, decay_y, decay_x)
            self.final_tau_x = ((self.est_tau_x-self.event_peak_x)
                                /self.s_r_c)
        else:
            self.est_tau_x = np.nan
            self.final_tau_x = np.nan
            self.est_tau_y = np.nan
    
    
    def s_exp(self, x_array, amp, tau):
        decay = amp * np.exp(-x_array/tau)
        return decay
    
    
    def db_exp(self, x_array, a_fast, tau1, a_slow, tau2):
        y = ((a_fast * np.exp(-x_array/tau1))
            + (a_slow * np.exp(-x_array/tau2)))
        return y
    

    def fit_decay(self, fit_type):
        try:
            baselined_event = self.event_array - self.event_start_y
            amp = self.event_peak_x - self.array_start
            decay_y = baselined_event[amp:]
            decay_x = np.arange(len(decay_y))
            if fit_type == 'db_exp':
                upper_bounds = [0, np.inf, 0, np.inf]
                lower_bounds = [-np.inf, 0, -np.inf, 0]
                init_param = np.array([self.event_peak_y, self.final_tau_x, 0, 0])
                popt, pcov = curve_fit(self.db_exp, decay_x, decay_y, p0=init_param,
                                    bounds=[lower_bounds, upper_bounds])
                amp_1, self.fit_tau, amp_2, tau_2 = popt
                self.fit_decay_y = (self.db_exp(decay_x, amp_1, self.fit_tau,
                                                amp_2, tau_2)
                                                +self.event_start_y)
            else:
                upper_bounds = [0, np.inf]
                lower_bounds = [-np.inf, 0]
                init_param = np.array([self.event_peak_y, self.final_tau_x])
                popt, pcov = curve_fit(self.s_exp, decay_x, decay_y, p0=init_param,
                                    bounds=[lower_bounds, upper_bounds])
                amp_1, self.fit_tau = popt
                self.fit_decay_y = (self.s_exp(decay_x, amp_1, self.fit_tau)
                                    +self.event_start_y)
            self.fit_decay_x = (decay_x + self.event_peak_x)/self.s_r_c
        except:
            self.fit_decay_x = np.nan
            self.fit_decay_y = np.nan
            self.fit_tau = np.nan
    
    
    def find_event_parameters(self, y_array):
        if self.event_peak_x is np.nan:
            pass
        else:
            self.find_baseline()
            self.calc_event_amplitude(y_array)
            self.mini_plot_x = [self.event_start_x
                                / self.s_r_c,
                            self.event_peak_x
                            / self.s_r_c]
            self.mini_plot_y = [self.event_start_y, self.event_peak_y]
            self.est_decay()
            self.calc_event_rise_time()
            self.peak_align_value = self.event_peak_x - self.array_start
            if self.curve_fit_decay:
                self.fit_decay(fit_type=self.curve_fit_type)
    
    
    def mini_x_comp(self):
        x = [self.event_start_x/self.s_r_c,
                    self.event_peak_x/self.s_r_c,
                    self.est_tau_x/self.s_r_c]
        return x


    def mini_y_comp(self):
        y = [self.event_start_y, self.event_peak_y,
                    self.est_tau_y]
        return y

    def mini_x_array(self):
        return self.x_array/self.s_r_c


    def change_amplitude(self, x, y):
        self.event_peak_x = int(x)
        self.event_peak_y = y
        self.amplitude = abs(self.event_peak_y - self.event_start_y)
        self.calc_event_rise_time()
        self.est_decay()
        self.peak_align_value = self.event_peak_x - self.array_start
        if self.curve_fit_decay:
            self.fit_decay(fit_type=self.curve_fit_type)
        self.peak_align_value = self.event_peak_x - self.array_start
        self.mini_plot_x = [self.event_start_x
                                / self.s_r_c,
                            self.event_peak_x
                            / self.s_r_c]
        self.mini_plot_y = [self.event_start_y, self.event_peak_y]
       

    def change_baseline(self, x, y):
        self.event_start_x = int(x)
        self.event_start_y = y
        start = int((self.event_start_x -self.array_start)
                    - (0.5 * self.s_r_c))
        end = int(self.event_start_x - self.array_start)
        self.amplitude = abs(self.event_peak_y - self.event_start_y)
        self.calc_event_rise_time()
        self.est_decay()
        if self.curve_fit_decay:
           self.fit_decay(fit_type=self.curve_fit_type)
        self.peak_align_value = self.event_peak_x - self.array_start
        self.mini_plot_x = [self.event_start_x
                                / self.s_r_c,
                            self.event_peak_x
                            / self.s_r_c]
        self.mini_plot_y = [self.event_start_y, self.event_peak_y]
        

class LoadLFP(LFP):
    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class LoadoEPSC(oEPSC):
    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class LoadCurrentClamp(CurrentClamp):
    '''
    This class loads the saved CurrentClamp JSON file.
    '''
    
    def __init__(self, *args, **kwargs):
        self.sample_rate_correction = None
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.peaks = np.asarray(self.peaks, dtype=np.int64)
        self.array = np.asarray(self.array)
        if self.sample_rate_correction is not None:
            self.s_r_c = self.sample_rate_correction
            
    
class LoadMiniAnalysis(MiniAnalysis):
    '''
    This class loads the saved JSON file for an entire miniAnalysis session.
    '''
    
    def __init__(self, *args, **kwargs):
        self.sample_rate_correction = None
        
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        if self.sample_rate_correction is not None:
            self.s_r_c = self.sample_rate_correction
        
        self.final_array = np.array(self.final_array)
        self.create_postsynaptic_events()
        self.x_array = np.arange(
            len(self.final_array))/self.s_r_c
        self.event_arrays = [i.event_array-i.event_start_y
                             for i in self.postsynaptic_events]
        
    
    def create_postsynaptic_events(self):
        self.postsynaptic_events = []
        for i in self.saved_events_dict:
            h = LoadMini(i, final_array=self.final_array)
            self.postsynaptic_events += [h]


class LoadMini(PostSynapticEvent):
    '''
    This class create a new mini event from dictionary within a
    LoadMiniAnalysis JSON file.
    '''
    
    def __init__(self, *args, final_array, **kwargs):
        self.sample_rate_correction = None
        
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        if self.sample_rate_correction is not None:
            self.s_r_c = self.sample_rate_correction
            
        self.x_array = np.arange(self.array_start, self.array_end)
        self.event_array = final_array[self.array_start:self.array_end]
  


if __name__ == '__main__':
    Acquisition(),
    LFP(),
    CurrentClamp(),
    MiniAnalysis(),
    MiniAnalysis(),
    PostSynapticEvent(),
    LoadMini(),
    LoadMiniAnalysis(),
    LoadCurrentClamp(),
    oEPSC(),
    LoadoEPSC(),
    LoadLFP()
    
