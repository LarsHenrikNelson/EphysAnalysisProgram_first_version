# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:03:02 2022

@author: LarsNelson
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm


class FinalMiniAnalysis:
    '''
    This class is used to compile all the data from a dictionary of
    acquistions that contain mini data. The class contains the raw data, and
    the averaged data. The number of events and acquisitions that were deleted
    also needed as input to the class however, the initial value is set to 0.
    '''
    
    def __init__(self, acq_dict, events_deleted=0, acqs_deleted=0,
                 sample_rate=10000, curve_fit_decay=False,
                 curve_fit_type='db_exp'):
        self.acq_dict = acq_dict
        self.events_deleted = events_deleted
        self.acqs_deleted = acqs_deleted
        self.sample_rate = sample_rate
        self.compute_data()
        
        
    def compute_minis(self):
        '''
        This function loops through and finalizes the mini data for each
        acquisition using the MiniAnalysis.final_acq_data() function.

        Returns
        -------
        None.

        '''
        for item in self.acq_dict.values():
            item.final_acq_data()
        
     
    def extract_raw_data(self):
        '''
        This function compiles the data from each acquisitions and puts it
        into a pandas dataframe. The data that are included are: amplitudes,
        taus, events times, time stamp of acquisition, rise times, rise rates,
        ieis, each event time adjust for the time stamp of the respective
        acquisition and the array for the average mini. One thing to note is
        that ieis have an added nan at the end of the data for each
        acquisition so that the ieis are aligned with the other data from
        the acquisition.

        Returns
        -------
        None.

        '''
        
        acq_list = pd.Series(
            sum([item.acq_number_list for item in self.acq_dict.values()], []),
            name='Acquisition')
        amplitude_list = pd.Series(
            sum([item.amplitudes for item in self.acq_dict.values()], []),
            name='Amplitude (pA)')
        taus_list = pd.Series(
            sum([item.taus for item in self.acq_dict.values()], []),
            name='Est tau (ms)')
        event_times_list = pd.Series(
            sum([item.event_times for item in self.acq_dict.values()], []),
            name='Event time (ms)')
        time_stamp_events = pd.Series(
            sum([item.time_stamp_events for item in self.acq_dict.values()], []),
            name='Acq time stamp')
        rise_times_list = pd.Series(
            sum([item.rise_times for item in self.acq_dict.values()], []),
            name='Rise time (ms)')
        rise_rates_list = pd.Series(
            sum([item.rise_rates for item in self.acq_dict.values()], []),
            name='Rise rate (pA/ms)')
        iei_list = pd.Series(
            np.concatenate([item.iei for item in self.acq_dict.values()]),
            name='IEI (ms)')
        
        self.raw_df = pd.concat([acq_list, amplitude_list,
                        taus_list, event_times_list, time_stamp_events,
                        rise_times_list, rise_rates_list, iei_list], axis=1)
        self.raw_df['Acq time stamp'] = (
            self.raw_df['Acq time stamp']
            -self.raw_df['Acq time stamp'].unique()[0])*1000
        self.raw_df['Real time'] = (self.raw_df['Acq time stamp']
                                    + self.raw_df['Event time (ms)'])
        self.raw_df['Ave event'] = pd.Series(self.average_mini)
    
    
    def extract_final_data(self):
        columns_for_analysis = ['IEI (ms)',
                'Amplitude (pA)', 'Est tau (ms)', 'Rise rate (pA/ms)',
                'Rise time (ms)']
        means = self.raw_df[columns_for_analysis].mean().to_frame().T
        std =  self.raw_df[columns_for_analysis].std().to_frame().T
        sem = self.raw_df[columns_for_analysis].sem().to_frame().T
        median = self.raw_df[columns_for_analysis].median().to_frame().T
        skew = self.raw_df[columns_for_analysis].skew().to_frame().T
        cv = std/means
        self.final_df = pd.concat([means, std, sem, median, skew, cv])
        self.final_df.insert(0, 'Statistic', ['mean', 'std', 'sem', 'median',
                                              'skew', 'cv'])
       
        
        events = len(sum([item.amplitudes for item in self.acq_dict.values()],
                         []))
        self.final_df['Ave event tau'] = [self.fit_tau_x, np.nan, np.nan,
                                          np.nan, np.nan, np.nan]
        self.final_df['Events'] = [events, np.nan, np.nan, np.nan, np.nan,
                                   np.nan]
        self.final_df['Events deleted'] = [self.events_deleted, np.nan,
                                 np.nan, np.nan, np.nan, np.nan]
        self.final_df['Acqs'] = [len(self.acq_dict.keys()), np.nan, np.nan,
                                 np.nan, np.nan, np.nan]
        self.final_df['Acqs deleted'] = [self.acqs_deleted, np.nan, np.nan,
                                 np.nan, np.nan, np.nan]
        
        self.final_df.reset_index(inplace=True)
        self.final_df.drop(['index'], axis=1, inplace=True)
        
        
    def create_average_mini(self):
        peak_align_values = sum([item.peak_align_values
                       for item in self.acq_dict.values()], [])
        events_list = sum([item.event_arrays
                       for item in self.acq_dict.values()], [])
        max_min = max(peak_align_values)
        start_values = [max_min-i for i in peak_align_values]
        arrays = [np.append(i*[j[0]], j) for i, j in
                  zip(start_values, events_list)]
        max_length = max(map(len, arrays))
        end_values = [max_length-len(i) for i in arrays]
        final_arrays = [np.append(j, i*[j[-1]]) for i, j in zip(end_values, arrays)]
        self.average_mini = np.average(np.array(final_arrays), axis=0)
        self.average_mini_x = (np.arange(self.average_mini.shape[0])
                               /(self.sample_rate/1000))

        # peak_align_values = sum([item.peak_align_values
        #                for item in self.acq_dict.values()], [])
        # events_list = sum([item.event_arrays
        #                for item in self.acq_dict.values()], [])
        # max_min = max(peak_align_values)
        # start_values = [max_min-i for i in peak_align_values]
        # arrays = [np.append(i*[j[0]], j) for i, j in
        #           zip(start_values, events_list)]
        # min_length = min(map(len, arrays))
        # arrays = [i [:min_length] for i in arrays]
        # self.average_mini = np.average(np.array(arrays), axis=0)
        # self.average_mini_x = (np.arange(self.average_mini.shape[0])
        #                        /(self.sample_rate/1000))
        
        
    def analyze_average_mini(self):
        self.average_mini = self.average_mini-np.mean(self.average_mini[0:10])
        event_peak_x = np.argmin(self.average_mini)
        event_peak_y = np.min(self.average_mini)
        est_tau_y = event_peak_y* (1 / np.exp(1))
        self.decay_y = self.average_mini[event_peak_x:]
        self.decay_x = np.arange(len(self.decay_y))/10
        self.est_tau_x = np.interp(est_tau_y, self.decay_y, self.decay_x)
        init_param = np.array([event_peak_y, self.est_tau_x])
        upper_bound = (event_peak_y + 5, self.est_tau_x+10)
        lower_bound = (event_peak_y - 5, self.est_tau_x-10)
        bounds = [lower_bound, upper_bound]
        popt, pcov = curve_fit(self.decay_exp, self.decay_x, self.decay_y, 
                               p0=init_param, bounds=bounds)
        fit_amp, self.fit_tau_x = popt
        self.fit_decay_y = self.decay_exp(self.decay_x, fit_amp,
                                          self.fit_tau_x)
        self.decay_x = self.decay_x + event_peak_x/10
        
    
    def decay_exp(self, x, amp, tau):
        decay = amp * np.exp(-x/tau)
        return decay

    
    # def alt_decay_1(self, t, amp, tau):
    #     decay = amp * (1 - np.exp(-t/tau))
    #     return decay
    
    
    # def alt_decay_2(self, x, amp, tau, t0):
    #     decay = amp * (np.exp(-(t-t0)/tau))
    #     return decay
    
    
    # def double_decay(self, b0, b1, tau1, b2, tau2, x):
    #     decay = b0 + b1 * np.exp(-x/tau1) + b * np.exp(-x/tau2)
    #     return decay
    

    def compute_data(self):
        self.compute_minis()
        self.create_average_mini()
        self.analyze_average_mini()
        self.extract_raw_data()
        self.extract_final_data()
        

    def stem_components(self, column):
        array_x = self.final_obj.raw_df['Real time'].to_numpy()
        array_y = self.final_obj.raw_df[column].to_numpy()
        stems_y = np.stack([array_y,array_y*0],axis=-1).flatten()
        stems_x = np.stack([array_x,array_x],axis=-1).flatten()
        return array_x, array_y, stems_x, stems_y


    def save_data(self, save_filename):
        '''
        This function saves the resulting pandas data frames to an excel file.
        The function saves the data to the current directory so all that is
        needed is a name for the excel file.
        '''
        temp_list = [pd.Series(self.average_mini, name="ave_mini"),
                     pd.Series(self.average_mini_x, name='ave_mini_x'),
                     pd.Series(self.fit_decay_y, name='fit_decay_y'),
                     pd.Series(self.decay_x, name='decay_x')]
        extra_data = pd.concat(temp_list, axis=1)
        with pd.ExcelWriter(f"{save_filename}.xlsx",
                    mode='w', engine='openpyxl') as writer:
                    self.raw_df.to_excel(writer, index=False,
                                         sheet_name='Raw data')
                    self.final_df.to_excel(writer, index=False,
                                           sheet_name='Final data')
                    extra_data.to_excel(writer, index=False,
                                       sheet_name='Extra data')


class FinalCurrentClampAnalysis:
    def __init__(self, acq_dict, iv_start=1, iv_end=6):
        self.acq_dict = acq_dict
        self.iv_start = iv_start
        self.iv_end = iv_end
        self.create_raw_data()
        self.final_analysis()
    
        
    def create_raw_data(self):
        self.raw_df = pd.DataFrame(
            [self.acq_dict[i].create_dict() for i in
             self.acq_dict.keys()])
        self.raw_df['Epoch'] = pd.to_numeric(self.raw_df['Epoch'])
        self.raw_df['Pulse_pattern'] = pd.to_numeric(
            self.raw_df['Pulse_pattern'])
        self.raw_df['Pulse_amp'] = pd.to_numeric(
            self.raw_df['Pulse_amp'])
       
        
    def final_analysis(self):
        #I need to separate the delta v and spike ieis from the rest of the
        #dataframe to clean it up.
        self.df_averaged_data = self.raw_df.groupby(['Pulse_pattern', 
                                       'Epoch', 'Pulse_amp', 
                                       'Ramp']).mean()
        self.df_averaged_data.reset_index(inplace=True)
        self.df_averaged_data.drop(['Pulse_pattern'], axis=1, inplace=True)
        self.df_averaged_data[['Pulse_amp', 'Ramp',
                          'Epoch']] = self.df_averaged_data[['Pulse_amp', 'Ramp',
                                                        'Epoch']].astype(int)
         
        #Pivot the dataframe to get it into wideform format
        pivoted_df = self.df_averaged_data.pivot_table(index=['Epoch', 'Ramp'], 
                                                        columns = ['Pulse_amp'],
                                                        aggfunc=np.nanmean)
        
        pivoted_df.reset_index(inplace=True)
        
        #Add the input resistance calculate
        resistance_df = self.membrane_resistance(pivoted_df)
        self.final_df = pd.concat([pivoted_df, resistance_df], axis=1)
        
        #Clean up the final dataframe
        del self.final_df['Acquisition']
        
        self.final_df['Baseline_ave','Average'] = self.final_df[
            'Baseline'].mean(axis=1)
        del self.final_df['Baseline']
        
        if 'Spike_threshold' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike_threshold_ap', 'first')] = self.final_df[
                'Spike_threshold'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Spike_threshold']
        
        if 'Hertz' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Pulse', 'Rheo')] = (
                self.final_df['Hertz'] > 0).idxmax(axis=1, skipna=True)
        
        if 'Spike_width' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'width')] = self.final_df[
                'Spike_width'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Spike_width']
        
        if 'Spike_freq_adapt' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Spike_freq_adapt')] = self.final_df[
                'Spike_freq_adapt'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Spike_freq_adapt']
            
        if 'Local_sfa' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Local_sfa')] = self.final_df[
                'Local_sfa'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Local_sfa']
            
        if 'Divisor_sfa' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Divisor_sfa')] = self.final_df[
                'Divisor_sfa'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Divisor_sfa']
        
        if 'Max_AP_vel' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Max_AP_vel')] = self.final_df[
                'Max_AP_vel'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Max_AP_vel']
            
        if 'Peak_AHP' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Peak_AHP')] = self.final_df[
                'Peak_AHP'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Peak_AHP']
            
        if 'Spike_peak_volt' in self.final_df.columns.levels[0].tolist():
            self.final_df[('Spike', 'Spike_peak_volt')] = self.final_df[
                'Spike_peak_volt'].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
            del self.final_df['Spike_peak_volt']
        
        #Creating the final first action potentials and final data. It takes
        #manipulating to get the data to show up correctly in the TableWidget.
        pulse_dict, ramp_dict = self.create_first_aps()
        
        if pulse_dict:
            pulse_ap = self.first_ap_dict(pulse_dict)
        else:
            pulse_ap = {}
        
        if ramp_dict:
             ramp_ap = self.first_ap_dict(ramp_dict)
        else:
            ramp_ap = {}
        
        if pulse_ap:
            self.pulse_df = pd.DataFrame(dict([(
                k, pd.Series(v)) for k,v in pulse_ap.items()]))
        else:
            self.pulse_df = pd.DataFrame()
        
        if ramp_ap:
            self.ramp_df = pd.DataFrame(dict([(
                k, pd.Series(v)) for k,v in ramp_ap.items()]))
        else:
            self.ramp_df = pd.DataFrame()
            
        self.iv_curve_dataframe()
        self.deltav_dataframe()
            
    
    def create_first_aps(self):
        pulse_dict = defaultdict(lambda: defaultdict(list))
        ramp_dict = defaultdict(lambda: defaultdict(list))
        for i in self.acq_dict.keys():
            if len(self.acq_dict[i].first_ap) <= 1:
                pass
            else:
                if self.acq_dict[i].ramp == '0':
                    pulse_dict[
                        self.acq_dict[i].epoch][
                            self.acq_dict[i].pulse_amp].append(
                                self.acq_dict[i].first_ap)
                if self.acq_dict[i].ramp == '1':
                    ramp_dict[self.acq_dict[i].epoch][
                        self.acq_dict[i].pulse_amp].append(
                            self.acq_dict[i].first_ap)
        return pulse_dict, ramp_dict
    
    
    def first_ap_dict(self, dictionary):
        ap_dict = {}
        if len(dictionary.keys()) > 1:
            for i in dictionary.keys():
                average = self.average_aps(dictionary[i])
                ap_dict[i] = average
        else:
            i = list(dictionary.keys())[0]
            average = self.average_aps(dictionary[i])
            ap_dict[i] = average
        return ap_dict
    
    
    def average_aps(self, dict_entry):
        '''
        This function takes a list of a lists/arrays, finds the max values
        and then aligns all the lists/arrays to the max value by adding an 
        array of values to the beginning of each list/array (the value is the 
        first value in each list/array)

        Parameters
        ----------
        dict_entry : Dictionary entry that contains a several lists/arrays.

        Returns
        -------
        average : The averaged list/array of several lists/arrays based on on
        the index of the maximum value.

        '''
        first_pulse = min(map(int, list(dict_entry)))
        ap_max_values = [np.argmax(i) for i in dict_entry[str(first_pulse)]]
        max_ap = max(ap_max_values)
        start_values = [max_ap-i for i in ap_max_values]
        arrays = [np.append(i*[j[0]], j) for i, j in
                  zip(start_values, dict_entry[str(first_pulse)])]
        length = min(map(len, arrays))
        arrays = [i [:length] for i in arrays]
        average = np.average(np.array(arrays), axis=0)
        return average    


    def membrane_resistance(self, df):
        df1 = df[df[('Ramp', '')]  == 0]
        if df1.empty == True:
            return df
        else:
            df2 = df1['Delta_v'].copy()
            df2.dropna(axis=0, how='all', inplace=True)
            index_1 = df2.index.values
            self.plot_epochs = df1['Epoch'].to_list()
            self.iv_y = df2.to_numpy()
            self.deltav_x = np.array(df2.T.index.map(int))
            self.iv_plot_x = self.deltav_x[self.iv_start - 1:self.iv_end]
            x_constant = sm.add_constant(self.iv_plot_x)
            slope = []
            self.iv_line = []
            if len(self.iv_y) == 1:
                y = self.iv_y[0][self.iv_start - 1:self.iv_end]
                model_1 = sm.OLS(y, x_constant)
                results_1 = model_1.fit()
                slope_1 = results_1.params
                slope += [slope_1[1]*1000]
                self.iv_line += [slope_1[1]*self.iv_plot_x + slope_1[0]]
            else:
                for i in self.iv_y:
                    y = i[self.iv_start - 1:self.iv_end]
                    model_1 = sm.OLS(y, x_constant)
                    results_1 = model_1.fit()
                    slope_1 = results_1.params
                    slope += [slope_1[1]*1000]
                    self.iv_line += [slope_1[1]*self.iv_plot_x + slope_1[0]]
            resistance = pd.DataFrame(data = slope, index= index_1, 
                                      columns=['I/V Curve'])
            resistance.columns = pd.MultiIndex.from_product(
                [resistance.columns, ['Resistance']])
            return resistance


    def iv_curve_dataframe(self):
        self.iv_df = pd.DataFrame(self.iv_line)
        self.iv_df = self.iv_df.transpose()
        self.iv_df.columns = self.plot_epochs
        self.iv_df['iv_plot_x'] = self.iv_plot_x
    
    def deltav_dataframe(self):
        self.deltav_df = pd.DataFrame(self.iv_y)
        self.deltav_df = self.deltav_df.transpose()
        self.deltav_df.columns = self.plot_epochs
        self.deltav_df['deltav_x'] = self.deltav_x


    def temp_df(self):
        temp_df = self.final_df.copy()
        mi = temp_df.columns
        ind = pd.Index([e[0] + '_' + str(e[1]) for e in mi.tolist()])
        temp_df.columns = ind
        return temp_df


    def save_data(self, save_filename):
        '''
        This function saves the resulting pandas data frames to an excel file.
        The function saves the data to the current directory so all that is
        needed is a name for the excel file.
        '''
        with pd.ExcelWriter(f"{save_filename}.xlsx",
                    mode='w', engine='xlsxwriter') as writer:
                    self.raw_df.to_excel(writer, index=False,
                                         sheet_name='Raw data')
                    self.final_df.to_excel(writer, sheet_name='Final data')
                    self.iv_df.to_excel(writer, sheet_name='IV_df')
                    self.deltav_df.to_excel(writer, sheet_name='Deltav_df')
                    if not self.pulse_df.empty:
                        self.pulse_df.to_excel(writer, index=False,
                                               sheet_name='Pulse APs')
                    if not self.ramp_df.empty:
                        self.ramp_df.to_excel(writer, index=False,
                                               sheet_name='Ramp APs')
        return None


class FinalEvokedCurrent:
    def __init__(self, o_acq_dict=None, lfp_acq_dict=None):
        self.o_acq_dict = o_acq_dict
        self.lfp_acq_dict = lfp_acq_dict
        self.raw_data()
        self.final_data()
        
    def raw_data(self):
        if self.o_acq_dict is not None:
            o_raw_df = pd.DataFrame(
                [self.o_acq_dict[i].create_dict() for i in
                 self.o_acq_dict.keys()])
        if self.lfp_acq_dict is not None:
            lfp_raw_df = pd.DataFrame(
                [self.lfp_acq_dict[i].create_dict() for i in
                 self.lfp_acq_dict.keys()])
        if self.lfp_acq_dict is not None and self.o_acq_dict is not None:
            self.raw_df = pd.merge(lfp_raw_df, o_raw_df,
                on=['Acq number', 'Epoch'], suffixes=['', ''])
        elif self.o_acq_dict is None and self.lfp_acq_dict is not None:
            self.raw_df = lfp_raw_df
        else:
            self.raw_df = o_raw_df
        self.raw_df['Epoch'] = pd.to_numeric(self.raw_df['Epoch'])
    
    
    def final_data(self):
        if self.lfp_acq_dict is not None and self.o_acq_dict is not None:
            self.final_df = self.raw_df.groupby(['Epoch',
                'Peak direction']).mean()
            self.final_df.reset_index(inplace=True)
        elif self.o_acq_dict is not None and self.lfp_acq_dict is None:
            self.final_df = self.raw_df.groupby(['Epoch',
                                                 'Peak direction']).mean()
            self.final_df.reset_index(inplace=True)
        else:
            self.final_df = self.raw_df.groupby(['Epoch']).mean()
            self.final_df.reset_index(inplace=True)
            
            
    def save_data(self, save_filename):
        '''
        This function saves the resulting pandas data frames to an excel file.
        The function saves the data to the current directory so all that is
        needed is a name for the excel file.
        '''
        with pd.ExcelWriter(f"{save_filename}.xlsx",
                    mode='w', engine='openpyxl') as writer:
            s