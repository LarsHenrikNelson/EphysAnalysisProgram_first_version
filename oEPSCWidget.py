# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:51:49 2021

@author: LarsNelson
"""

from math import log10, floor, isnan, nan
from glob import glob
import json

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
                             QWidget, QLabel, QFormLayout, QComboBox, 
                             QSpinBox, QCheckBox,  QProgressBar, QMessageBox,
                             QTabWidget)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThreadPool
from PyQt5 import QtCore
import pyqtgraph as pg

from acq_class import LFP, oEPSC
from final_analysis_classes import FinalEvokedCurrent
from utility_classes import (LineEdit, SaveWorker, YamlWorker)


class oEPSCWidget(QWidget):
    def __init__(self, parent=None):
        
        super(oEPSCWidget, self).__init__(parent)
        
        self.parent = parent
        
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab1, 'Setup')
        self.tabs.addTab(self.tab2, 'Analysis')     
        
        self.setStyleSheet('''QTabWidget::tab-bar 
                                          {alignment: left;}''')
           
        self.main_layout.addWidget(self.tabs)
                 
        #Tab 1 layout
        self.tab1_layout = QVBoxLayout()
        self.form_layouts = QHBoxLayout()
        self.input_layout_1 = QFormLayout()
        self.input_layout_2 = QFormLayout()
        self.form_layouts.addLayout(self.input_layout_1)
        self.form_layouts.addLayout(self.input_layout_2)
        self.tab1.setLayout(self.tab1_layout)
        self.tab1_layout.addLayout(self.form_layouts, 0)
        
        #Tab 2 layout
        self.tab2_layout = QVBoxLayout()
        self.analysis_buttons_layout = QFormLayout()
        self.tab2_layout.addLayout(self.analysis_buttons_layout)
        self.final_data_tabs = QTabWidget()
        self.raw_datatable = pg.TableWidget(sortable=False)
        self.final_data_tabs.addTab(self.raw_datatable,
                                    'Raw data')
        self.final_datatable = pg.TableWidget(sortable=False)
        self.final_data_tabs.addTab(self.final_datatable,
                                    'Final data')
        self.tab2_layout.addWidget(self.final_data_tabs)
        self.tab2.setLayout(self.tab2_layout)
        
        #Plots
        self.oepsc_plot = pg.PlotWidget()
        self.lfp_plot = pg.PlotWidget()
        self.oepsc_plot_layout = QHBoxLayout()
        self.lfp_plot_layout = QHBoxLayout()
        self.o_info_layout = QFormLayout()
        self.lfp_info_layout = QFormLayout()
        self.plot_layout = QVBoxLayout()
        self.oepsc_plot_layout.addLayout(self.o_info_layout, 0)
        self.oepsc_plot_layout.addWidget(self.oepsc_plot, 1)
        self.lfp_plot_layout.addLayout(self.lfp_info_layout, 0)
        self.lfp_plot_layout.addWidget(self.lfp_plot,1)
        self.plot_layout.addLayout(self.oepsc_plot_layout,1)
        self.plot_layout.addLayout(self.lfp_plot_layout,1)
        self.main_layout.addLayout(self.plot_layout, 1)
        
        #oEPSC buttons and line edits
        self.oepsc_input = QLabel('oEPSC')
        self.input_layout_1.addRow(self.oepsc_input)
        
        self.o_analyze_label = QLabel('Analyze oEPSC')
        self.o_analyze = QCheckBox(self)
        self.o_analyze.setChecked(True)
        self.o_analyze.setTristate(False)
        
        self.input_layout_1.addRow(self.o_analyze_label,
                                 self.o_analyze)
        
        self.o_acq_id_label = QLabel('Acq ID')
        self.o_acq_id_edit = LineEdit()
        self.o_acq_id_edit.setEnabled(True)
        self.o_acq_id_edit.setObjectName('o_acq_id_edit')
        self.input_layout_1.addRow(self.o_acq_id_label, self.o_acq_id_edit)
    
        self.o_start_acq_label = QLabel('Start acq')
        self.o_start_acq_edit = LineEdit()
        self.o_start_acq_edit.setEnabled(True)
        self.o_start_acq_edit.setObjectName('o_start_acq_edit')
        self.input_layout_1.addRow(self.o_start_acq_label,
                                   self.o_start_acq_edit)
        
        self.o_end_acq_label = QLabel('End acq')
        self.o_end_acq_edit = LineEdit()
        self.o_end_acq_edit.setEnabled(True)
        self.o_end_acq_edit.setObjectName('o_end_acq_edit')
        self.input_layout_1.addRow(self.o_end_acq_label, self.o_end_acq_edit)
    
        self.o_b_start_label = QLabel('Baseline start')
        self.o_b_start_edit = LineEdit()
        self.o_b_start_edit.setEnabled(True)
        self.o_b_start_edit.setObjectName('o_b_start_edit')
        self.o_b_start_edit.setText('850')
        self.input_layout_1.addRow(self.o_b_start_label, self.o_b_start_edit)
        
        self.o_b_end_label = QLabel('Baseline end')
        self.o_b_end_edit = LineEdit()
        self.o_b_end_edit.setEnabled(True)
        self.o_b_end_edit.setObjectName('o_b_end_edit')
        self.o_b_end_edit.setText('950')
        self.input_layout_1.addRow(self.o_b_end_label, self.o_b_end_edit)
        
        self.o_sample_rate_label = QLabel('Sample rate')
        self.o_sample_rate_edit = LineEdit()
        self.o_sample_rate_edit.setEnabled(True)
        self.o_sample_rate_edit.setObjectName('o_sample_rate_edit')
        self.o_sample_rate_edit.setText('10000')
        self.input_layout_1.addRow(self.o_sample_rate_label,
                                 self.o_sample_rate_edit)
    
        self.o_filter_type_label = QLabel('Filter Type')
        filters = ['remez_2', 'fir_zero_2', 'bessel', 'fir_zero_1', 'savgol', 
                   'median', 'remez_1', 'None']
        self.o_filter_selection= QComboBox(self)
        self.o_filter_selection.addItems(filters)
        self.o_filter_selection.setObjectName('o_filter_selection')
        self.o_filter_selection.setCurrentText('savgol')
        self.input_layout_1.addRow(self.o_filter_type_label,
                                   self.o_filter_selection)
        
        self.o_order_label = QLabel('Order')
        self.o_order_edit = LineEdit()
        self.o_order_edit.setValidator(QIntValidator())
        self.o_order_edit.setEnabled(True)
        self.o_order_edit.setObjectName('o_order_edit')
        self.o_order_edit.setText('5')
        self.input_layout_1.addRow(self.o_order_label, self.o_order_edit)
        
        self.o_high_pass_label = QLabel('High-pass')
        self.o_high_pass_edit = LineEdit()
        self.o_high_pass_edit.setValidator(QIntValidator())
        self.o_high_pass_edit.setObjectName('o_high_pass_edit')
        self.o_high_pass_edit.setEnabled(True)
        self.input_layout_1.addRow(self.o_high_pass_label,
                                   self.o_high_pass_edit)
        
        self.o_high_width_label = QLabel('High-width')
        self.o_high_width_edit = LineEdit()
        self.o_high_width_edit.setValidator(QIntValidator())
        self.o_high_width_edit.setObjectName('o_high_width_edit')
        self.o_high_width_edit.setEnabled(True)
        self.input_layout_1.addRow(self.o_high_width_label,
                                   self.o_high_width_edit)
        
        self.o_low_pass_label = QLabel('Low-pass')
        self.o_low_pass_edit = LineEdit()
        self.o_low_pass_edit.setValidator(QIntValidator())
        self.o_low_pass_edit.setObjectName('o_low_pass_edit')
        self.o_low_pass_edit.setEnabled(True)
        self.input_layout_1.addRow(self.o_low_pass_label,
                                   self.o_low_pass_edit)
        
        self.o_low_width_label = QLabel('Low-width')
        self.o_low_width_edit = LineEdit()
        self.o_low_width_edit.setValidator(QIntValidator())
        self.o_low_width_edit.setObjectName('o_low_width_edit')
        self.o_low_width_edit.setEnabled(True)
        self.input_layout_1.addRow(self.o_low_width_label,
                                   self.o_low_width_edit)
        
        self.o_window_label = QLabel('Window type')
        windows = ['hann', 'hamming', 'blackmmaharris', 'barthann', 'nuttall',
            'blackman']
        self.o_window_edit = QComboBox(self)
        self.o_window_edit.setObjectName('o_window_edit')
        self.o_window_edit.addItems(windows)
        self.input_layout_1.addRow(self.o_window_label, self.o_window_edit)
        
        self.o_polyorder_label = QLabel('Polyorder')
        self.o_polyorder_edit = LineEdit()
        self.o_polyorder_edit.setValidator(QIntValidator())
        self.o_polyorder_edit.setEnabled(True)
        self.o_polyorder_edit.setObjectName('o_polyorder_edit')
        self.o_polyorder_edit.setText('3')
        self.input_layout_1.addRow(self.o_polyorder_label,
                                   self.o_polyorder_edit)
        
        self.o_pulse_start = QLabel('Pulse start')
        self.o_pulse_start_edit = LineEdit()
        self.o_pulse_start_edit.setEnabled(True)
        self.o_pulse_start_edit.setObjectName('o_pulse_start_edit')
        self.o_pulse_start_edit.setText('1000')
        self.input_layout_1.addRow(self.o_pulse_start,
                                   self.o_pulse_start_edit)
        
        self.o_neg_window_start = QLabel('Negative window start')
        self.o_neg_start_edit = LineEdit()
        self.o_neg_start_edit.setValidator(QIntValidator())
        self.o_neg_start_edit.setEnabled(True)
        self.o_neg_start_edit.setObjectName('o_pulse_start_edit')
        self.o_neg_start_edit.setText('1001')
        self.input_layout_1.addRow(self.o_neg_window_start,
                                   self.o_neg_start_edit)
        
        self.o_neg_window_end = QLabel('Negative window end')
        self.o_neg_end_edit = LineEdit()
        self.o_neg_end_edit.setValidator(QIntValidator())
        self.o_neg_end_edit.setObjectName('o_neg_end_edit')
        self.o_neg_end_edit.setEnabled(True)
        self.o_neg_end_edit.setText('1050')
        self.input_layout_1.addRow(self.o_neg_window_end,
                                   self.o_neg_end_edit)
        
        self.o_pos_window_start = QLabel('Positive window start')
        self.o_pos_start_edit = LineEdit()
        self.o_pos_start_edit.setValidator(QIntValidator())
        self.o_pos_start_edit.setEnabled(True)
        self.o_pos_start_edit.setObjectName('o_pos_start_edit')
        self.o_pos_start_edit.setText('1045')
        self.input_layout_1.addRow(self.o_pos_window_start,
                                   self.o_pos_start_edit)
        
        self.o_pos_window_end = QLabel('Positive window end')
        self.o_pos_end_edit = LineEdit()
        self.o_pos_end_edit.setValidator(QIntValidator())
        self.o_pos_end_edit.setEnabled(True)
        self.o_pos_end_edit.setObjectName('o_pos_end_edit')
        self.o_pos_end_edit.setText('1055')
        self.input_layout_1.addRow(self.o_pos_window_end,
                                   self.o_pos_end_edit)
        
        #LFP input
        self.lfp_input = QLabel('LFP')
        self.input_layout_2.addRow(self.lfp_input)
        
        self.lfp_analyze_label = QLabel('Analyze LFP')
        self.lfp_analyze = QCheckBox(self)
        self.lfp_analyze.setChecked(True)
        self.lfp_analyze.setTristate(False)
        self.lfp_analyze.setObjectName('lfp_analyze')
        self.input_layout_2.addRow(self.lfp_analyze_label,
                                 self.lfp_analyze)
        
        self.lfp_acq_id_label = QLabel('Acq ID')
        self.lfp_acq_id_edit = LineEdit()
        self.lfp_acq_id_edit.setEnabled(True)
        self.lfp_acq_id_edit.setObjectName('lfp_acq_id_edit')
        self.input_layout_2.addRow(self.lfp_acq_id_label,
                                   self.lfp_acq_id_edit)
    
        self.lfp_start_acq_label = QLabel('Start acq')
        self.lfp_start_acq_edit = LineEdit()
        self.lfp_start_acq_edit.setEnabled(True)
        self.lfp_start_acq_edit.setObjectName('lfp_start_acq_edit')
        self.input_layout_2.addRow(self.lfp_start_acq_label,
                                   self.lfp_start_acq_edit)
        
        self.lfp_end_acq_label = QLabel('End acq')
        self.lfp_end_acq_edit = LineEdit()
        self.lfp_end_acq_edit.setEnabled(True)
        self.lfp_end_acq_edit.setObjectName('lfp_end_acq_edit')
        self.input_layout_2.addRow(self.lfp_end_acq_label,
                                   self.lfp_end_acq_edit)
    
        self.lfp_b_start_label = QLabel('Baseline start')
        self.lfp_b_start_edit = LineEdit()
        self.lfp_b_start_edit.setEnabled(True)
        self.lfp_b_start_edit.setObjectName('lfp_b_start_edit')
        self.lfp_b_start_edit.setText('850')
        self.input_layout_2.addRow(self.lfp_b_start_label,
                                   self.lfp_b_start_edit)
        
        self.lfp_b_end_label = QLabel('Baseline end')
        self.lfp_b_end_edit = LineEdit()
        self.lfp_b_end_edit.setEnabled(True)
        self.lfp_b_end_edit.setObjectName('lfp_b_end_edit')
        self.lfp_b_end_edit.setText('950')
        self.input_layout_2.addRow(self.lfp_b_end_label,
                                   self.lfp_b_end_edit)
        
        self.lfp_sample_rate_label = QLabel('Sample rate')
        self.lfp_sample_rate_edit = LineEdit()
        self.lfp_sample_rate_edit.setEnabled(True)
        self.lfp_sample_rate_edit.setObjectName('lfp_sample_rate_edit')
        self.lfp_sample_rate_edit.setText('10000')
        self.input_layout_2.addRow(self.lfp_sample_rate_label,
                                 self.lfp_sample_rate_edit)
    
        self.lfp_filter_type_label = QLabel('Filter Type')
        filters = ['remez_2', 'fir_zero_2', 'bessel', 'fir_zero_1', 'savgol', 
                   'median', 'remez_1', 'None']
        self.lfp_filter_selection= QComboBox(self)
        self.lfp_filter_selection.addItems(filters)
        self.lfp_filter_selection.setObjectName('lfp_filter_selection')
        self.lfp_filter_selection.setCurrentText('savgol')
        self.input_layout_2.addRow(self.lfp_filter_type_label,
                                   self.lfp_filter_selection)
        
        self.lfp_order_label = QLabel('Order')
        self.lfp_order_edit = LineEdit()
        self.lfp_order_edit.setValidator(QIntValidator())
        self.lfp_order_edit.setEnabled(True)
        self.lfp_order_edit.setObjectName('lfp_order_edit')
        self.lfp_order_edit.setText('5')
        self.input_layout_2.addRow(self.lfp_order_label, self.lfp_order_edit)
        
        self.lfp_high_pass_label = QLabel('High-pass')
        self.lfp_high_pass_edit = LineEdit()
        self.lfp_high_pass_edit.setValidator(QIntValidator())
        self.lfp_high_pass_edit.setObjectName('lfp_high_pass_edit')
        self.lfp_high_pass_edit.setEnabled(True)
        self.input_layout_2.addRow(self.lfp_high_pass_label,
                                   self.lfp_high_pass_edit)
        
        self.lfp_high_width_label = QLabel('High-width')
        self.lfp_high_width_edit = LineEdit()
        self.lfp_high_width_edit.setValidator(QIntValidator())
        self.lfp_high_width_edit.setObjectName('lfp_high_width_edit')
        self.lfp_high_width_edit.setEnabled(True)
        self.input_layout_2.addRow(self.lfp_high_width_label,
                                   self.lfp_high_width_edit)
        
        self.lfp_low_pass_label = QLabel('Low-pass')
        self.lfp_low_pass_edit = LineEdit()
        self.lfp_low_pass_edit.setValidator(QIntValidator())
        self.lfp_low_pass_edit.setObjectName('lfp_low_pass_edit')
        self.lfp_low_pass_edit.setEnabled(True)
        self.input_layout_2.addRow(self.lfp_low_pass_label,
                                   self.lfp_low_pass_edit)
        
        self.lfp_low_width_label = QLabel('Low-width')
        self.lfp_low_width_edit = LineEdit()
        self.lfp_low_width_edit.setValidator(QIntValidator())
        self.lfp_low_width_edit.setObjectName('lfp_low_width_edit')
        self.lfp_low_width_edit.setEnabled(True)
        self.input_layout_2.addRow(self.lfp_low_width_label,
                                   self.lfp_low_width_edit)
        
        self.lfp_window_label = QLabel('Window type')
        windows = ['hann', 'hamming', 'blackmmaharris', 'barthann', 'nuttall',
            'blackman']
        self.lfp_window_edit = QComboBox(self)
        self.lfp_window_edit.addItems(windows)
        self.lfp_window_edit.setObjectName('lfp_window_edit')
        self.input_layout_2.addRow(self.lfp_window_label, self.lfp_window_edit)
        
        self.lfp_polyorder_label = QLabel('Polyorder')
        self.lfp_polyorder_edit = LineEdit()
        self.lfp_polyorder_edit.setValidator(QIntValidator())
        self.lfp_polyorder_edit.setObjectName('lfp_polyorder_edit')
        self.lfp_polyorder_edit.setEnabled(True)
        self.lfp_polyorder_edit.setText('3')
        self.input_layout_2.addRow(self.lfp_polyorder_label,
                                   self.lfp_polyorder_edit)
        
        self.lfp_pulse_start = QLabel('Pulse start')
        self.lfp_pulse_start_edit = LineEdit()
        self.lfp_pulse_start_edit.setEnabled(True)
        self.lfp_pulse_start_edit.setObjectName('lfp_pulse_start_edit')
        self.lfp_pulse_start_edit.setText('1000')
        self.input_layout_2.addRow(self.lfp_pulse_start,
                                   self.lfp_pulse_start_edit)
        
        
        #Tab1 buttons
        self.analyze_acq_button = QPushButton('Analyze acquisitions')
        self.tab1_layout.addWidget(self.analyze_acq_button)
        self.analyze_acq_button.clicked.connect(self.analyze)
        
        self.reset_button = QPushButton('Reset analysis')
        self.tab1_layout.addWidget(self.reset_button)   
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(False)
        
        #Analysis layout
        self.acquisition_number_label = QLabel('Acq number')
        self.acquisition_number = QSpinBox()
        self.acquisition_number.valueChanged.connect(self.acq_spinbox)
        self.analysis_buttons_layout.addRow(self.acquisition_number_label, 
                                     self.acquisition_number)
        self.acquisition_number.setEnabled(False)
        
        self.epoch_label = QLabel('Epoch')
        self.epoch_number = QLineEdit()
        self.analysis_buttons_layout.addRow(self.epoch_label,
                                            self.epoch_number)
        
        self.final_analysis_button = QPushButton('Final analysis')
        self.analysis_buttons_layout.addRow(self.final_analysis_button)
        self.final_analysis_button.clicked.connect(self.final_analysis)
        self.final_analysis_button.setEnabled(False)
        
        self.oepsc_amp_label = QLabel('oEPSC amp')
        self.oepsc_amp_edit = QLineEdit()
        self.o_info_layout.addRow(self.oepsc_amp_label,
                                     self.oepsc_amp_edit)
        
        self.oepsc_decay_label = QLabel('oEPSC decay')
        self.oepsc_decay_edit = QLineEdit()
        self.o_info_layout.addRow(self.oepsc_decay_label,
                                      self.oepsc_decay_edit)
        
        self.set_peak_button = QPushButton('Set point as peak')
        # self.set_fv_button.clicked.connect(self.set_fv)
        self.o_info_layout.addRow(self.set_peak_button)
        self.set_peak_button.setEnabled(False)
        
        self.delete_oepsc_button = QPushButton('Delete oEPSC')
        self.delete_oepsc_button.clicked.connect(self.delete_oepsc)
        self.o_info_layout.addRow(self.delete_oepsc_button)
        self.delete_oepsc_button.setEnabled(False)
        
        self.lfp_fv_label = QLabel('Fiber volley')
        self.lfp_fv_edit = QLineEdit()
        self.lfp_info_layout.addRow(self.lfp_fv_label,
                                    self.lfp_fv_edit)
        
        self.lfp_fp_label = QLabel('Field potential')
        self.lfp_fp_edit = QLineEdit()
        self.lfp_info_layout.addRow(self.lfp_fp_label,
                                    self.lfp_fp_edit)
        
        self.lfp_fp_slope_label = QLabel('FP slope')
        self.lfp_fp_slope_edit = QLineEdit()
        self.lfp_info_layout.addRow(self.lfp_fp_slope_label,
                                     self.lfp_fp_slope_edit)
        
        self.set_fv_button = QPushButton('Set point as fiber volley')
        self.set_fv_button.clicked.connect(self.set_point_as_fv)
        self.lfp_info_layout.addRow(self.set_fv_button)
        self.set_fv_button.setEnabled(False)
        
        self.set_fp_button = QPushButton('Set point as field potential')
        self.set_fp_button.clicked.connect(self.set_point_as_fp)
        self.lfp_info_layout.addRow(self.set_fp_button)
        self.set_fp_button.setEnabled(False)
    
        self.delete_lfp_button = QPushButton('Delete LFP')
        self.delete_lfp_button.clicked.connect(self.delete_lfp)
        self.lfp_info_layout.addRow(self.delete_lfp_button)
        self.delete_lfp_button.setEnabled(False)
        
        self.threadpool = QThreadPool()
        
        #Lists
        self.last_oepsc_point_clicked = []
        self.last_lfp_point_clicked = []
        self.oepsc_acq_dict = {}
        self.lfp_acq_dict = {}
        self.last_lfp_point_clicked = []
        self.last_oepsc_point_clicked = []
        self.oepsc_acqs_deleted = 0
        self.lfp_acqs_deleted = 0
        self.l_analysis_list = []
        self.o_analysis_list = []
        self.file_list_l = []
        self.file_list_o = []
        self.pref_dict = {}
        
        
    def load_files(self):
        if self.o_analyze.isChecked():
            file_extension_o = self.o_acq_id_edit.toText() + '_*.mat'
            file_list_o = glob(file_extension_o)
            filename_o = self.o_acq_id_edit.toText() + '_'
            filtered_list_o = [ x for x in file_list_o if "avg" not in x ]
            cleaned_o = [n.replace(filename_o, '') for n in filtered_list_o]
            self.file_list_o = sorted([int(n.replace('.mat', ''))
                                     for n in cleaned_o])
        else:
            file_list_o = []
        
        if self.lfp_analyze.isChecked():
            file_extension_l = self.lfp_acq_id_edit.toText() + '_*.mat'
            file_list_l = glob(file_extension_l)
            filename_l = self.lfp_acq_id_edit.toText() + '_'
            filtered_list_l = [ x for x in file_list_l if "avg" not in x ]
            cleaned_l = [n.replace(filename_l, '') for n in filtered_list_l]
            self.file_list_l = sorted([int(n.replace('.mat', ''))
                                     for n in cleaned_l])
        else:
            file_list_l = []
    
    
    def analyze(self):
        self.load_files()
        if len(self.file_list_l) == 0 and len(self.file_list_o) == 0:
            self.file_does_not_exist()
            pass
            return
        # else:
        #     self.pbar.setFormat('Analyzing...')
        #     self.pbar.setValue(0)
        if self.o_analyze.isChecked():
            self.delete_oepsc_button.setEnabled(True)
            self.o_analysis_list = np.arange(
                self.o_start_acq_edit.toInt(),
                self.o_end_acq_edit.toInt() + 1).tolist()
            for count, i in enumerate(self.o_analysis_list):
                if i in self.file_list_o:
                    oepsc = oEPSC(self.o_acq_id_edit.toText(), 
                        i, 
                        self.o_sample_rate_edit.toInt(), 
                        self.o_b_start_edit.toInt(), 
                        self.o_b_end_edit.toInt(), 
                        filter_type=self.o_filter_selection.currentText(),
                        order=self.o_order_edit.toInt(), 
                        high_pass=self.o_high_pass_edit.toInt(), 
                        high_width=self.o_high_width_edit.toInt(), 
                        low_pass=self.o_low_pass_edit.toInt(), 
                        low_width=self.o_low_width_edit.toInt(), 
                        window=self.o_window_edit.currentText(), 
                        polyorder=self.o_polyorder_edit.toInt(),
                        pulse_start=self.o_pulse_start_edit.toInt(),
                        n_window_start=self.o_neg_start_edit.toInt(),
                        n_window_end=self.o_neg_end_edit.toInt(),
                        p_window_start=self.o_pos_start_edit.toInt(),
                        p_window_end=self.o_pos_end_edit.toInt())
                    self.oepsc_acq_dict[str(i)] = oepsc
                else:
                    self.o_analysis_list.remove(i)       
        
        if self.lfp_analyze.isChecked():
            self.delete_lfp_button.setEnabled(True)
            self.l_analysis_list = np.arange(
                self.lfp_start_acq_edit.toInt(),
                self.lfp_end_acq_edit.toInt() + 1).tolist()
            for count, i in enumerate(self.l_analysis_list):
                if i in self.file_list_l:
                    # print(i)
                    lfp = LFP(self.lfp_acq_id_edit.toText(), 
                        i, 
                        self.lfp_sample_rate_edit.toInt(), 
                        self.lfp_b_start_edit.toInt(), 
                        self.lfp_b_end_edit.toInt(), 
                        filter_type=self.lfp_filter_selection.currentText(),
                        order=self.lfp_order_edit.toInt(), 
                        high_pass=self.lfp_high_pass_edit.toInt(), 
                        high_width=self.lfp_high_width_edit.toInt(), 
                        low_pass=self.lfp_low_pass_edit.toInt(), 
                        low_width=self.lfp_low_width_edit.toInt(), 
                        window=self.lfp_window_edit.currentText(), 
                        polyorder=self.lfp_polyorder_edit.toInt(),
                        pulse_start=self.lfp_pulse_start_edit.toInt())
                    self.lfp_acq_dict[str(i)] = lfp
                else:
                    self.l_analysis_list.remove(i)  
        # self.pbar.setValue(int(((count+1)/len(self.analysis_list))*100))
        if self.o_analyze.isChecked():
            acq_number = list(self.oepsc_acq_dict.keys())
        else:
            acq_number = list(self.lfp_acq_dict.keys())
        self.acquisition_number.setMaximum(int(acq_number[-1]))
        self.acquisition_number.setMinimum(int(acq_number[0])) 
        self.acquisition_number.setValue(int(acq_number[0]))
        self.acq_spinbox(int(acq_number[0]))
        self.analyze_acq_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.acquisition_number.setEnabled(True)
        self.final_analysis_button.setEnabled(True)
        self.set_fv_button.setEnabled(True)
        self.set_fp_button.setEnabled(True)
        # self.pbar.setFormat('Analysis finished')


    def acq_spinbox(self, h):
        self.acquisition_number.setDisabled(True)
        self.oepsc_plot.clear()
        self.lfp_plot.clear()
        self.last_oepsc_point_clicked = []
        self.last_lfp_point_clicked = []
        if (int(self.acquisition_number.value()) in self.o_analysis_list):
            self.oepsc_object = self.oepsc_acq_dict[str(
                self.acquisition_number.text())]
            self.oepsc_acq_plot = pg.PlotDataItem(
                x=self.oepsc_object.x_array, 
                y=self.oepsc_object.filtered_array, 
                name=str('oepsc_' + self.acquisition_number.text()),
                symbol='o', symbolSize=8, symbolBrush=(0,0,0,0),
                symbolPen=(0,0,0,0))
            self.oepsc_peak_plot = pg.PlotDataItem(
                x=self.oepsc_object.plot_peak_x(),
                y=self.oepsc_object.plot_peak_y(), symbol='o', symbolSize=8,
                symbolBrush='m', pen=None)
            self.oepsc_acq_plot.sigPointsClicked.connect(self.oepsc_plot_clicked)
            self.oepsc_plot.addItem(self.oepsc_acq_plot)
            self.oepsc_plot.addItem(self.oepsc_peak_plot)
            self.oepsc_plot.setXRange(self.o_pulse_start_edit.toInt()-10,
                self.o_b_start_edit.toInt()+350)
            self.oepsc_amp_edit.setText(
                str(self.round_sig(self.oepsc_object.peak_y)))
        if (int(self.acquisition_number.value()) in self.l_analysis_list):
            self.lfp_object = self.lfp_acq_dict[str(
                self.acquisition_number.text())]
            self.lfp_acq_plot = pg.PlotDataItem(
                x=self.lfp_object.x_array, 
                y=self.lfp_object.filtered_array, 
                name=str('oepsc_' + self.acquisition_number.text()),
                symbol='o', symbolSize=10, symbolBrush=(0,0,0,0),
                symbolPen=(0,0,0,0))
            self.lfp_points = pg.PlotDataItem(
                x=self.lfp_object.plot_elements_x(),
                y=self.lfp_object.plot_elements_y(), symbol='o', symbolSize=8,
                symbolBrush='m', pen=None)
            self.lfp_reg = pg.PlotDataItem(
                x=(self.lfp_object.slope_x
                   /self.lfp_object.s_r_c),
                y=self.lfp_object.reg_line, pen='g')
            self.lfp_acq_plot.sigPointsClicked.connect(self.lfp_plot_clicked)
            self.lfp_plot.addItem(self.lfp_acq_plot)
            self.lfp_plot.addItem(self.lfp_points)
            self.lfp_plot.addItem(self.lfp_reg)
            self.lfp_plot.setXRange(self.lfp_pulse_start_edit.toInt()-10,
                self.lfp_b_start_edit.toInt()+250)
            self.lfp_fv_edit.setText(
                str(self.round_sig(self.lfp_object.fv_y)))
            self.lfp_fp_edit.setText(
                str(self.round_sig(self.lfp_object.fp_y))) 
            self.lfp_fp_slope_edit.setText(
                str(self.round_sig(self.lfp_object.slope)))
        if self.o_analyze.isChecked():
            self.epoch_number.setText(self.oepsc_object.epoch)
        else:
            self.epoch_number.setText(self.lfp_object.epoch)
        self.acquisition_number.setEnabled(True)


    def reset(self):
        self.oepsc_plot.clear()
        self.lfp_plot.clear()
        self.o_analysis_list = []
        self.l_analysis_list = []
        self.oepsc_acq_dict = {}
        self.lfp_acq_dict = {}
        self.oepsc_acqs_deleted = []
        self.file_list_l = []
        self.file_list_o = []
        self.analyze_acq_button.setEnabled(True)
        self.lfp_info_layout.setEnabled(False)
        self.delete_oepsc_button.setEnabled(False)
        self.acquisition_number.setEnabled(False)
        self.final_analysis_button.setEnabled(False)
        self.set_fv_button.setEnabled(False)
        self.set_fp_button.setEnabled(False)
    
    
    def oepsc_plot_clicked(self, item, points):
        if len(self.last_oepsc_point_clicked) > 0:
            self.last_oepsc_point_clicked[0].resetPen()
            self.last_oepsc_point_clicked[0].resetBrush()
            self.last_oepsc_point_clicked[0].setSize(size=3)
        points[0].setPen('g', width=2)
        points[0].setBrush('w')
        points[0].setSize(size=8)
        # print(points[0].pos())
        self.last_oepsc_point_clicked = points
    
    
    def lfp_plot_clicked(self, item, points):
        if len(self.last_lfp_point_clicked) > 0:
            self.last_lfp_point_clicked[0].resetPen()
            self.last_lfp_point_clicked[0].resetBrush()
            self.last_lfp_point_clicked[0].setSize(size=3)
        points[0].setPen('g', width=2)
        points[0].setBrush('w')
        points[0].setSize(size=8)
        # print(points[0].pos())
        self.last_lfp_point_clicked = points
        

    def set_point_as_fv(self):
        '''
        This will set the LFP fiber volley as the point selected on the 
        lfp plot and update the other two acquisition plots.
    
        Returns
        -------
        None.
    
        '''
        x = (self.last_lfp_point_clicked[0].pos()[0]
             *self.lfp_acq_dict[self.acquisition_number.text()].s_r_c)
        y = self.last_lfp_point_clicked[0].pos()[1]
        self.lfp_acq_dict[self.acquisition_number.text()].change_fv(x, y)
        self.lfp_points.setData(x=self.lfp_object.plot_elements_x(),
            y=self.lfp_object.plot_elements_y(), symbol='o', symbolSize=8,
            symbolBrush='m', pen=None)
        self.lfp_fv_edit.setText(
            str(self.round_sig(self.lfp_object.fv_y)))
        self.last_lfp_point_clicked[0].resetPen()
        self.last_lfp_point_clicked[0].resetBrush()
        self.last_lfp_point_clicked = []


    def set_point_as_fp(self):
        '''
        This will set the LFP field potential as the point selected on the 
        lfp plot and update the other two acquisition plots.
    
        Returns
        -------
        None.
    
        '''
        x = (self.last_lfp_point_clicked[0].pos()[0]
             *self.lfp_acq_dict[self.acquisition_number.text()].s_r_c)
        y = self.last_lfp_point_clicked[0].pos()[1]
        self.lfp_acq_dict[self.acquisition_number.text()].change_fp(x, y)
        self.lfp_points.setData(x=self.lfp_object.plot_elements_x(),
            y=self.lfp_object.plot_elements_y(), symbol='o', symbolSize=8,
            symbolBrush='m', pen=None)
        self.lfp_fp_edit.setText(
            str(self.round_sig(self.lfp_object.fp_y))) 
        self.lfp_fp_slope_edit.setText(
            str(self.round_sig(self.lfp_object.slope)))
        self.last_lfp_point_clicked[0].resetPen()
        self.last_lfp_point_clicked[0].resetBrush()
        self.last_lfp_point_clicked = []


    def delete_oepsc(self):
        # self.deleted_acqs[str(
        #     self.acquisition_number.text())] = self.oepsc_acq_dict[str(
        #         self.acquisition_number.text())]
        # self.recent_reject_acq[str(
        #     self.acquisition_number.text())] = self.acq_dict[str(
        #         self.acquisition_number.text())]
        # print(self.acq_dict.keys())
        self.oepsc_plot.clear()
        del self.oepsc_acq_dict[str(self.acquisition_number.text())]
        # print(self.acq_dict.keys())
        self.o_analysis_list.remove(int(self.acquisition_number.text()))
        self.oepsc_acqs_deleted +=1
        
    
    def delete_lfp(self):
        # self.deleted_acqs[str(
        #     self.acquisition_number.text())] = self.oepsc_acq_dict[str(
        #         self.acquisition_number.text())]
        # self.recent_reject_acq[str(
        #     self.acquisition_number.text())] = self.acq_dict[str(
        #         self.acquisition_number.text())]
        # print(self.acq_dict.keys())
        self.lfp_plot.clear()
        del self.lfp_acq_dict[str(self.acquisition_number.text())]
        # print(self.acq_dict.keys())
       
        self.l_analysis_list.remove(int(self.acquisition_number.text()))
        self.lfp_acqs_deleted +=1
        
        
    def file_does_not_exist(self):
        self.dlg = QMessageBox(self)
        self.dlg.setWindowTitle('Error')
        self.dlg.setText('File does not exist')
        self.dlg.exec()
    
    
    def round_sig(self, x, sig=4):
        if isnan(x):
            return np.nan
        elif x == 0:
            return 0
        elif x != 0 or x is not np.nan or nan:
            return round(x, sig-int(floor(log10(abs(x))))-1)
        
    
    def final_analysis(self):
        self.final_analysis_button.setEnabled(False)
        if self.o_analyze.isChecked() and self.lfp_analyze.isChecked():
            self.final_data = FinalEvokedCurrent(self.oepsc_acq_dict,
                                                 self.lfp_acq_dict)
        elif self.o_analyze.isChecked() and self.lfp_analyze.isChecked():
            self.final_data = FinalEvokedCurrent(self.o_acq_dict)
        else:
            self.final_data = FinalEvokedCurrent(o_acq_dict=None,
                                    lfp_acq_dict = self.o_acq_dict)
        self.raw_datatable.setData(self.final_data.raw_df.T.to_dict('dict'))
        self.final_datatable.setData(
            self.final_data.final_df.T.to_dict('dict'))
       
        
    def save_as(self, save_filename):
        # self.pbar.setValue(0)
        # self.pbar.setFormat('Saving...')
        self.final_data.save_data(save_filename)
        if self.o_analyze.isChecked():
            self.worker = SaveWorker(save_filename, self.oepsc_acq_dict)
            # self.worker.signals.progress.connect(self.update_save_progress)
            # self.worker.signals.finished.connect(self.progress_finished)
            self.threadpool.start(self.worker)
        if self.lfp_analyze.isChecked():
            self.worker = SaveWorker(save_filename, self.lfp_acq_dict)
            # self.worker.signals.progress.connect(self.update_save_progress)
            # self.worker.signals.finished.connect(self.progress_finished)
            self.threadpool.start(self.worker)
        # self.pbar.setFormat('Data saved')
    
    
    def open_files(self):
        pass
    
    
    def create_pref_dict(self):
        line_edits = self.findChildren(QLineEdit)
        line_edit_dict = {}
        for i in line_edits:
            if i.objectName() != '':
                line_edit_dict[i.objectName()] = i.text()
        self.pref_dict['line_edits'] = line_edit_dict
        
        combo_box_dict = {}
        combo_boxes = self.findChildren(QComboBox)
        for i in combo_boxes:
            if i.objectName() != '':
                combo_box_dict[i.objectName()] = i.currentText()
        self.pref_dict['combo_boxes'] = combo_box_dict
        
        check_box_dict = {}
        check_boxes = self.findChildren(QCheckBox)
        for i in check_boxes:
            if i.objectName() != '':
                check_box_dict[i.objectName()] = i.isChecked()
        self.pref_dict['check_boxes'] = check_box_dict  
           
        buttons_dict = {}
        buttons = self.findChildren(QPushButton)
        for i in buttons:
            if i.objectName() != '':
                buttons_dict[i.objectName()] = i.isEnabled()
        self.pref_dict['buttons'] = buttons_dict
    
    
    def set_preferences(self, pref_dict):
        line_edits = self.findChildren(QLineEdit)
        for i in line_edits:
            if i.objectName() != '':
                try:
                    i.setText(pref_dict['line_edits'][i.objectName()])
                except:
                    pass
                
        combo_boxes = self.findChildren(QComboBox)
        for i in combo_boxes:
            if i.objectName() != '':
                try:
                    i.setCurrentText(pref_dict['combo_boxes'][i.objectName()])
                except:
                    pass
        
        check_boxes = self.findChildren(QCheckBox)
        for i in check_boxes:
            if i.objectName() != '':
                try:
                    i.setChecked(pref_dict['check_boxes'][i.objectName()])
                except:
                    pass
                
        buttons = self.findChildren(QPushButton)
        for i in buttons:
            if i.objectName() != '':
                i.setEnabled(pref_dict['buttons'][i.objectName()])
    
    
    def load_preferences(self, file_name):
        load_dict = YamlWorker.load_yaml(file_name)
        self.set_preferences(load_dict)
    
    
    def save_preferences(self, save_filename):
        self.create_pref_dict()
        if self.pref_dict:
            YamlWorker.save_yaml(self.pref_dict, save_filename)
        else:
            pass
    
if __name__ == '__main__':
    oEPSCWidget()
    
    
    
    
    
    
    