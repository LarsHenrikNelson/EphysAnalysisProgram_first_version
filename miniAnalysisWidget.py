#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:15:23 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: larsnelson
"""

from math import log10, floor
from glob import glob
import json

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
                             QWidget, QLabel, QFormLayout, QComboBox, QSpinBox,
                             QCheckBox, QProgressBar, QMessageBox, QTabWidget,
                             QScrollArea, QSizePolicy, QListWidget)
from PyQt5.QtGui import QIntValidator, QKeySequence, QShortcut
from PyQt5.QtCore import QThreadPool, Qt
import pyqtgraph as pg

from acq_class import MiniAnalysis, LoadMiniAnalysis
from final_analysis_classes import FinalMiniAnalysis
from load_classes import LoadMiniSaveData
from utilities import load_scanimage_file
from utility_classes import (LineEdit, MiniSaveWorker, MplWidget,
                             DistributionPlot, YamlWorker, ListView,
                             ListModel)


class miniAnalysisWidget(QWidget):

    def __init__(self, parent=None):
        
        super(miniAnalysisWidget, self).__init__(parent)
        
        # self.parent = parent
        
        #Create tabs for part of the analysis program
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        self.tab1_scroll = QScrollArea()
        self.tab1_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tab1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tab1_scroll.setWidgetResizable(True)
        
        self.tab1 = QWidget()
        self.tab1_scroll.setWidget(self.tab1)
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        self.tab_widget.addTab(self.tab1_scroll, 'Setup')
        self.tab_widget.addTab(self.tab2, 'Analysis')
        self.tab_widget.addTab(self.tab3, 'Final data')
        
        self.setStyleSheet('''QTabWidget::tab-bar 
                                          {alignment: left;}''')
        
                                            
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.main_layout.addWidget(self.pbar)
        
        
        #Tab 1 layouts
        self.setup_layout = QHBoxLayout()
        self.extra_layout = QVBoxLayout()
        self.other_layout = QHBoxLayout()
        self.input_layout = QFormLayout()
        self.load_layout = QVBoxLayout()
        self.load_widget = ListView()
        self.acq_model = ListModel()
        self.load_widget.setModel(self.acq_model)
        self.load_layout.addWidget(self.load_widget)
        self.input_layout.setFieldGrowthPolicy(
            QFormLayout.FieldsStayAtSizeHint)
        self.settings_layout = QFormLayout()
        self.settings_layout.setFieldGrowthPolicy(
            QFormLayout.FieldsStayAtSizeHint)
        self.template_form = QFormLayout()
        self.template_form.setFieldGrowthPolicy(
            QFormLayout.FieldsStayAtSizeHint)
        self.tab1.setLayout(self.setup_layout)
        self.setup_layout.addLayout(self.input_layout, 0)
        self.setup_layout.addLayout(self.extra_layout, 0)
        self.setup_layout.addLayout(self.load_layout)
        self.extra_layout.addLayout(self.other_layout, 0)
        self.other_layout.addLayout(self.settings_layout, 0)
        self.other_layout.addLayout(self.template_form, 0)
        
        #Tab 2 layouts
        self.plot_layout = QVBoxLayout()
        self.acq_layout = QHBoxLayout()
        self.mini_view_layout = QHBoxLayout()
        self.mini_tab = QTabWidget()
        self.acq_buttons = QFormLayout()
        self.mini_layout = QFormLayout()
        self.tab2.setLayout(self.plot_layout)
        self.plot_layout.addLayout(self.mini_view_layout, 1)
        self.plot_layout.addLayout(self.acq_layout, 1)
        self.acq_layout.addLayout(self.acq_buttons, 0)
        self.acq_layout.addWidget(self.mini_tab, 1)
        
        #Tab 3 layouts and setup
        self.table_layout = QVBoxLayout()
        self.tab3.setLayout(self.table_layout)
        self.data_layout = QHBoxLayout()
        self.table_layout.addLayout(self.data_layout, 1)
        self.raw_data_table = pg.TableWidget(sortable=False)
        self.final_table = pg.TableWidget(sortable=False)
        self.ave_mini_plot = pg.PlotWidget()
        self.data_layout.addWidget(self.raw_data_table, 1)
        self.final_data_layout = QVBoxLayout()
        self.final_data_layout.addWidget(self.final_table,1)
        self.final_data_layout.addWidget(self.ave_mini_plot,2)
        self.data_layout.addLayout(self.final_data_layout)
        self.mw = MplWidget()
        self.amp_dist = DistributionPlot()
        self.plot_selector = QComboBox()
        self.plot_selector.currentTextChanged.connect(self.plot_raw_data)
        self.matplotlib_layout_h = QHBoxLayout()
        self.matplotlib_layout_h.addWidget(self.plot_selector, 0)
        self.matplotlib_layout_h.addWidget(self.mw, 2)
        self.matplotlib_layout_h.addWidget(self.amp_dist, 1)
        self.table_layout.addLayout(self.matplotlib_layout_h, 1)
        
        #Tab2 acq_buttons layout
        self.acquisition_number_label = QLabel('Acq number')
        self.acquisition_number = QSpinBox()
        self.acquisition_number.valueChanged.connect(self.acq_spinbox)
        self.acq_buttons.addRow(self.acquisition_number_label, 
                                     self.acquisition_number)
        
        self.epoch_label = QLabel('Epoch')
        self.epoch_edit = QLineEdit()
        self.acq_buttons.addRow(self.epoch_label, self.epoch_edit)
        
        self.baseline_mean_label = QLabel('Baseline mean')
        self.baseline_mean_edit = QLineEdit()
        self.acq_buttons.addRow(self.baseline_mean_label,
                                     self.baseline_mean_edit)
        
        self.create_mini_button = QPushButton('Create new mini')
        self.create_mini_button.clicked.connect(self.create_mini)
        self.acq_buttons.addRow(self.create_mini_button)
        
        self.delete_acq_button = QPushButton('Delete acquisition')
        self.delete_acq_button.clicked.connect(self.delete_acq)
        self.acq_buttons.addRow(self.delete_acq_button)
        
        self.reset_recent_acq_button = QPushButton('Reset recent acq')
        self.reset_recent_acq_button.clicked.connect(
            self.reset_recent_reject_acq)
        self.acq_buttons.addRow(self.reset_recent_acq_button)
        
         #Filling the plot layout.
        pg.setConfigOptions(antialias=True)
        self.p1 = pg.PlotWidget()
        self.p1.setMinimumWidth(500)
        self.acq_layout.addWidget(self.p1, 1)
        
        self.p2 = pg.PlotWidget()
        self.mini_view_layout.addWidget(self.p2, 1)
        
        self.region = pg.LinearRegionItem()

        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
        self.region.sigRegionChanged.connect(self.update)
        self.p1.sigRangeChanged.connect(self.updateRegion)
        
        #Template plot
        # self.p3 = pg.PlotWidget()
        # self.p3.sigRangeChanged.connect(self.updateRegion)
        
        self.tab1 = self.p1
        # self.tab2 = self.p3
        self.mini_tab.addTab(self.tab1, 'Acq view')
        # self.mini_tab.addTab(self.tab2, 'Create event')
        
        self.mini_view_layout.addLayout(self.mini_layout)
        self.mini_number_label = QLabel('Event')
        self.mini_number = QSpinBox()
        self.mini_number.valueChanged.connect(self.mini_spinbox)
        self.mini_layout.addRow(self.mini_number_label, self.mini_number)
        
        self.mini_baseline_label = QLabel('Baseline (pA)')
        self.mini_baseline = QLineEdit()
        self.mini_layout.addRow(self.mini_baseline_label,
                                self.mini_baseline)
        
        self.mini_amplitude_label = QLabel('Amplitude (pA)')
        self.mini_amplitude = QLineEdit()
        self.mini_layout.addRow(self.mini_amplitude_label,
                                self.mini_amplitude)
        
        self.mini_tau_label = QLabel('Tau (ms)')
        self.mini_tau = QLineEdit()
        self.mini_layout.addRow(self.mini_tau_label,
                                self.mini_tau)
        
        self.mini_rise_time_label = QLabel('Rise time (ms)')
        self.mini_rise_time = QLineEdit()
        self.mini_layout.addRow(self.mini_rise_time_label,
                                self.mini_rise_time)
        
        self.mini_rise_rate_label = QLabel('Rise rate (pA/ms)')
        self.mini_rise_rate = QLineEdit()
        self.mini_layout.addRow(self.mini_rise_rate_label,
                                self.mini_rise_rate)
        
        self.delete_mini_button = QPushButton('Delete event')
        self.mini_layout.addRow(self.delete_mini_button)
        self.delete_mini_button.clicked.connect(self.delete_mini)
        
        
        self.set_baseline = QPushButton('Set point as baseline')
        self.mini_layout.addRow(self.set_baseline)
        self.set_baseline.clicked.connect(self.set_point_as_baseline)
        
        self.set_peak = QPushButton('Set point as peak')
        self.mini_layout.addRow(self.set_peak)
        self.set_peak.clicked.connect(self.set_point_as_peak)
        
        self.mini_view_widget = pg.PlotWidget()
        self.mini_view_layout.addWidget(self.mini_view_widget, 1)
        
        #Tab1 input
        # self.acq_id_label = QLabel('Acq ID')
        # self.acq_id_edit = LineEdit()
        # self.acq_id_edit.setObjectName('acq_id_edit')
        # self.acq_id_edit.setEnabled(True)
        # self.input_layout.addRow(self.acq_id_label, self.acq_id_edit)
    
        # self.start_acq_label = QLabel('Start acq')
        # self.start_acq_edit = LineEdit()
        # self.start_acq_edit.setObjectName('start_acq_edit')
        # self.start_acq_edit.setEnabled(True)
        # self.input_layout.addRow(self.start_acq_label, self.start_acq_edit)
        
        # self.end_acq_label = QLabel('End acq')
        # self.end_acq_edit = LineEdit()
        # self.end_acq_edit.setObjectName('end_acq_edit')
        # self.end_acq_edit.setEnabled(True)
        # self.input_layout.addRow(self.end_acq_label, self.end_acq_edit)

        self.b_start_label = QLabel('Baseline start (ms)')
        self.b_start_edit = LineEdit()
        self.b_start_edit.setObjectName('b_start_edit')
        self.b_start_edit.setEnabled(True)
        self.b_start_edit.setText('0')
        self.input_layout.addRow(self.b_start_label, self.b_start_edit)
        
        self.b_end_label = QLabel('Baseline end (ms)')
        self.b_end_edit = LineEdit()
        self.b_end_edit.setObjectName('b_end_edit')
        self.b_end_edit.setEnabled(True)
        self.b_end_edit.setText('80')
        self.input_layout.addRow(self.b_end_label, self.b_end_edit)
        
        self.sample_rate_label = QLabel('Sample rate')
        self.sample_rate_edit = LineEdit()
        self.sample_rate_edit.setObjectName('sample_rate_edit')
        self.sample_rate_edit.setEnabled(True)
        self.sample_rate_edit.setText('10000')
        self.input_layout.addRow(self.sample_rate_label,
                                 self.sample_rate_edit)

        self.rc_checkbox_label = QLabel('RC check')
        self.rc_checkbox = QCheckBox(self)
        self.rc_checkbox.setObjectName('rc_checkbox')
        self.rc_checkbox.setChecked(True)
        self.rc_checkbox.setTristate(False)
        self.input_layout.addRow(self.rc_checkbox_label,
                                 self.rc_checkbox)

        self.rc_check_start = QLabel('RC Check Start (ms)')
        self.rc_check_start_edit = LineEdit()
        self.rc_check_start_edit.setEnabled(True)
        self.rc_check_start_edit.setObjectName('rc_check_start_edit')
        self.rc_check_start_edit.setText('10000')
        self.input_layout.addRow(self.rc_check_start,
                                 self.rc_check_start_edit)
        
        self.rc_check_end = QLabel('RC Check End (ms)')
        self.rc_check_end_edit = LineEdit()
        self.rc_check_end_edit.setEnabled(True)
        self.rc_check_end.setObjectName('rc_check_end_edit')
        self.rc_check_end_edit.setText('10300')
        self.input_layout.addRow(self.rc_check_end,
                                 self.rc_check_end_edit)

        self.filter_type_label = QLabel('Filter Type')
        filters = ['remez_2', 'fir_zero_2', 'bessel', 'fir_zero_1', 'savgol', 
                   'median', 'remez_1', 'None']
        self.filter_selection= QComboBox(self)
        self.filter_selection.addItems(filters)
        self.filter_selection.setObjectName('filter_selection')
        self.input_layout.addRow(self.filter_type_label, self.filter_selection)
        
        self.order_label = QLabel('Order')
        self.order_edit = LineEdit()
        self.order_edit.setValidator(QIntValidator())
        self.order_edit.setObjectName('order_edit')
        self.order_edit.setEnabled(True)
        self.order_edit.setText('201')
        self.input_layout.addRow(self.order_label, self.order_edit)
        
        self.high_pass_label = QLabel('High-pass')
        self.high_pass_edit = LineEdit()
        self.high_pass_edit.setValidator(QIntValidator())
        self.high_pass_edit.setObjectName('high_pass_edit')
        self.high_pass_edit.setEnabled(True)
        self.input_layout.addRow(self.high_pass_label, self.high_pass_edit)
        
        self.high_width_label = QLabel('High-width')
        self.high_width_edit = LineEdit()
        self.high_width_edit.setValidator(QIntValidator())
        self.high_width_edit.setObjectName('high_width_edit')
        self.high_width_edit.setEnabled(True)
        self.input_layout.addRow(self.high_width_label, self.high_width_edit)
        
        self.low_pass_label = QLabel('Low-pass')
        self.low_pass_edit = LineEdit()
        self.low_pass_edit.setValidator(QIntValidator())
        self.low_pass_edit.setObjectName('low_pass_edit')
        self.low_pass_edit.setEnabled(True)
        self.low_pass_edit.setText('600')
        self.input_layout.addRow(self.low_pass_label, self.low_pass_edit)
        
        self.low_width_label = QLabel('Low-width')
        self.low_width_edit = LineEdit()
        self.low_width_edit.setValidator(QIntValidator())
        self.low_width_edit.setObjectName('low_width_edit')
        self.low_width_edit.setEnabled(True)
        self.low_width_edit.setText('600')
        self.input_layout.addRow(self.low_width_label, self.low_width_edit)
        
        self.window_label = QLabel('Window type')
        windows = ['hann', 'hamming', 'blackmmaharris', 'barthann', 'nuttall',
            'blackman']
        self.window_edit = QComboBox(self)
        self.window_edit.addItems(windows)
        self.window_edit.setObjectName('window_edit')
        self.input_layout.addRow(self.window_label, self.window_edit)
        
        self.polyorder_label = QLabel('Polyorder')
        self.polyorder_edit = LineEdit()
        self.polyorder_edit.setValidator(QIntValidator())
        self.polyorder_edit.setObjectName('polyorder_edit')
        self.polyorder_edit.setEnabled(True)
        self.input_layout.addRow(self.polyorder_label, self.polyorder_edit)
        
        self.analyze_acq_button = QPushButton('Analyze acquisitions')
        self.input_layout.addRow(self.analyze_acq_button)
        self.analyze_acq_button.setObjectName('analyze_acq_button')
        self.analyze_acq_button.setSizePolicy(QSizePolicy.Preferred,
                                                QSizePolicy.Preferred)
        self.analyze_acq_button.setMaximumWidth(230)
        self.analyze_acq_button.clicked.connect(self.analyze)
        
        self.calculate_parameters = QPushButton('Calculate Parameters')
        self.input_layout.addRow(self.calculate_parameters)
        self.calculate_parameters.setObjectName('calculate_parameters')
        self.calculate_parameters.setSizePolicy(QSizePolicy.Preferred,
                                                QSizePolicy.Preferred)
        self.calculate_parameters.setMaximumWidth(230)
        self.calculate_parameters.clicked.connect(self.final_analysis)
        self.calculate_parameters.setEnabled(False)
        
        self.reset_button = QPushButton('Reset Analysis')
        self.input_layout.addRow(self.reset_button)
        self.reset_button.setSizePolicy(QSizePolicy.Preferred,
                                                QSizePolicy.Preferred)
        self.reset_button.setMaximumWidth(230)
        self.reset_button.clicked.connect(self.reset)
        
        self.reset_button.setObjectName('reset_button')
        
        self.sensitivity_label = QLabel('Sensitivity')
        self.sensitivity_edit = LineEdit()
        self.sensitivity_edit.setObjectName('sensitivity_edit')
        self.sensitivity_edit.setEnabled(True)
        self.sensitivity_edit.setText('4')
        self.settings_layout.addRow(self.sensitivity_label,
                                 self.sensitivity_edit)
        
        self.amp_thresh_label = QLabel('Amplitude Threshold (pA)')
        self.amp_thresh_edit = LineEdit()
        self.amp_thresh_edit.setObjectName('amp_thresh_edit')
        self.amp_thresh_edit.setEnabled(True)
        self.amp_thresh_edit.setText('4')
        self.settings_layout.addRow(self.amp_thresh_label,
                                 self.amp_thresh_edit)
        
        self.mini_spacing_label = QLabel('Min mini spacing (ms)')
        self.mini_spacing_edit = LineEdit()
        self.mini_spacing_edit.setObjectName('mini_spacing_edit')
        self.mini_spacing_edit.setEnabled(True)
        self.mini_spacing_edit.setText('7.5')
        self.settings_layout.addRow(self.mini_spacing_label,
                                 self.mini_spacing_edit)
        
        self.min_rise_time_label = QLabel('Min rise time (ms)')
        self.min_rise_time = LineEdit()
        self.min_rise_time.setObjectName('min_rise_time')
        self.min_rise_time.setEnabled(True)
        self.min_rise_time.setText('0.5')
        self.settings_layout.addRow(self.min_rise_time_label,
                                    self.min_rise_time)
        
        self.min_decay_label = QLabel('Min decay time (ms)')
        self.min_decay = LineEdit()
        self.min_decay.setObjectName('min_decay')
        self.min_decay.setEnabled(True)
        self.min_decay.setText('0.5')
        self.settings_layout.addRow(self.min_decay_label,
                                    self.min_decay)
    
    
        self.curve_fit_decay_label = QLabel('Curve fit decay')
        self.curve_fit_decay = QCheckBox(self)
        self.curve_fit_decay.setObjectName('curve_fit_decay')
        self.curve_fit_decay.setChecked(False)
        self.curve_fit_decay.setTristate(False)
        self.settings_layout.addRow(self.curve_fit_decay_label,
                                 self.curve_fit_decay)
        
        self.invert_label = QLabel('Invert (For positive currents)')
        self.invert_checkbox = QCheckBox(self)
        self.invert_checkbox.setObjectName('invert_checkbox')
        self.invert_checkbox.setChecked(False)
        self.invert_checkbox.setTristate(False)
        self.settings_layout.addRow(self.invert_label,
                                 self.invert_checkbox)
        
        self.decon_type_label = QLabel('Deconvolution type')
        self.decon_type_edit = QComboBox(self)
        decon_list = ['weiner', 'fft']
        self.decon_type_edit.addItems(decon_list)
        self.decon_type_edit.setObjectName('decon_type_edit')
        self.settings_layout.addRow(self.decon_type_label,
                                    self.decon_type_edit)
        
        self.tau_1_label = QLabel('Rise tau (ms)')
        self.tau_1_edit = LineEdit()
        self.tau_1_edit.setObjectName('tau_1_edit')
        self.tau_1_edit.setEnabled(True)
        self.tau_1_edit.setText('0.3')
        self.template_form.addRow(self.tau_1_label, self.tau_1_edit)
        
        self.tau_2_label = QLabel('Decay tau (ms)')
        self.tau_2_edit = LineEdit()
        self.tau_2_edit.setObjectName('tau_2_edit')
        self.tau_2_edit.setEnabled(True)
        self.tau_2_edit.setText('5')
        self.template_form.addRow(self.tau_2_label, self.tau_2_edit)
        
        self.amplitude_label = QLabel('Amplitude (pA)')
        self.amplitude_edit = LineEdit()
        self.amplitude_edit.setObjectName('amplitude_edit')
        self.amplitude_edit.setEnabled(True)
        self.amplitude_edit.setText('-20')
        self.template_form.addRow(self.amplitude_label, self.amplitude_edit)
        
        self.risepower_label = QLabel('Risepower')
        self.risepower_edit = LineEdit()
        self.risepower_edit.setObjectName('risepower_edit')
        self.risepower_edit.setEnabled(True)
        self.risepower_edit.setText('0.5')
        self.template_form.addRow(self.risepower_label, self.risepower_edit)
        
        self.temp_length_label = QLabel('Template length (ms)')
        self.temp_length_edit = LineEdit()
        self.temp_length_edit.setObjectName('temp_length_edit')
        self.temp_length_edit.setEnabled(True)
        self.temp_length_edit.setText('30')
        self.template_form.addRow(self.temp_length_label,
                                  self.temp_length_edit)
        
        self.spacer_label = QLabel('Spacer (ms)')
        self.spacer_edit = LineEdit()
        self.spacer_edit.setObjectName('spacer_edit')
        self.spacer_edit.setEnabled(True)
        self.spacer_edit.setText('2')
        self.template_form.addRow(self.spacer_label, self.spacer_edit)
        
        self.template_button = QPushButton('Create template')
        self.template_form.addRow(self.template_button)
        self.template_button.clicked.connect(self.create_template)
        self.template_button.setMaximumWidth(235)
        self.template_button.setObjectName('template_button')
        
        self.template_plot = pg.PlotWidget()
        self.template_plot.setMinimumWidth(300)
        self.template_plot.setMinimumHeight(300)
        self.extra_layout.addWidget(self.template_plot, 0)
        

        #Setup for the drag and drop load layout
        self.del_sel_button = QPushButton('Delete selection')
        self.load_layout.addWidget(self.del_sel_button)
        self.del_sel_button.clicked.connect(self.del_selection)

        self.threadpool = QThreadPool()
        
        self.acq_dict = {}
        self.analysis_list = []
        self.acq_object = None
        self.file_list = []
        self.last_mini_deleted = {}
        self.last_mini_deleted = []
        self.deleted_acqs = {}
        self.recent_reject_acq = {}
        self.last_mini_point_clicked = []
        self.last_acq_point_clicked = []
        self.recent_reject_acq = {}
        self.mini_spinbox_list = []
        self.last_mini_clicked_1 = []
        self.last_mini_clicked_2 = []
        self.sort_index = []
        self.template = []
        self.mini_spinbox_list = []
        self.save_values = []
        self.minis_deleted = 0
        self.acqs_deleted = 0
        self.calc_param_clicked = False
        self.pref_dict = {}
        
        #Shortcuts
        self.del_mini_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self.del_mini_shortcut.activated.connect(self.delete_mini)
        
        self.create_mini_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.create_mini_shortcut.activated.connect(self.create_mini)
        
        self.del_acq_shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        self.del_acq_shortcut.activated.connect(self.delete_acq)


    def del_selection(self):
        indexes = self.load_widget.selectedIndexes()
        print([i.row() for i in indexes])
        print(self.acq_model.acq_list)
        if len(indexes) > 0:
            for index in sorted(indexes, reverse=True):
                print(index.row())
                del self.acq_model.acq_list[index.row()]
                del self.acq_model.fname_list[index.row()]
            print(self.acq_model.acq_list)
            self.acq_model.layoutChanged.emit()
            self.load_widget.clearSelection()

    
    def tm_psp(self):
        '''
        This function create template that can be use for the mini analysis.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        s_r_c = self.sample_rate_edit.toInt()/1000
        amplitude = self.amplitude_edit.toFloat()
        tau_1 = self.tau_1_edit.toFloat() * s_r_c
        tau_2 = self.tau_2_edit.toFloat() * s_r_c
        risepower = self.risepower_edit.toFloat()
        t_psc =  np.arange(0, 
                           int(self.temp_length_edit.toFloat()
                               * s_r_c))
        spacer = int(self.spacer_edit.toFloat() * s_r_c)
        self.template = np.zeros(len(t_psc)+spacer)
        offset = len(self.template)-len(t_psc)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        y = amplitude/Aprime*((1-(np.exp(-t_psc/tau_1)))**risepower
                              * np.exp((-t_psc/tau_2)))
        self.template[offset:] = y
        # return self.template
    
    
    def create_template(self):
        self.template_plot.clear()
        self.tm_psp()
        s_r_c = self.sample_rate_edit.toInt()/1000
        self.template_plot.plot(x=(np.arange(len(self.template))
                                            /s_r_c),
                                            y=self.template)


    # def load_files(self):
        # file_extension = self.acq_id_edit.toText() + '_' + '*.mat'
        # filename = self.acq_id_edit.toText() + '_'
        # file_list = glob(file_extension)
        # if not file_list:
        #     self.file_list = None
        # else:
        #     filtered_list = [ x for x in file_list if "avg" not in x ]
        #     cleaned_1 = [n.replace(filename, '') for n in filtered_list]
        #     self.file_list = sorted([int(n.replace('.mat', ''))
        #                              for n in cleaned_1])
   

    def analyze(self):
        self.analyze_acq_button.setEnabled(False)
        if len(self.template) == 0:
            self.create_template()
        # self.load_files()
        if len(self.acq_model.fname_list) == 0:
            self.file_does_not_exist()
            self.analyze_acq_button.setEnabled(True)
        else:
            # self.analysis_list = np.arange(self.start_acq_edit.toInt(),
            #                           self.end_acq_edit.toInt()+1).tolist()
            self.pbar.setFormat('Analyzing...')
            self.pbar.setValue(0)
            # if len(self.template) > 0:
            #     template = self.template
            # else:
            #     template = None
            for count, i in enumerate(self.acq_model.fname_list):
                acq_components = load_scanimage_file(i)
                x = MiniAnalysis(
                    acq_components = acq_components,
                    sample_rate=self.sample_rate_edit.toInt(), 
                    baseline_start=self.b_start_edit.toInt(), 
                    baseline_end=self.b_end_edit.toInt(),
                    filter_type=self.filter_selection.currentText(), 
                    order=self.order_edit.toInt(), 
                    high_pass=self.high_pass_edit.toInt(), 
                    high_width=self.high_width_edit.toInt(), 
                    low_pass=self.low_pass_edit.toInt(), 
                    low_width=self.low_width_edit.toInt(), 
                    window=self.window_edit.currentText(), 
                    polyorder=self.polyorder_edit.toInt(),
                    template=self.template,
                    rc_check=self.rc_checkbox.isChecked(),
                    rc_check_start=self.rc_check_start_edit.toInt(),
                    rc_check_end=self.rc_check_end_edit.toInt(),
                    sensitivity=self.sensitivity_edit.toFloat(),
                    amp_threshold=self.amp_thresh_edit.toFloat(),
                    mini_spacing=self.mini_spacing_edit.toFloat(),
                    min_rise_time=self.min_rise_time.toFloat(),
                    min_decay_time=self.min_decay.toFloat(),
                    invert=self.invert_checkbox.isChecked(),
                    decon_type = self.decon_type_edit.currentText(),
                    curve_fit_decay = self.curve_fit_decay.isChecked()
                    )
                x.analyze()
                self.acq_dict[x.acq_number] = x
                self.pbar.setValue(int(((count+1)/len(self.acq_model.fname_list))*100))
            acq_number = list(self.acq_dict.keys())
            self.analysis_list = [int(i) for i in self.acq_dict.keys()]
            print(self.acq_dict.keys())
            self.acquisition_number.setMaximum(int(acq_number[-1]))
            self.acquisition_number.setMinimum(int(acq_number[0])) 
            self.acquisition_number.setValue(int(acq_number[0]))
            self.mini_number.setMinimum(0)
            # self.acq_spinbox(self.start_acq_edit.toInt())
            self.mini_spinbox(0)
            self.analyze_acq_button.setEnabled(True)
            self.calculate_parameters.setEnabled(True)
            self.pbar.setFormat('Analysis finished')

    
    def acq_spinbox(self, h):
        self.p1.clear()
        self.p2.clear()
        self.mini_view_widget.clear()
        self.last_acq_point_clicked = []
        self.last_mini_clicked_1 = []
        self.last_mini_clicked_2 = []
        self.last_mini_point_clicked = []
        self.acq_object = None
        self.sort_index = []
        self.mini_spinbox_list = []
        if (int(self.acquisition_number.value()) in self.analysis_list):
            self.acquisition_number.setDisabled(True)
            self.acq_object = self.acq_dict[
                str(self.acquisition_number.value())]
            self.epoch_edit.setText(self.acq_object.epoch)
            acq_plot = pg.PlotDataItem(x=self.acq_object.x_array, 
                            y=self.acq_object.final_array, 
                            name=str(self.acquisition_number.text()),
                            symbol='o', symbolSize=8, symbolBrush=(0,0,0,0),
                            symbolPen=(0,0,0,0))
            acq_plot.sigPointsClicked.connect(self.acq_plot_clicked)
            self.p1.addItem(acq_plot)
            self.p2.plot(x=self.acq_object.x_array, 
                         y=self.acq_object.final_array)
            self.p2.addItem(self.region, ignoreBounds=True)
            self.region.setRegion([0, 400])
            self.region.setZValue(10)
            self.p1.setAutoVisible(y=True)
            # self.p3.setAutoVisible(y=True)
            if self.acq_object.postsynaptic_events:
                self.mini_spinbox_list = list(
                    range(len(self.acq_object.postsynaptic_events)))
                self.sort_index = list(
                    np.argsort(self.acq_object.final_events))
                for i in self.mini_spinbox_list:
                    mini_plot = pg.PlotCurveItem(
                        x=self.acq_object.postsynaptic_events[i].mini_plot_x, 
                        y=self.acq_object.postsynaptic_events[i].mini_plot_y,
                        pen='g', name = i, clickable=True)
                    mini_plot.sigClicked.connect(self.mini_clicked)
                    self.p1.addItem(mini_plot)
                    self.p2.plot(
                        x=self.acq_object.postsynaptic_events[i].mini_plot_x,
                        y=self.acq_object.postsynaptic_events[i].mini_plot_y,
                        pen='g')
                self.mini_number.setMinimum(self.mini_spinbox_list[0])
                self.mini_number.setMaximum(self.mini_spinbox_list[-1])
                self.mini_number.setValue(self.mini_spinbox_list[0])
                self.mini_spinbox(self.mini_spinbox_list[0])
                self.acquisition_number.setEnabled(True)
            else:
                pass
        else:
            pass
    
    
    def reset(self):
        self.p1.clear()
        self.p2.clear()
        self.acq_model.acq_list = []
        self.acq_model.fname_list = []
        self.acq_model.layoutChanged.emit()
        self.calc_param_clicked = False
        self.mini_view_widget.clear()
        self.acq_dict = {}
        self.acq_object = None
        self.analysis_list = []
        self.file_list = []
        self.last_mini_point_clicked = []
        self.last_acq_point_clicked = []
        self.deleted_acqs = {}
        self.recent_reject_acq = {}
        self.last_mini_clicked_1 = []
        self.last_mini_clicked_2 = []
        self.mini_spinbox_list = []
        self.sort_index = []
        self.raw_df = {}
        self.save_values = []
        self.analyze_acq_button.setEnabled(True)
        self.pbar.setValue(0)
        self.pbar.setFormat('')
        self.pbar.setValue(0)
        self.calculate_parameters.setEnabled(True)
        self.raw_data_table.clear()
        self.mw.clear()
        self.amp_dist.clear()
        self.minis_deleted = 0
        self.acqs_deleted = 0
        self.final_table.clear()
        self.ave_mini_plot.clear()
        self.pref_dict = {}
        self.calc_param_clicked = False
        self.template = []
        # self.load_widget.clearSelections()


    def update(self):
        self.region.setZValue(10)
        self.minX, self.maxX = self.region.getRegion()
        self.p1.setXRange(self.minX, self.maxX, padding=0)
    
    
    def updateRegion(self, window, viewRange):
        self.rgn = viewRange[0]
        self.region.setRegion(self.rgn)
        
    
    def acq_plot_clicked(self, item, points):
        if len(self.last_acq_point_clicked) > 0:
            self.last_acq_point_clicked[0].resetPen()
            self.last_acq_point_clicked[0].setSize(size=3)
        points[0].setPen('g', width=2)
        points[0].setSize(size=12)
        # print(points[0].pos())
        self.last_acq_point_clicked = points
    
    
    def mini_clicked(self, item):
        self.mini_number.setValue(self.sort_index.index(int(item.name())))
    
    
    def mini_spinbox(self, h):
        if h in self.mini_spinbox_list:
            self.last_mini_point_clicked = []
            if self.last_mini_clicked_1:
                self.last_mini_clicked_1.setPen(color='g')
                self.last_mini_clicked_2.setPen(color='g')
            self.mini_view_widget.clear()
            mini_index = self.sort_index[h]
            mini = self.acq_object.postsynaptic_events[mini_index]
            mini_item = pg.PlotDataItem(
                x=mini.x_array/mini.s_r_c,
                y=mini.event_array, pen=pg.mkPen(linewidth=3), symbol='o',
                symbolPen=None, symbolBrush='w', symbolSize=6)
            mini_plot_items = pg.PlotDataItem(x=mini.mini_comp_x,
                y=mini.mini_comp_y, pen=None, symbol='o', symbolBrush='g',
                symbolSize=12)
            self.mini_view_widget.addItem(mini_item)
            self.mini_view_widget.addItem(mini_plot_items)
            if mini.fit_tau is not np.nan and self.curve_fit_decay.isChecked():
                mini_decay_items = pg.PlotDataItem(x=mini.fit_decay_x,
                    y=mini.fit_decay_y, pen=pg.mkPen((255,0,255, 175),
                                                     width=3))
                self.mini_view_widget.addItem(mini_decay_items)
            mini_item.sigPointsClicked.connect(self.mini_plot_clicked)
            self.p2.listDataItems()[mini_index+1].setPen(color='m', width=2)
            self.p1.listDataItems()[mini_index+1].setPen(color='m', width=2)
            self.last_mini_clicked_2 = self.p2.listDataItems()[mini_index+1]
            self.last_mini_clicked_1 = self.p1.listDataItems()[mini_index+1]
            self.mini_amplitude.setText(str(
                self.round_sig(mini.amplitude, sig=4)))
            self.mini_tau.setText(str(
                self.round_sig(mini.final_tau_x, sig=4)))
            self.mini_rise_time.setText(str(
                self.round_sig(mini.rise_time, sig=4)))
            self.mini_rise_rate.setText(str(
                self.round_sig(mini.rise_rate, sig=4)))
            self.mini_baseline.setText(str(
                self.round_sig(mini.event_start_y, sig=4)))
        else:
            self.mini_view_widget.clear()
            self.mini_amplitude.setText('')
            self.mini_tau.setText('')
            self.mini_rise_time.setText('')
            self.mini_rise_rate.setText('')
            self.mini_baseline.setText('')
            pass
    
    
    def mini_plot_clicked(self, item, points):
        if self.last_mini_point_clicked:
            self.last_mini_point_clicked[0].resetPen()
            self.last_mini_point_clicked = []
            # self.last_mini_point.clicked[0].resetBrush()
        points[0].setPen('m', width=4)
        # points[0].setBrush('m')
        self.last_mini_point_clicked = points

    
    def set_point_as_peak(self):
        '''
        This will set the mini peak as the point selected on the mini plot and
        update the other two acquisition plots.

        Returns
        -------
        None.

        '''
        x = (self.last_mini_point_clicked[0].pos()[0]
             *self.acq_object.s_r_c)
        y = self.last_mini_point_clicked[0].pos()[1]
        mini_index = self.sort_index[int(self.mini_number.text())]
        self.acq_object.postsynaptic_events[
                mini_index].change_amplitude(x, y)
        self.last_mini_clicked_1.setData(
            x=self.acq_object.postsynaptic_events[
            mini_index].mini_plot_x,
            y=self.acq_object.postsynaptic_events[mini_index].mini_plot_y,
            color='m', width=2)
        self.last_mini_clicked_2.setData(
            x=self.acq_object.postsynaptic_events[
            mini_index].mini_plot_x,
            y=self.acq_object.postsynaptic_events[mini_index].mini_plot_y, 
            color='m', width=2)
        self.mini_spinbox(int(self.mini_number.text()))
        # self.last_mini_point_clicked[0].resetPen()
        self.last_mini_point_clicked = []
    
    
    def set_point_as_baseline(self):
        '''
        This will set the baseline as the point selected on the mini plot and
        update the other two acquisition plots.

        Returns
        -------
        None.

        '''
        if self.last_mini_point_clicked:
            x = (self.last_mini_point_clicked[0].pos()[0]
                 *self.acq_object.s_r_c)
            y = self.last_mini_point_clicked[0].pos()[1]
            mini_index = self.sort_index[int(self.mini_number.text())]
            self.acq_object.postsynaptic_events[
                    mini_index].change_baseline(x, y)
            self.last_mini_clicked_1.setData(
                x=self.acq_object.postsynaptic_events[
                mini_index].mini_plot_x,
                y=self.acq_object.postsynaptic_events[mini_index].mini_plot_y, 
                color='m', width=2)
            self.last_mini_clicked_2.setData(
                x=self.acq_object.postsynaptic_events[
                mini_index].mini_plot_x,
                y=self.acq_object.postsynaptic_events[mini_index].mini_plot_y, 
                color='m', width=2)
            self.mini_spinbox(int(self.mini_number.text()))
            # self.last_mini_point_clicked[0].resetPen()
            self.last_mini_point_clicked = []
        else:
            pass
        
        
    def delete_mini(self):
        # self.last_mini_deleted = \
        #     self.acq_object.postsynaptic_events[int(self.mini_number.text())]
        self.last_mini_deleted_number = self.mini_number.text()
        self.mini_view_widget.clear()
        mini_index = self.sort_index[int(self.mini_number.text())]
        # mini_index = self.sort_index.index(
        #             int(self.mini_number.text()))
        self.p1.removeItem(self.p1.listDataItems()[mini_index + 1])
        self.p2.removeItem(self.p2.listDataItems()[mini_index + 1])
        del self.acq_dict[str(self.acquisition_number.text())
                          ].postsynaptic_events[mini_index]
        del self.acq_dict[str(self.acquisition_number.text())
                          ].final_events[mini_index]
        self.sort_index = list(np.argsort(self.acq_object.final_events))
        self.mini_spinbox_list = list(
                    range(len(self.acq_object.postsynaptic_events)))
        for num, i, j in zip(self.mini_spinbox_list,
                    self.p1.listDataItems()[1:], self.p2.listDataItems()[1:]):
            i.opts['name'] = num
            j.opts['name'] = num
        self.last_mini_clicked_1 = []
        self.last_mini_clicked_2 = []
        self.mini_number.setMaximum(self.mini_spinbox_list[-1])
        self.mini_number.setValue(int(self.mini_number.text()))
        self.mini_spinbox(int(self.mini_number.text()))
        self.minis_deleted += 1
    
    
    def create_mini(self):
        if self.last_acq_point_clicked:
            x = (self.last_acq_point_clicked[0].pos()[0]
                 * self.acq_object.s_r_c)
            self.acq_dict[str(
                self.acquisition_number.text())].create_new_mini(x)
            self.mini_spinbox_list = list(
                    range(len(self.acq_object.postsynaptic_events)))
            self.sort_index = list(np.argsort(self.acq_object.final_events))
            id_value = self.mini_spinbox_list[-1]
            mini_plot = pg.PlotCurveItem(
                x=self.acq_object.postsynaptic_events[id_value].mini_plot_x, 
                y=self.acq_object.postsynaptic_events[id_value].mini_plot_y,
                pen='g', name = id_value, clickable=True)
            mini_plot.sigClicked.connect(self.mini_clicked)
            self.p1.addItem(mini_plot)
            self.p2.plot(
                x=self.acq_object.postsynaptic_events[id_value].mini_plot_x,
                y=self.acq_object.postsynaptic_events[id_value].mini_plot_y,
                pen='g', name = id_value)
            self.mini_number.setMaximum(self.mini_spinbox_list[-1])
            self.mini_number.setValue(self.sort_index.index(id_value))
            self.last_acq_point_clicked[0].resetPen()
            self.last_acq_point_clicked = []
        else:
            pass
        
    
    def delete_acq(self):
        self.deleted_acqs[str(
            self.acquisition_number.text())] = self.acq_dict[str(
                self.acquisition_number.text())]
        self.recent_reject_acq[str(
            self.acquisition_number.text())] = self.acq_dict[str(
                self.acquisition_number.text())]
        del self.acq_dict[str(self.acquisition_number.text())]
        self.p1.clear()
        self.p2.clear()
        self.mini_view_widget.clear()
        self.analysis_list = [int(i) for i in self.acq_dict.keys()]
        self.acq_spinbox(int(self.acquisition_number.text())+1)
        self.acqs_deleted += 1
        
    
    def reset_rejected_acqs(self):
        self.acq_dict.update(self.deleted_acqs)
        self.analysis_list = [int(i) for i in self.acq_dict.keys()]
        self.deleted_acqs = {}
        self.recent_reject_acq = {}
        self.acqs_deleted = 0
 
    
    def reset_recent_reject_acq(self):
        self.acq_dict.update(self.recent_reject_acq)
        self.analysis_list = [int(i) for i in self.acq_dict.keys()]
        self.acqs_deleted -= 1
        
    
    def final_analysis(self):
        self.calculate_parameters.setEnabled(False)
        self.calc_param_clicked = True
        self.pbar.setFormat('Analyzing...')
        self.final_obj = FinalMiniAnalysis(self.acq_dict,
                                              self.minis_deleted,
                                              self.acqs_deleted)
        self.ave_mini_plot.clear()
        self.ave_mini_plot.plot(x=self.final_obj.average_mini_x,
                                y=self.final_obj.average_mini)
        self.ave_mini_plot.plot(x=self.final_obj.decay_x,
                                    y=self.final_obj.fit_decay_y, pen='g')
        self.raw_data_table.setData(
            self.final_obj.raw_df.T.to_dict('dict'))
        self.final_table.setData(self.final_obj.final_df.T.to_dict('dict'))
        plots = ['Amplitude (pA)', 'Est tau (ms)', 'Rise time (ms)',
                 'Rise rate (pA/ms)', 'IEI (ms)']
        self.plot_selector.addItems(plots)
        if self.plot_selector.currentText() != 'IEI (ms)':
            self.mw.plot(x='Real time', y=self.plot_selector.currentText(),
                         df=self.final_obj.raw_df)
        self.amp_dist.plot(self.final_obj.raw_df,
                           self.plot_selector.currentText())
        
        self.pbar.setFormat('Finished analysis')
        self.calculate_parameters.setEnabled(True)
    
    
    def plot_raw_data(self, column):
        if column != 'IEI (ms)':
            self.mw.plot(x='Real time', y=column, df=self.final_obj.raw_df)
        self.amp_dist.plot(self.final_obj.raw_df, column)
    
    
    def round_sig(self, x, sig=2):
        if np.isnan(x):
            return np.nan
        elif x == 0:
            return 0
        elif x != 0 or not np.isnan(x):
            if np.isnan(floor(log10(abs(x)))):
                return round(x, 0)
            else:
                return round(x, sig-int(floor(log10(abs(x))))-1)
    
    
    def file_does_not_exist(self):
        self.dlg = QMessageBox(self)
        self.dlg.setWindowTitle('Error')
        self.dlg.setText('File does not exist')
        self.dlg.exec()
  
    
    def open_files(self):
        self.reset()
        self.pbar.setFormat('Loading...')
        load_dict = YamlWorker.load_yaml()
        self.set_preferences(load_dict)
        self.calculate_parameters.setEnabled(
            load_dict['buttons']['calculate_parameters'])
        self.analyze_acq_button.setEnabled(
            load_dict['buttons']['analyze_acq_button'])
        self.acquisition_number.setValue(load_dict['Acq_number'])
        self.acq_spinbox(int(load_dict['Acq_number']))
        file_list = glob('*.json')
        if not file_list:
            self.file_list = None
            pass
        else:
            for i in range(len(file_list)):
                with open(file_list[i]) as file:
                    data = json.load(file)
                    x = LoadMiniAnalysis(data)
                    self.acq_dict[str(x.acq_number)] = x
                    self.pbar.setValue(
                        int(((i+1)/len(file_list))*100))
            self.analysis_list = [int(i) for i in self.acq_dict.keys()]
            self.acquisition_number.setMaximum(max(self.analysis_list))
            self.acquisition_number.setMinimum(min(self.analysis_list)) 
            if load_dict['Final Analysis']:
                excel_file = glob('*.xlsx')[0]
                save_values = pd.read_excel(excel_file, sheet_name=None)
                self.final_obj = LoadMiniSaveData(save_values)
                self.ave_mini_plot.clear()
                self.ave_mini_plot.plot(x=self.final_obj.average_mini_x,
                                    y=self.final_obj.average_mini)
                self.ave_mini_plot.plot(x=self.final_obj.decay_x,
                                        y=self.final_obj.fit_decay_y, pen='g')
                self.raw_data_table.setData(
                    self.final_obj.raw_df.T.to_dict('dict'))
                self.final_table.setData(
                    self.final_obj.final_df.T.to_dict('dict'))
                plots = ['Amplitude (pA)', 'Est tau (ms)', 'Rise time (ms)',
                         'Rise rate (pA/ms)', 'IEI (ms)']
                self.plot_selector.addItems(plots)
            self.pbar.setFormat('Loaded')
                   
    
    def save_as(self, save_filename):
        self.pbar.setFormat('Saving...')
        self.pbar.setValue(0)
        self.create_pref_dict()
        self.pref_dict['Final Analysis'] = self.calc_param_clicked
        self.pref_dict['Acq_number'] = self.acquisition_number.value()
        self.pref_dict['Deleted Acqs'] = list(self.deleted_acqs.keys())
        YamlWorker.save_yaml(self.pref_dict, save_filename)
        if self.pref_dict['Final Analysis']:
            self.final_obj.save_data(save_filename)
        self.worker = MiniSaveWorker(save_filename, self.acq_dict)
        self.worker.signals.progress.connect(self.update_save_progress)
        self.worker.signals.finished.connect(self.progress_finished)
        self.threadpool.start(self.worker)
    
    
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
        
    
    def update_save_progress(self, progress):
        self.pbar.setValue(progress)
    
    
    def progress_finished(self, finished):
        self.pbar.setFormat(finished)


if __name__ == '__main__':
    miniAnalysisWidget()