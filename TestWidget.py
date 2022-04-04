# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:15:29 2022

@author: LarsNelson
"""
import os
from os.path import expanduser
import sys

from PyQt5.QtWidgets import (QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
                             QWidget, QLabel, QFormLayout, QComboBox, 
                             QSpinBox, QCheckBox,  QProgressBar, QMessageBox,
                             QTabWidget, QMainWindow, QApplication)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThreadPool
from PyQt5 import QtCore
import pyqtgraph as pg
import qdarkstyle

from acq_class import MiniAnalysis, LoadMiniAnalysis
from final_analysis_classes import FinalMiniAnalysis
from load_classes import LoadMiniSaveData
from utility_classes import (LineEdit, MiniSaveWorker, MplWidget,
                             DistributionPlot, YamlWorker)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    
    def initUI(self):
        #Tab 1 layouts
        
        
        self.tab1 = QWidget()  

        self.setCentralWidget(self.tab1)
                              
        self.tab1_layout = QVBoxLayout()
        self.setup_layout = QHBoxLayout()
        self.input_layout = QFormLayout()
        self.settings_layout = QFormLayout()
        self.template_form = QFormLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab1_layout.addLayout(self.setup_layout, 0)
        self.setup_layout.addLayout(self.input_layout, 0)
        self.setup_layout.addLayout(self.settings_layout, 0)
        self.setup_layout.addLayout(self.template_form, 0)
        
        self.acq_id_label = QLabel('Acq ID')
        self.acq_id_edit = LineEdit()
        self.acq_id_edit.setObjectName('acq_id_edit')
        self.acq_id_edit.setEnabled(True)
        self.input_layout.addRow(self.acq_id_label, self.acq_id_edit)
    
        self.start_acq_label = QLabel('Start acq')
        self.start_acq_edit = LineEdit()
        self.start_acq_edit.setObjectName('start_acq_edit')
        self.start_acq_edit.setEnabled(True)
        self.input_layout.addRow(self.start_acq_label, self.start_acq_edit)
        
        self.end_acq_label = QLabel('End acq')
        self.end_acq_edit = LineEdit()
        self.end_acq_edit.setObjectName('end_acq_edit')
        self.end_acq_edit.setEnabled(True)
        self.input_layout.addRow(self.end_acq_label, self.end_acq_edit)
    
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
        self.rc_check_end_edit.setObjectName('rc_check_end_edit')
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
        self.analyze_acq_button.clicked.connect(self.analyze_children)
        
        self.calculate_parameters = QPushButton('Calculate Parameters')
        self.input_layout.addRow(self.calculate_parameters)
        self.calculate_parameters.setObjectName('calculate_parameters')
        # self.calculate_parameters.clicked.connect(self.final_analysis)
        self.calculate_parameters.setEnabled(False)
        
        self.reset_button = QPushButton('Reset Analysis')
        self.input_layout.addRow(self.reset_button)
        self.reset_button.setObjectName('reset_button')
        # self.reset_button.clicked.connect(self.reset)
        
        self.sensitivity_label = QLabel('Sensitivity')
        self.sensitivity_edit = LineEdit()
        self.sensitivity_edit.setObjectName('sensitivity_edit')
        self.sensitivity_edit.setEnabled(True)
        self.sensitivity_edit.setText('4')
        self.settings_layout.addRow(self.sensitivity_label,
                                 self.sensitivity_edit)
        
        # self.pbar = QProgressBar(self)
        # self.pbar.setValue(0)
        # self.tab1_layout.addWidget(self.pbar)
        
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
        self.min_rise_time.setText('1')
        self.settings_layout.addRow(self.min_rise_time_label,
                                    self.min_rise_time)
        
        self.min_rise_rate_label = QLabel('Min rise rate (pA/ms)')
        self.min_rise_rate = LineEdit()
        self.min_rise_rate.setObjectName('min_rise_rate')
        self.min_rise_rate.setEnabled(True)
        self.min_rise_rate.setText('3')
        self.settings_layout.addRow(self.min_rise_rate_label,
                                    self.min_rise_rate)
        
        self.max_decay_tau_label = QLabel('Max decay tau (ms)')
        self.max_decay_tau = LineEdit()
        self.max_decay_tau.setObjectName('max_decay_tau')
        self.max_decay_tau.setEnabled(True)
        self.max_decay_tau.setText('20')
        self.settings_layout.addRow(self.max_decay_tau_label,
                                    self.max_decay_tau)
        
        self.invert_label = QLabel('Invert')
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
        self.template_button.setObjectName('template_button')
        # self.template_button.clicked.connect(self.create_template)
        
        
    def analyze_children(self):
        save_dict = {}
        
        line_edits = self.tab1.findChildren(QLineEdit)
        line_edit_dict = {}
        for i in line_edits:
            line_edit_dict[i.objectName()] = i.text()
        save_dict['line_edits'] = line_edit_dict
        
        combo_box_dict = {}
        combo_boxes = self.tab1.findChildren(QComboBox)
        for i in combo_boxes:
            combo_box_dict[i.objectName()] = i.currentText()
        save_dict['combo_boxes'] = combo_box_dict
        
        check_box_dict = {}
        check_boxes = self.tab1.findChildren(QCheckBox)
        for i in check_boxes:
            check_box_dict[i.objectName()] = i.isChecked()
        save_dict['check_boxes'] = check_box_dict  
           
        buttons_dict = {}
        buttons = self.tab1.findChildren(QPushButton)
        for i in buttons:
            buttons_dict[i.objectName()] = i.isEnabled()
        save_dict['buttons'] = buttons_dict
        
        YamlWorker.save_yaml(save_dict, 'test')
        
        
def run_program():
    # os.chdir(expanduser("~"))
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    window = MainWindow()
    window.show()
    app.exec()
    

if __name__ == '__main__':
    run_program()