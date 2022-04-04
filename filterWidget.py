#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:20:28 2022

@author: Lars
"""
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QComboBox, QWidget,
                             QLabel, QFormLayout, QSpinBox,)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import (QSize)
import pyqtgraph as pg


from acq_class import Acquisition
from utility_classes import (LineEdit)

class filterWidget(QWidget):
    '''
    This is how to subclass a widget.
    '''
    
    
    def __init__(self, parent=None):
        
        super(filterWidget, self).__init__(parent)
        
        self.parent = parent
    
        # self.path_layout = QHBoxLayout()
        self.plot_layout = QHBoxLayout()
        self.filt_layout = QFormLayout()
        
        #Since the class is inheriting from QWdiget there is no need to set
        #or define a central widget like the mainwindow setCentralWidget
        self.setLayout(self.plot_layout)
        self.plot_layout.addLayout(self.filt_layout, 0)
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.p1 = self.plot_widget.addPlot(row=0, col=0)
        pg.setConfigOptions(antialias=True)
        self.plot_layout.addWidget(self.plot_widget, 1)
        self.p1.setMinimumWidth(500)
        
        self.v1 = self.plot_widget.addViewBox(row=0, col=1)
        self.v1.setMaximumWidth(300)
        self.v1.setMinimumWidth(300)
        
        self.legend = pg.LegendItem()
        self.v1.addItem(self.legend)
        self.legend.setParentItem(self.v1)
        self.legend.anchor((0,0), (0,0))
        
        self.acq_id_label = QLabel('Acq ID')
        self.acq_id_edit = LineEdit()
        self.acq_id_edit.setEnabled(True)
        self.filt_layout.addRow(self.acq_id_label, self.acq_id_edit)
        
        self.acq_number_label = QLabel('Acq #')
        self.acquisition_number = QSpinBox()
        self.acquisition_number.setMaximum(400)
        self.acquisition_number.setMinimum(1)
        self.acquisition_number.valueChanged.connect(self.spinbox)
        self.filt_layout.addRow(self.acq_number_label,
                                self.acquisition_number)

        self.b_start_label = QLabel('Baseline start')
        self.b_start_edit = LineEdit()
        self.b_start_edit.setEnabled(True)
        self.b_start_edit.setText('0')
        self.filt_layout.addRow(self.b_start_label, self.b_start_edit)
        
        self.b_end_label = QLabel('Baseline end')
        self.b_end_edit = LineEdit()
        self.b_end_edit.setEnabled(True)
        self.b_end_edit.setText('80')
        self.filt_layout.addRow(self.b_end_label, self.b_end_edit)
        
        self.sample_rate_label = QLabel('Sample rate')
        self.sample_rate_edit = LineEdit()
        self.sample_rate_edit.setEnabled(True)
        self.sample_rate_edit.setText('10000')
        self.filt_layout.addRow(self.sample_rate_label, self.sample_rate_edit)

        
        self.filter_type_label = QLabel('Filter Type')
        
        filters = ['None', 'bessel', 'fir_zero_1', 'fir_zero_2', 'savgol',
                   'median', 'remez_1', 'remez_2', 'subtractive']
        
        self.filter_selection= QComboBox(self)
        self.filter_selection.addItems(filters)
        self.filt_layout.addRow(self.filter_type_label, self.filter_selection)
        
        self.order_label = QLabel('Order')
        self.order_edit = LineEdit()
        self.order_edit.setValidator(QIntValidator())
        self.order_edit.setEnabled(True)
        self.filt_layout.addRow(self.order_label, self.order_edit)
        
        self.high_pass_label = QLabel('High-pass')
        self.high_pass_edit = LineEdit()
        self.high_pass_edit.setValidator(QIntValidator())
        self.high_pass_edit.setEnabled(True)
        self.filt_layout.addRow(self.high_pass_label, self.high_pass_edit)
        
        self.high_width_label = QLabel('High-width')
        self.high_width_edit = LineEdit()
        self.high_width_edit.setValidator(QIntValidator())
        self.high_width_edit.setEnabled(True)
        self.filt_layout.addRow(self.high_width_label, self.high_width_edit)
        
        self.low_pass_label = QLabel('Low-pass')
        self.low_pass_edit = LineEdit()
        self.low_pass_edit.setValidator(QIntValidator())
        self.low_pass_edit.setEnabled(True)
        self.filt_layout.addRow(self.low_pass_label, self.low_pass_edit)
        
        self.low_width_label = QLabel('Low-width')
        self.low_width_edit = LineEdit()
        self.low_width_edit.setValidator(QIntValidator())
        self.low_width_edit.setEnabled(True)
        self.filt_layout.addRow(self.low_width_label, self.low_width_edit)
        
        self.window_label = QLabel('Window type')
        self.window_edit = LineEdit()
        self.window_edit.setEnabled(True)
        self.filt_layout.addRow(self.window_label, self.window_edit)
        
        self.polyorder_label = QLabel('Polyorder')
        self.polyorder_edit = LineEdit()
        self.polyorder_edit.setValidator(QIntValidator())
        self.polyorder_edit.setEnabled(True)
        self.filt_layout.addRow(self.polyorder_label, self.polyorder_edit)
        
        #Plot acquisition button
        # self.plot_acq = QPushButton('Plot acq')
        # self.plot_acq.clicked.connect(self.plot_acq_button)
        # self.plot_acq.setMaximumSize(QSize(300,25))
        # self.filt_layout.addRow(self.plot_acq)
        
        self.plot_filt = QPushButton('Plot acq')
        self.plot_filt.clicked.connect(self.plot_filt_button)
        self.plot_filt.setMaximumSize(QSize(300,25))
        self.filt_layout.addRow(self.plot_filt)
        
        self.clear_plot = QPushButton('Clear plot')
        self.clear_plot.clicked.connect(self.clear_plot_button)
        self.clear_plot.setMaximumSize(QSize(300,25))
        self.filt_layout.addRow(self.clear_plot)
        
        # self.color_list = ['b', 'g', 'r', 'c', 'm', 'y']
        self.plot_list = {}
        self.pencil_list = []
        self.counter = 0

    
    def plot_filt_button(self):
        h = Acquisition(self.acq_id_edit.toText(), 
                             self.acquisition_number.text(), 
                             self.sample_rate_edit.toInt(), 
                             self.b_start_edit.toInt(), 
                             self.b_end_edit.toInt(), 
                             self.filter_selection.currentText(), 
                             self.order_edit.toInt(), 
                             self.high_pass_edit.toInt(), 
                             self.high_width_edit.toInt(), 
                             self.low_pass_edit.toInt(), 
                             self.low_width_edit.toInt(), 
                             self.window_edit.toText(), 
                             self.polyorder_edit.toInt())  
        h.filter_array()
        if len(self.plot_list.keys()) == 0:
            pencil=pg.mkPen(color='w', alpha=int(0.75*255))
        else:
            pencil = pg.mkPen(color=pg.intColor(self.counter))
        plot_item = self.p1.plot(x=h.x_array, 
                              y = h.filtered_array,
                              pen=pencil,
                              name = (self.filter_selection.currentText()
                                      + '_' + str(self.counter)))
        self.legend.addItem(plot_item,  self.filter_selection.currentText()
                            + '_' + str(self.counter))
        self.plot_list[str(self.counter)] = h
        self.counter += 1
        self.pencil_list += [pencil]

    
    def spinbox(self, h):
        if len(self.plot_list.keys()) > 1:
            self.p1.clear()
            for i, j in zip(self.plot_list.keys(), self.pencil_list):
                h = Acquisition(self.plot_list[i].prefix, 
                             self.acquisition_number.text(), 
                             self.plot_list[i].sample_rate, 
                             self.plot_list[i].baseline_start, 
                             self.plot_list[i].baseline_end, 
                             self.plot_list[i].filter_type, 
                             self.plot_list[i].order, 
                             self.plot_list[i].high_pass, 
                             self.plot_list[i].high_width, 
                             self.plot_list[i].low_pass, 
                             self.plot_list[i].low_width, 
                             self.plot_list[i].window, 
                             self.plot_list[i].polyorder)
                self.h.filter_array()
                self.p1.plot(x=h.x_array, y = h.filtered_array, pen=j)
            # self.plot_list[str(self.counter)] = h
            
        elif len(self.plot_list.keys()) == 1:
            self.p1.clear()
            h = Acquisition(self.plot_list['0'].prefix, 
                             self.acquisition_number.text(),
                             self.plot_list['0'].sample_rate, 
                             self.plot_list['0'].baseline_start, 
                             self.plot_list['0'].baseline_end, 
                             self.plot_list['0'].filter_type, 
                             self.plot_list['0'].order, 
                             self.plot_list['0'].high_pass, 
                             self.plot_list['0'].high_width, 
                             self.plot_list['0'].low_pass, 
                             self.plot_list['0'].low_width, 
                             self.plot_list['0'].window,
                             self.plot_list['0'].polyorder)
            self.h.filter_array()
            self.p1.plot(x=h.x_array, y = h.filtered_array,      
                                     pen = self.pencil_list[0]) 
    
    
    def clear_plot_button(self):
         self.p1.clear()
         self.legend.clear()
         self.counter = 0
         self.plot_list = {}
         self.pencil_list = []    
         


if __name__ == '__main__':
    filterWidget()