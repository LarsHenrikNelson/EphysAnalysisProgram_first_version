#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 08:22:14 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: larsnelson
"""
import os
from os.path import expanduser
import sys

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QComboBox,
                             QFileDialog, QMainWindow, QWidget, QLabel, 
                             QFormLayout, QApplication, QSpinBox,
                             QToolBar, QAction)
import qdarkstyle

from currentClampWidget import currentClampWidget
from miniAnalysisWidget import miniAnalysisWidget
from oEPSCWidget import oEPSCWidget
from filterWidget import filterWidget
from utility_classes import LineEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.set_widget('Mini Analysis')
    
    def initUI(self):
        self.setWindowTitle('Electrophysiology Analysis')

        #Set the menu bar
        self.bar = self.menuBar()
        self.file_menu = self.bar.addMenu('File')
        # self.prefences_menu = self.bar.addMenu('Preferences')        
        
        self.openFile = QAction('Open', self)
        self.openFile.setStatusTip('Open file')
        self.openFile.setShortcut('Ctrl+O')
        self.openFile.triggered.connect(self.open_files)
        self.file_menu.addAction(self.openFile)
        
        self.saveFile = QAction('Save', self)
        self.saveFile.setStatusTip('Save file')
        self.saveFile.setShortcut('Ctrl+S')
        self.saveFile.triggered.connect(self.save_as)
        self.file_menu.addAction(self.saveFile)
        
        self.loadPref = QAction('Load preferences', self)
        self.loadPref.setStatusTip('Load preferences')
        self.loadPref.triggered.connect(self.load_preferences)
        self.file_menu.addAction(self.loadPref)
        
        self.savePref = QAction('Save preferences', self)
        self.savePref.setStatusTip('Save preferences')
        self.savePref.triggered.connect(self.save_preferences)
        self.file_menu.addAction(self.savePref)
        
        self.tool_bar = QToolBar()
        self.addToolBar(self.tool_bar)
   
        self.widget_chooser = QComboBox()
        self.tool_bar.addWidget(self.widget_chooser)
        self.widget_chooser.addItems(['Mini Analysis', 'oEPSC', 'Current Clamp',
                                     'Filtering setup'])
        self.widget_chooser.currentTextChanged.connect(self.set_widget)

        self.path_edit = LineEdit(expanduser("~"))
        self.path_edit.setEnabled(False)
        self.tool_bar.addWidget(self.path_edit)
        
        self.button = QPushButton('Set Path')
        self.button.clicked.connect(self.set_path)
        self.tool_bar.addWidget(self.button)
        
        self.directory = str(os.chdir(expanduser("~")))
    
    
    def set_widget(self, text):
        if text == 'Mini Analysis':
            self.central_widget = miniAnalysisWidget()
        elif text == 'oEPSC':
            self.central_widget = oEPSCWidget()
        elif text == 'Current Clamp':
            self.central_widget = currentClampWidget()
        elif text == 'Filtering setup':
            self.central_widget = filterWidget()
        self.setCentralWidget(self.central_widget)


    def set_path(self, click):
       self.directory = str(QFileDialog.getExistingDirectory())
       if len(self.directory) == 0:
          os.chdir(expanduser("~")) 
       else:
           self.path_edit.setText('{}'.format(self.directory))
           os.chdir(self.directory)    


    def save_as(self):
        save_filename, _extension = QFileDialog.getSaveFileName(
            self, 'Save data as...', 
            f'{self.directory}/save_filename')
        if save_filename:
            self.central_widget.save_as(save_filename)


    def open_files(self):
        self.directory = str(QFileDialog.getExistingDirectory())
        if len(self.directory) == 0:
            #This prevents an error from showing up when the path is not
            #selected
            pass 
        else:
            self.path_edit.setText('{}'.format(self.directory))
            os.chdir(self.directory)  
            self.central_widget.open_files()
    
    
    def load_preferences(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if len(file_name) == 0:
            #This prevents an error from showing up when the path is not
            #selected
            pass 
        else:
            self.central_widget.load_preferences(file_name)
    
    
    def save_preferences(self):
        save_filename, _extension = QFileDialog.getSaveFileName(
            self, 'Save data as...', 
            '')
        # print(save_filename)
        if save_filename:
            self.central_widget.save_preferences(save_filename)
                    


def run_program():
    os.chdir(expanduser("~"))
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    run_program()