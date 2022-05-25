#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 08:22:14 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: larsnelson
"""
import os
from os.path import expanduser
from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QWidget,
    QLabel,
    QFormLayout,
    QApplication,
    QSpinBox,
    QToolBar,
    QAction,
    QStackedWidget,
)
import qdarkstyle

from currentClampWidget import currentClampWidget
from filterWidget import filterWidget
from miniAnalysisWidget import MiniAnalysisWidget
from oEPSCWidget import oEPSCWidget
from PreferencesWidget import PreferencesWidget
from utility_classes import LineEdit, YamlWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.set_widget("Mini Analysis")

    def initUI(self):
        self.setWindowTitle("Electrophysiology Analysis")

        # Set the menu bar
        self.bar = self.menuBar()
        self.file_menu = self.bar.addMenu("File")
        self.preferences_menu = self.bar.addMenu("Preferences")

        self.openFile = QAction("Open", self)
        self.openFile.setStatusTip("Open file")
        self.openFile.setShortcut("Ctrl+O")
        self.openFile.triggered.connect(self.open_files)
        self.file_menu.addAction(self.openFile)

        self.saveFile = QAction("Save", self)
        self.saveFile.setStatusTip("Save file")
        self.saveFile.setShortcut("Ctrl+S")
        self.saveFile.triggered.connect(self.save_as)
        self.file_menu.addAction(self.saveFile)

        self.loadPref = QAction("Load analysis preferences", self)
        self.loadPref.setStatusTip("Load analysis preferences")
        self.loadPref.triggered.connect(self.load_preferences)
        self.file_menu.addAction(self.loadPref)

        self.savePref = QAction("Save analysis preferences", self)
        self.savePref.setStatusTip("Save analysis preferences")
        self.savePref.triggered.connect(self.save_preferences)
        self.file_menu.addAction(self.savePref)

        self.setApplicationPreferences = QAction("Set preferences", self)
        self.setApplicationPreferences.triggered.connect(self.set_appearance)
        self.preferences_menu.addAction(self.setApplicationPreferences)

        self.tool_bar = QToolBar()
        self.addToolBar(self.tool_bar)

        self.widget_chooser = QComboBox()
        self.tool_bar.addWidget(self.widget_chooser)
        self.widget_chooser.addItems(
            ["Mini Analysis", "oEPSC", "Current Clamp", "Filtering setup"]
        )
        self.widget_chooser.currentTextChanged.connect(self.set_widget)

        self.path_edit = LineEdit(expanduser("~"))
        self.path_edit.setEnabled(False)
        self.tool_bar.addWidget(self.path_edit)

        self.button = QPushButton("Set Path")
        self.button.clicked.connect(self.set_path)
        self.tool_bar.addWidget(self.button)

        self.preferences_widget = PreferencesWidget()

        self.directory = str(os.chdir(expanduser("~")))

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.mini_widget = MiniAnalysisWidget()
        self.central_widget.addWidget(self.mini_widget)
        self.oepsc_widget = oEPSCWidget()
        self.central_widget.addWidget(self.oepsc_widget)
        self.current_clamp_widget = currentClampWidget()
        self.central_widget.addWidget(self.current_clamp_widget)
        self.filter_widget = filterWidget()
        self.central_widget.addWidget(self.filter_widget)

    def set_widget(self, text):
        if text == "Mini Analysis":
            self.central_widget.setCurrentWidget(self.mini_widget)
        elif text == "oEPSC":
            self.central_widget.setCurrentWidget(self.oepsc_widget)
        elif text == "Current Clamp":
            self.central_widget.setCurrentWidget(self.current_clamp_widget)
        elif text == "Filtering setup":
            self.central_widget.setCurrentWidget(self.filter_widget)

    def set_path(self, click):
        self.directory = str(QFileDialog.getExistingDirectory())
        if len(self.directory) == 0:
            os.chdir(expanduser("~"))
        else:
            self.path_edit.setText("{}".format(self.directory))
            os.chdir(self.directory)

    def save_as(self):
        save_filename, _extension = QFileDialog.getSaveFileName(
            self, "Save data as...", f"{self.directory}/save_filename"
        )
        if save_filename:
            self.central_widget.save_as(save_filename)

    def open_files(self):
        self.directory = str(QFileDialog.getExistingDirectory())
        if len(self.directory) == 0:
            # This prevents an error from showing up when the path is not
            # selected
            pass
        else:
            self.path_edit.setText("{}".format(self.directory))
            os.chdir(self.directory)
            self.central_widget.open_files()

    def load_preferences(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if len(file_name) == 0:
            # This prevents an error from showing up when the path is not
            # selected
            pass
        else:
            self.central_widget.load_preferences(file_name)

    def save_preferences(self):
        save_filename, _extension = QFileDialog.getSaveFileName(
            self, "Save data as...", ""
        )
        # print(save_filename)
        if save_filename:
            self.central_widget.save_preferences(save_filename)

    def set_appearance(self):
        # Creates a separate window to set the appearance of the application
        self.preferences_widget.show()

    def startup_function(self):
        p = Path.home()
        h = "EphysAnalysisProgram"
        file_name = "Preferences.yaml"

        if Path(p / h).exists():
            if Path(p / h / file_name).exists():
                pref_dict = YamlWorker.load_yaml(p / h / file_name)
            else:
                pass
        else:
            os.mkdir(p / h)


def run_program():
    app = QApplication([])
    dark_stylesheet = qdarkstyle.load_stylesheet()
    app.setStyleSheet(dark_stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_program()
