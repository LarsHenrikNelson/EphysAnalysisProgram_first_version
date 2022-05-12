from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QComboBox, QWidget,
                             QLabel, QFormLayout, QSpinBox, QApplication,
                             QVBoxLayout, QTabWidget)
from PyQt5 import QtGui
from PyQt5.QtCore import (QSize)
import pyqtgraph as pg
import qdarkstyle


class PreferencesWidget(QWidget):
    def __init__(self):
        self.initUI() 

    def initUi(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.tabs_widget = QTabWidget()
        self.layout.addWidget(self.tabs_widget)
        
        self.test_button = QPushButton('Test button')
        self.layout.addWidget(self.test_button)
        self.test_button.clicked.connect(self.click_me)

        self.tab1 = QWidget()
        self.tabs_widget.addTab(self.tab1)


    def click_me(self):
        print(self.parent())


    def style_choice(self, text):
        if text == 'Dark style':
            stylesheet = qdarkstyle.load_stylesheet(qdarkstyle.dark.palette.DarkPalette)
        else:
            stylesheet = qdarkstyle.load_stylesheet(qdarkstyle.light.palette.LightPalette)
        app = QApplication.instance()
        app.setStyle(stylesheet)