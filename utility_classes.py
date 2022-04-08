# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:29:22 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: LarsNelson
"""
from copy import deepcopy
from glob import glob
import json

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import (QLineEdit, QSizePolicy, QWidget, QVBoxLayout)
from PyQt5.QtCore import (QRunnable, pyqtSlot, QObject,
                          pyqtSignal)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import yaml


class NumpyEncoder(json.JSONEncoder):
    '''
    Special json encoder for numpy types. Numpy types are not accepted by the
    json encoder and need to be converted to python types.
    '''
    
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)      


class LineEdit(QLineEdit):
    '''
    This is a subclass of QLineEdit that returns values that are usable for
    Python.
    '''
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
        
    def toInt(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = int(self.text())
        return x


    def toText(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = self.text()
        return x
    
    
    def toFloat(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = float(self.text())
        return x
    
    
class SaveWorker(QRunnable):
    '''
    This class is used to create a 'runner' in a different thread than the
    main GUI. This prevents that GUI from freezing during saving.
    '''
    def __init__(self, save_filename, dictionary):
        super().__init__()
        
        self.save_filename = save_filename
        self.dictionary = dictionary
        self.signals = WorkerSignals()
        
    @pyqtSlot()
    def run(self):
        for i, key in enumerate(self.dictionary.keys()):
            x = self.dictionary[key]
            with open(f"{self.save_filename}_{x.name}.json", "w") as write_file:
                json.dump(x.__dict__, write_file, cls=NumpyEncoder)
            self.signals.progress.emit(
                int((100*(i+1)/len(self.dictionary.keys()))))
        self.signals.finished.emit('Saved')
        

class MiniSaveWorker(QRunnable):
    '''
    This class is used to create a 'runner' in a different thread than the
    main GUI. This prevents that GUI from freezing during saving. This is a
    variant of the SaveWorker class used for the MiniAnalysisWidgit since a
    specific function needs to be run on the mini-dictionary to prevent it
    from taking up a lot of space in the json file.
    '''
    def __init__(self, save_filename, dictionary):
        super().__init__()
        
        self.save_filename = save_filename
        self.dictionary = dictionary
        self.signals = WorkerSignals()
        
    @pyqtSlot()
    def run(self):
        for i, key in enumerate(self.dictionary.keys()):
            x = deepcopy(self.dictionary[key])
            x.save_postsynaptic_events()
            with open(f"{self.save_filename}_{x.name}.json", "w") as write_file:
                json.dump(x.__dict__, write_file, cls=NumpyEncoder)
            self.signals.progress.emit(
                int((100*(i+1)/len(self.dictionary.keys()))))
        self.signals.finished.emit('Saved')


class WorkerSignals(QObject):
    '''
    This is general 'worker' that provides feedback from the 'runner' to the
    main GUI thread to prevent it from freezing.
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)


class StemPlotCanvas(FigureCanvasQTAgg):
    '''
    Creating a matplotlib window this way 
    '''
    
    def __init__(self, parent=None, width=3, height=2, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # self.plot()


    def plot(self, x, y, df):
        ax = self.figure.add_subplot(111)
        ax.stemplot(x , y)
        ax.set_title(f'{y} over time'.format(y))
        self.draw()


class MplWidget(QWidget):

    def __init__(self, parent = None):
        QWidget.__init__(self,parent)
        
        
        self.fig = Figure()
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(self.vertical_layout)
    
    
    def plot(self, x, y, df):
        self.canvas.axes.cla()
        self.canvas.draw()
        self.canvas.axes.set_title('{} distribution'.format(y))
        self.canvas.axes.stem(df[x] , df[y], 'black', markerfmt='ko')
        self.canvas.draw()

    def clear(self):
        self.canvas.axes.cla()
        self.canvas.draw()


class DistributionPlot(QWidget):

    def __init__(self, parent = None):
        QWidget.__init__(self,parent)
        
        
        self.fig = Figure()
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(self.vertical_layout)
    
    
    def plot(self, df, column):
        self.canvas.axes.cla()
        self.canvas.draw()
        self.canvas.axes.set_title('{} distribution'.format(column))
        y = df[column].dropna().to_numpy()[:, np.newaxis]
        x = np.arange(y.shape[0])[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian')
        # bandwidth = np.arange(0.05, 2, .05)
        bandwidth = np.logspace(-1,1, 20)
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(y)
        kde = grid.best_estimator_
        logprob = kde.score_samples(x)
        
        self.canvas.axes.fill_between(np.arange(y.size), 0,
                              np.exp(logprob), alpha=0.5)
        self.canvas.axes.set_xlim(min(y)[0], max(y)[0])
        self.canvas.axes.plot(x[:, 0], np.exp(logprob))
        self.canvas.axes.plot(df[column].dropna().to_numpy(), 
                              np.zeros(y.shape[0]), '|', color='black',
                              alpha=0.15)
        self.canvas.draw()


    def clear(self):
        self.canvas.axes.cla()
        self.canvas.draw()


class YamlWorker:
    @staticmethod
    def load_yaml(path=None):
        if path is None:
            file_name = glob('*.yaml')[0]
        else:
            file_name = path
        with open(file_name, 'r') as file:
            yaml_file = yaml.safe_load(file)
        return yaml_file
    
    
    @staticmethod
    def save_yaml(dictionary, save_filename):
        with open(f'{save_filename}.yaml', 'w') as file:
            yaml.dump(dictionary, file)


if __name__ == '__main__':
    LineEdit()
    SaveWorker()
    MiniSaveWorker()
    WorkerSignals()
    MplWidget()
    DistributionPlot()
    NumpyEncoder()
    YamlWorker()