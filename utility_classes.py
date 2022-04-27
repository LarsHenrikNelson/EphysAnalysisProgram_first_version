# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:29:22 2021

Last updated on Wed Feb 16 12:33:00 2021

@author: LarsNelson
"""
from copy import deepcopy
from glob import glob
import json
from pathlib import PurePath

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLineEdit, QSizePolicy, QWidget, QVBoxLayout,
                             QListView, QSpinBox, QStyledItemDelegate,
                             QHBoxLayout, QLabel, QAbstractItemDelegate)
from PyQt5.QtCore import (QRunnable, pyqtSlot, QObject,
                          pyqtSignal, Qt, QAbstractListModel,
                          QPointF, QLineF, QSize)
import pyqtgraph as pg
from scipy.fftpack import sc_diff
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import yaml

from utilities import load_scanimage_file

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


class ListView(QListView):

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # self.item_list = []
        self.setSelectionMode(self.MultiSelection)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, e):
        """
        This function will detect the drag enter event from the mouse on the
        main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
	
    
    def dragMoveEvent(self, e):
        """
		This function will detect the drag move event on the main window
		"""
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
	
    
    def dropEvent(self, e):
        """
		This function will enable the drop file directly on to the 
		main window. The file location will be stored in the self.filename
		"""
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            for url in e.mimeData().urls():
                fname = PurePath(str(url.toLocalFile()))
                if fname not in self.model().fname_list:
                    self.model().fname_list += [fname]
                    acq_components = load_scanimage_file(fname)
                    self.model().acq_list += [acq_components]
            self.model().acq_list.sort(key=lambda x: int(x[0].split('_')[-1]))
            self.model().fname_list.sort(key=lambda x: int(x.stem.split('_')[-1]))
            self.model().layoutChanged.emit()
        else:
            e.ignore()


class ListModel(QAbstractListModel):
    def __init__(self, acq_list=None, fname_list=None,
                 header_name='Acquisition(s)'):
        super().__init__()
        self.acq_list = acq_list or []
        self.fname_list = fname_list or []
        self.header_name = header_name

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            acq_component = self.acq_list[index.row()]
            return acq_component[0]
    

    def headerData(self, name, role):
        if role == Qt.ItemDataRole.DisplayRole:
            name = self.header_name
            return name

    def rowCount(self, index):
        return len(self.acq_list)


class StringBox(QSpinBox):
    def __init__(self, parent=None):
        super(StringBox, self).__init__(parent)
        strings = []
        self.setStrings(strings)


    def setStrings(self, strings):
        strings = list(strings)
        self._strings = tuple(strings)
        self._values = dict(zip(strings, range(len(strings))))
        self.setRange(0, len(strings)-1)

    def textFromValue(self, value):
        
        # returning string from index
        # _string = tuple
        return self._strings[value]


class ItemDelegate(QStyledItemDelegate):

    def __init__(self):
        super().__init__()

    def paint(self, painter, option, index):
        acq_components = index.model().data(index,
                            Qt.ItemDataRole.DisplayRole)
        array = acq_components[2]
        width = option.rect.width()
        height = option.rect.height()
        norm_array = (array-np.max(array))/np.min(array)*height+option.rect.y()
        x_array = np.linspace(width*0.1, width, len(array)).tolist()
        zip_list = [QPointF(i, j) for i, j in zip(x_array, norm_array)]
        h = [QLineF(i, j) for i, j in zip(zip_list[:-1], zip_list[1:])]
        painter.setClipping(True)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawLines(h)
        
    def sizeHint(self, option, index):
        return QSize(100, 200)


class ListModel2(QAbstractListModel):
    def __init__(self, acq_list=None, id_list=None,
    header_name='Acquisition(s)'):
        super().__init__()
        self.acq_list = acq_list or []
        self.id_list = id_list or []
        self.header_name = header_name

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            acq = self.acq_list[index.row()]
            return acq

    # def headerData(self, name, role):
    #     if role == Qt.ItemDataRole.DisplayRole:
    #         name = self.header_name
            # return name

    def rowCount(self, index):
        return len(self.acq_list)


class ListView2(QListView):

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(self.MultiSelection)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, e):
        """
        This function will detect the drag enter event from the mouse on the
        main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
	
    
    def dragMoveEvent(self, e):
        """
		This function will detect the drag move event on the main window
		"""
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
	
    
    def dropEvent(self, e):
        """
		This function will enable the drop file directly on to the 
		main window. The file location will be stored in the self.filename
		"""
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            for url in e.mimeData().urls():
                fname = PurePath(str(url.toLocalFile()))
                if fname.stem not in self.model().id_list:
                    self.model().acq_list += [load_scanimage_file(fname)]
                    self.model().id_list +=[fname.stem]
            self.model().acq_list.sort(key=lambda x: int(x[0].split('_')[1]))
            self.model().layoutChanged.emit()
        else:
            e.ignore()

class ItemDelegate(QAbstractItemDelegate):
    def createEditor(self, parent, options, index):
        widget = CustomPlotWidget()
        return widget

    def setEditorData(self, editor, index):
        return editor.setDisplayData(index.data())

    def setModelData(self, editor, model, index):
        pass

class CustomPlotWidget(QWidget):
    def __init__(self):
        self.layout = QHBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.label = QLabel()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.plot)
    
    
    def setDisplayData(self, data_tuple):
        self.plot(data_tuple[2])
        self.label.setText(data_tuple[0])


if __name__ == '__main__':
    LineEdit()
    SaveWorker()
    MiniSaveWorker()
    WorkerSignals()
    MplWidget()
    DistributionPlot()
    NumpyEncoder()
    YamlWorker()
    ListView()
    ListModel()
    StringBox()
    ListModel2()
    ListView2()
    ItemDelegate()