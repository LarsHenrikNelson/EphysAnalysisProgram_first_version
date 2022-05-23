from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QComboBox, QWidget,
                             QLabel, QFormLayout, QApplication,
                             QVBoxLayout, QTabWidget, QListWidget, QColorDialog,
                             QStackedLayout, QGraphicsView)
import pyqtgraph as pg
import qdarkstyle


class PreferencesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.tabs_widget = QListWidget()
        self.tabs_widget.addItems(['Application settings', 'Mini Analysis',
            'oEPSC Analysis', 'Current Clamp Analysis'])
        self.tabs_widget.setSelectionMode(QListWidget.SingleSelection)
        self.tabs_widget.setSelectionBehavior(QListWidget.SelectItems)
        self.tabs_widget.itemClicked.connect(self.set_widget)
        self.layout.addWidget(self.tabs_widget)
        self.tabs_widget.setMaximumWidth(self.tabs_widget.sizeHintForColumn(0))
        self.tabs_widget.setCurrentRow(0)

        self.stackedlayout = QStackedLayout()
        self.layout.addLayout(self.stackedlayout)
        self.stackedlayout.setCurrentIndex(0)

        #Appearence tab
        self.app_appearance = QWidget()
        self.tab1_layout = QFormLayout()
        self.app_appearance.setLayout(self.tab1_layout)

        self.theme_label = QLabel('Theme')
        self.theme_box  = QComboBox()
        self.theme_box.addItems(["Dark style", "Light style"])
        self.theme_box.currentTextChanged.connect(self.style_choice)
        self.tab1_layout.addRow(self.theme_label, self.theme_box)
        self.stackedlayout.addWidget(self.app_appearance)


        #Mini Analysis tab
        self.mini_tab = MiniAnalysisSettings()
        self.stackedlayout.addWidget(self.mini_tab)


    def set_widget(self, clicked_item):
        if clicked_item.text() == "Application settings":
            self.stackedlayout.setCurrentIndex(0)
        elif clicked_item.text() == 'Mini Analysis':
            self.stackedlayout.setCurrentIndex(1)
        else:
            pass

    def style_choice(self, text):
        if text == 'Dark style':
            stylesheet = qdarkstyle.load_stylesheet(qdarkstyle.dark.palette.DarkPalette)
        else:
            stylesheet = qdarkstyle.load_stylesheet(qdarkstyle.light.palette.LightPalette)
        app = QApplication.instance()
        app.setStyleSheet(stylesheet)


class MiniAnalysisSettings(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.template_axis_label = QLabel('Template plot axis')
        self.template_axis_color = QPushButton()
        self.template_axis_color.setObjectName('Template plot axis')
        self.template_axis_color.clicked.connect(lambda checked: 
            self.setColor(self.template_axis_color, 'Template plot', 'axis'))
        self.layout.addRow(self.template_axis_label, self.template_axis_color)

        self.template_bgd_label = QLabel('Template plot background')
        self.template_bgd_color = QPushButton()
        self.template_bgd_color.setObjectName('Template plot background')
        self.template_bgd_color.clicked.connect(lambda checked: 
            self.setColor(self.template_bgd_color, 'Template plot', 'background'))
        self.layout.addRow(self.template_bgd_label, self.template_bgd_color)

        self.p1_axis_label = QLabel('Scroll plot axis')
        self.p1_axis_color = QPushButton()
        self.p1_axis_color.setObjectName('Scroll plot axis')
        self.p1_axis_color.clicked.connect(lambda checked: 
            self.setColor(self.p1_axis_color, 'p1', 'axis'))
        self.layout.addRow(self.p1_axis_label, self.p1_axis_color)

        self.p1_bgd_label = QLabel('Scroll plot background')
        self.p1_bgd_color = QPushButton()
        self.p1_bgd_color.setObjectName('Scroll plot background')
        self.p1_bgd_color.clicked.connect(lambda checked:
            self.setColor(self.p1_bgd_color, 'p1', 'background'))
        self.layout.addRow(self.p1_bgd_label, self.p1_bgd_color)

        self.p2_axis_label = QLabel('Inspection plot axis')
        self.p2_axis_color = QPushButton()
        self.p2_axis_color.setObjectName('Inspection plot axis')
        self.p2_axis_color.clicked.connect(lambda checked:
            self.setColor(self.p2_axis_color, 'p2', 'axis'))
        self.layout.addRow(self.p2_axis_label, self.p2_axis_color)

        self.p2_bgd_label = QLabel('Inpsection plot background')
        self.p2_bgd_color = QPushButton()
        self.p2_bgd_color.setObjectName('Inspection plot background')
        self.p2_bgd_color.clicked.connect(lambda checked:
            self.setColor(self.p2_bgd_color, 'p2', 'background'))
        self.layout.addRow(self.p2_bgd_label, self.p2_bgd_color)

        self.mini_axis_label = QLabel('Mini plot axis')
        self.mini_axis_color = QPushButton()
        self.mini_axis_color.setObjectName('Mini plot axis')
        self.mini_axis_color.clicked.connect(lambda checked:
            self.setColor(self.mini_axis_color, 'Mini view plot', 'axis'))
        self.layout.addRow(self.mini_axis_label, self.mini_axis_color)

        self.mini_bgd_label = QLabel('Mini plot background')
        self.mini_bgd_color = QPushButton()
        self.mini_bgd_color.setObjectName('Mini plot background')
        self.mini_bgd_color.clicked.connect(lambda checked:
            self.setColor(self.mini_bgd_color, 'Mini view plot', 'background'))
        self.layout.addRow(self.mini_bgd_label, self.mini_bgd_color)

        self.ave_mini_axis_label = QLabel('Average mini plot axis')
        self.ave_mini_axis_color = QPushButton()
        self.ave_mini_axis_color.setObjectName('Ave mini plot axis')
        self.ave_mini_axis_color.clicked.connect(lambda checked:
            self.setColor(self.mini_axis_color, 'Ave mini plot', 'axis'))
        self.layout.addRow(self.ave_mini_axis_label, self.ave_mini_axis_color)

        self.ave_mini_bgd_label = QLabel('Average mini plot background')
        self.ave_mini_bgd_color = QPushButton()
        self.ave_mini_bgd_color.setObjectName('Ave mini plot background')
        self.ave_mini_bgd_color.clicked.connect(lambda checked:
            self.setColor(self.ave_mini_bgd_color, 'Ave mini plot',
            'background'))
        self.layout.addRow(self.ave_mini_bgd_label, self.ave_mini_bgd_color)

        self.color_dict = {
            'Template plot axis': '#ffffff',
            'Template plot background': '#000000',
            'Scroll plot axis': '#ffffff',
            'Scroll plot background': '#000000',
            'Inspection plot axis': '#ffffff',
            'Inspection plot background': '#000000',
            'Mini plot axis': '#ffffff',
            'Mini plot background': '#000000',
            'Ave mini plot axis': '#ffffff',
            'Ave mini plot background': '#000000',
        }

        self.set_button_color()
        self.set_width()


    def setColor(self, push_button, name, part):
        color = QColorDialog.getColor()
        push_button.setStyleSheet(f"background-color : {color.name()}")
        self.color_dict[push_button.objectName()] = color.name()
        app = QApplication.instance()
        plot = [i for i in app.allWidgets() if i.objectName() == name][0]
        if part == 'background':
            plot.setBackground(self.color_dict.get(push_button.objectName()))
        elif part == 'axis':
            plot.getAxis('left').setPen(self.color_dict.get(push_button.objectName()))
            plot.getAxis('left').setTextPen(self.color_dict.get(push_button.objectName()))
            plot.getAxis('bottom').setPen(self.color_dict.get(push_button.objectName()))
            plot.getAxis('bottom').setTextPen(self.color_dict.get(push_button.objectName()))


    def set_button_color(self):
        buttons = self.findChildren(QPushButton)
        for i in buttons:
            color = self.color_dict.get(i.objectName())
            i.setStyleSheet(f"background-color : {color}")


    def set_width(self):
        push_buttons = self.findChildren(QPushButton)
        for i in push_buttons:
            i.setMaximumWidth(30)


if __name__ == '__main__':
    PreferencesWidget()
    MiniAnalysisSettings()