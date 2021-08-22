from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQTAgg
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class MplChart(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvasQTAgg(Figure())
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axe
