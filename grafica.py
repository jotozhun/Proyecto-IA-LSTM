from PyQt5.QtWidgets import QApplication, QWidget, QCom
from PyQt5 import uic
import sys
app = QApplication([])
win = uic.loadUi("mainGUI.ui")
win.show()

sys.exit(app.exec())
