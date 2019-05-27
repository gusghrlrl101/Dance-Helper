#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from Main_Window import Ui_MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    sys.exit(app.exec_())
