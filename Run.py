# -*- coding:utf-8 -*-
# @Time : 2022/2/22 22:09
# @Author : 西~南~北
# @File : Run.py
# @Software: PyCharm

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import *

import llp1
import Split_ch1
import Ch_r1
from LRRGUI import Ui_MainWindow


class DetailUI(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(DetailUI, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('车牌识别')

    def openImage(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'open file', '.', "Image files (*.jpg *.gif *.png)")
            self.label_5.setPixmap(QPixmap(fname))
            print(fname)

            llp = llp1.llp(fname)
            llp.fun()

            img = llp.reimg

            # split = Split_ch1.Split_ch(img)
            # split.fun()

            ch_r = Ch_r1.Ch_r1(img)
            ch_r.fun()

            self.label.setPixmap(QPixmap('./1.jpg'))
            self.label_2.setPixmap(QPixmap('./2.jpg'))
            self.label_3.setText(str(ch_r.result))

        except:
            self.textEdit.setText("打开文件失败，可能是文件内型错误")

    def runModel(self):
        pass


# 有重写
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DetailUI()
    ex.show()
    sys.exit(app.exec_())

# 无重写
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mainWindow = QMainWindow()
#     ui = LRRGUI.Ui_MainWindow()
#     ui.setupUi(mainWindow)
#     mainWindow.show()
#     sys.exit(app.exec_())
