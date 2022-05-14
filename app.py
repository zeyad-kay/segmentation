import gui
from PyQt5 import QtWidgets , QtCore, QtGui
import time
import sys
import cv2
import matplotlib.pyplot as plt
from src.threshold import global_threshold, local_threshold

class MainWindow(QtWidgets.QMainWindow , gui.Ui_MainWindow):
    # resized = QtCore.pyqtSignal()
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.widgets = [self.input_img,self.output_img,self.input_img_2,self.output_img_2]
        self.widget_configuration()
        self.default_img()
        self.open_button.clicked.connect(self.open_image)
        self.open_button_2.clicked.connect(self.open_image)
        self.apply_button.clicked.connect(self.apply_threshold)
    

    def open_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',"", "Image Files (*.png *jpeg *.jpg)")
        if self.tabWidget.currentIndex( ) == 0:
            self.threshold_img = cv2.imread(self.file_path)
            self.img_rgb = cv2.cvtColor(self.threshold_img, cv2.COLOR_BGR2RGB)
            self.apply_button.setEnabled(True)
            self.comboBox.setEnabled(True)
            self.radio_global.setEnabled(True)
            self.radio_local.setEnabled(True)
            self.size_input.setEnabled(True)
            self.display(self.img_rgb,self.widgets[0])
            self.output_img.clear()
        else:
            self.clustring_img = cv2.imread(self.file_path)
            self.img_rgb = cv2.cvtColor(self.clustring_img, cv2.COLOR_BGR2RGB)
            self.apply_button_2.setEnabled(True)
            self.comboBox_2.setEnabled(True)
            self.cluster_input.setEnabled(True)
            self.threshold_input.setEnabled(True)
            self.display(self.img_rgb,self.widgets[2])
            self.output_img.clear()


    def apply_threshold(self):
        start = time.time()
        if self.comboBox.currentText() == "Optimal Thresholding":
            if self.radio_global.isChecked():
                self.threshold_output = global_threshold(self.threshold_img,"optimal")
            else:
                self.threshold_output = local_threshold(self.threshold_img,int(self.size_input.text()),"optimal")
        elif self.comboBox.currentText() == "Otsu Thresholding":
            if self.radio_global.isChecked():
                self.threshold_output = global_threshold(self.threshold_img,"otsu")
            else:
                self.threshold_output = local_threshold(self.threshold_img,int(self.size_input.text()),"otsu")
        else:
            if self.radio_global.isChecked():
                self.threshold_output = global_threshold(self.threshold_img,"spectral")
            else:
                self.threshold_output = local_threshold(self.threshold_img,int(self.size_input.text()),"spectral")
        end = time.time()
        self.threshold_output = cv2.cvtColor(self.threshold_output, cv2.COLOR_BGR2RGB)
        self.display(self.threshold_output,self.widgets[1])
        self.time_label.setText(str("{:.3f}".format(end-start)) + " Seconds")
        
    def display(self , data , widget):
            data = cv2.transpose(data)
            widget.setImage(data)
            widget.view.setLimits(xMin=0, xMax=data.shape[0], yMin= 0 , yMax= data.shape[1])
            widget.view.setRange(xRange=[0, data.shape[0]], yRange=[0, data.shape[1]], padding=0)    

    def widget_configuration(self):

        for widget in self.widgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def default_img(self):
        defaultImg = plt.imread("images/default-image.jpg")
        for widget in self.widgets:
            self.display(defaultImg,widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()