from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from baseUI import Ui_MainWindow as Ui_MainWindow_base
from opmlify.opmlify import main
import cv2

class Ui_MainWindow(QMainWindow, Ui_MainWindow_base):
	def __init__(self):
        	super(self.__class__, self).__init__()
		self.setupUi(self)

		self.file_name = None


	def open_file(self):
		# open file
		self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Video (*.mp4 *.avi)')
		# get data
		img, size, frame, time = self.make_thumbnail(self.file_name)

		# set data
		self.label_img.setPixmap(img)
		self.label_img.setScaledContents(True)
		self.label_img.show()
		self.label_path.setText(self.file_name)
		self.label_size.setText(size)
		self.label_frame.setText(frame)
		self.label_time.setText(time)

	def process(self):
		if self.file_name is None:
			QMessageBox.information(QWidget(), "Error", "Open File First")
		else:
			main(10, 5000., 25., 'user6.avi', True)


	def make_thumbnail(self, fileName):
		# open video
		cap = cv2.VideoCapture(fileName)

		# get info
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
		frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		time = str(frame * 25 / 60) + " minutes"
		size = str(w) + " x " + str(h)

		# capture first image
		if cap.isOpened():
			_, img = cap.read()
		cap.release()

		# convert to QImage
		img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		qImg = QImage(img_RGB.data, w, h, w * 3, QImage.Format_RGB888)

		return QPixmap(qImg), size, str(frame), time


	def hyunho_click(self):
		if self.file_name is None:
			print 'not open'
		else:
			print 'open'
