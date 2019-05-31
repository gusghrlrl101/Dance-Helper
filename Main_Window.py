# -*- Encoding:UTF-8 -*- #
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from baseUI import Ui_MainWindow as Ui_MainWindow_base
from opmlify.opmlify import main
import cv2

class Ui_MainWindow(QMainWindow, Ui_MainWindow_base):
	def __init__(self):
        	super(self.__class__, self).__init__()
		self.setupUi(self)

		self.file_name = ''
		self.frame = -1

	def open_file(self):
		# open file
		file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Video (*.mp4 *.avi)')

		self.file_name = str(file_name)

		# if cancel, do nothing
		if self.file_name ==  '':
			return

		# get data
		img, imageSize, frame, time, fps, length = self.make_thumbnail(self.file_name)
		self.frame = int(frame)

		# set data
		self.label_img.setPixmap(img)
		self.label_img.setScaledContents(True)
		self.label_img.show()
		self.label_path.setText(self.file_name)
		self.label_imageSize.setText(imageSize)
		self.label_frame.setText(frame)
		self.label_time.setText(time)
		self.label_fps.setText(fps)
		self.label_length.setText(length)

	def process(self):
		if self.file_name is None:
			QMessageBox.information(QWidget(), "Error", "Open File First")
		else:
			out_name = self.file_name.split('/')[-1]

			num = 0
			while num != self.frame - 1:
				res = main(out_name, True, num)
				num = res
				print "@@@", res

	def make_thumbnail(self, fileName):
		# open video
		cap = cv2.VideoCapture(fileName)

		# get info
		fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
		frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		time = str(frame * 25 / 60) + " minute"
		size = str(w) + " x " + str(h)
		length = str(round(frame / fps, 2)) + " sec"

		# capture first image
		if cap.isOpened():
			_, img = cap.read()
		cap.release()

		# convert to QImage
		img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		qImg = QImage(img_RGB.data, w, h, w * 3, QImage.Format_RGB888)

		return QPixmap(qImg), size, str(frame), time, str(fps), length


	def hyunho_click(self):
		if self.file_name is None:
			print 'not open'
		else:
			print 'open'

	def open_render(self): #폴더명 두 개 받아서 rendering해서 넘기면서 비교 가능하게
		file_name1=''
		file_name2=''
		# sys(python render_similarity.py) 하면 될 듯?
		# 아니면 imshow 대신에 QT에 띄울까
		# reset, back, 유사도 측정 정도만 버튼 만들까
