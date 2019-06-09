# -*- Encoding:UTF-8 -*- #
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from baseUI import Ui_MainWindow as Ui_MainWindow_base
from opmlify.opmlify import main
import cv2
import os
import atexit

class Ui_MainWindow(QMainWindow, Ui_MainWindow_base):
	def __init__(self):
        	super(self.__class__, self).__init__()
		self.setupUi(self)

		self.file_name = ''
		self.folder_name1 = ''
		self.folder_name2 = ''
		self.frame = -1			

		self.processOn = False

		atexit.register(self.exit_handler)

	def exit_handler(self):
		if self.processOn:
			print "exit"

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

			self.obj = Worker(out_name)			
			self.thread = QThread()
			self.thread.started.connect(self.obj.process)
			self.obj.finish_process.connect(self.finish_process)
			self.obj.refresh_cur_frame.connect(self.refresh_cur_frame)
			self.obj.moveToThread(self.thread)

			self.thread.start()
			self.processOn = True
			self.btn_process.setEnabled(False)

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

	def refresh_cur_frame(self, i):
		print(i)
		
	def finish_process(self):
		self.processOn = False
		self.btn_process.setEnabled(True)

	def open_render1(self): #폴더명 두 개 받아서 rendering해서 넘기면서 비교 가능하게

                # sys(python render_similarity.py) 하면 될 듯?
                # 아니면 imshow 대신에 QT에 띄울까
                # reset, back, 유사도 측정 정도만 버튼 만들까
                #QfileDialog dialog(self)
#                QFileDialog.setFileMode(QFileDialog, QFileDialog.Directory)

 #               self.folder_name1=str(QFileDialog.getExistingDirectory())
#		self.label_folder1.setText(self.folder_name1)
		folder1, _ = QFileDialog.getExistingDirectory(self, 'Open Folder','','Folder')
		self.folder_name1 = str(folder1)
		self.label_folder1.setText(self.folder_name1)

        def open_render2(self):
                #QfileDialog dialog(self)
		folder2, _ = QFileDialog.getExistingDirectory(self, 'Open Foler','','Folder')
                self.folder_name2=str(folder2)
		self.label_folder2.setText(self.folder_name2)

        def start_render(self):
                if(self.folder_name1=='' or self.folder_name2==''):
                        return
                os.system("python render_similarity.py "+self.folder_name1+" "+self.folder_name2)


class Worker(QObject):
	refresh_cur_frame = pyqtSignal(int)
	finish_process = pyqtSignal()

	def __init__(self, out_name):
		super(self.__class__, self).__init__(None)
		self.out_name = out_name
		self.cur = 0

	def process(self):
		print "@@@", self.cur
		res, isFinish = main(self.out_name, True, self.cur)
		self.cur = res

		if isFinish:
			self.finish_process.emit()
		else:
			self.refresh_cur_frame.emit(res)

