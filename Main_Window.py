# -*- Encoding:UTF-8 -*- #
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt
from baseUI import Ui_MainWindow as Ui_MainWindow_base
import cv2
import os
import atexit
import numpy as np
from smpl_webuser.serialization import load_model
from glob import glob
import cPickle as pickle
import sys

from opmlify.opmlify import mmain
from render_similarity import render_model, similarity

class Ui_MainWindow(QMainWindow, Ui_MainWindow_base):
	def __init__(self):
        	super(self.__class__, self).__init__()
		self.setupUi(self)
		self.setWindowTitle('DANCE HELPER')

		self.file_name = ''
		self.frame = -1			

		self.processOn = False

		self.file_paths = []
		self.isFirst = True

		self.compared = False
		self.mode1 = 0
		self.mode2 = 0
		self.m = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
		self.m2 = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
		self.pkl_paths = []
		self.pkl_paths2 = []
		self.op_joints = []
		self.op_joints2 = []
		self.ind = 0
		self.ind2 = 0

		self.rotx = 0.0
		self.roty = 0.0
		self.rotz = 0.0
		self.rotx2 = 0.0
		self.roty2 = 0.0
		self.rotz2 = 0.0

		self.res = None
		self.res2 = None
		self.img = None
		self.img2 = None

		self.w = None
		self.h = None
		self.w2 = None
		self.h2 = None

		self.label_sim_title.hide()

		model_temp = QStandardItemModel()
		RESULT_DIR = 'opmlify/result'
		for dirpath, dirname, filename in os.walk(RESULT_DIR):
			if self.isFirst:
				self.isFirst = False
				for a in dirname:
					model_temp.appendRow(QStandardItem(a))
					self.file_paths.append(RESULT_DIR + '/' + a)
				self.listView1.setModel(model_temp)
				self.listView2.setModel(model_temp)
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
		img = img.scaledToHeight(180)
		self.label_img.setPixmap(img)
		self.label_img.setScaledContents(False)
		self.label_img.show()
		self.label_path.setText(self.file_name)
		self.label_imageSize.setText(imageSize)
		self.label_frame.setText(frame)
		self.label_time.setText(time)
		self.label_fps.setText(fps)
		self.label_length.setText(length)

	def process(self):
		if self.file_name is '':
			QMessageBox.information(QWidget(), "Error", "Open File First")
		else:
			print "main process"
			out_name = self.file_name.split('/')[-1]

			self.obj = Worker(out_name)			
			self.thread = QThread()
			self.thread.started.connect(self.obj.process)
			self.obj.finish_process.connect(self.finish_process)
			self.obj.refresh_cur_frame.connect(self.refresh_cur_frame)
			self.obj.moveToThread(self.thread)

			self.processOn = True
			self.btn_process.setEnabled(False)
			self.thread.start()

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

	def refresh_cur_frame(self, i):
		print "refresh_cur_frame", i
		self.process()
		
	def finish_process(self):
		print "finish"
		self.processOn = False
		self.btn_process.setEnabled(True)

	def open_render1(self): 
		folder1, _ = QFileDialog.getOpenFileName(self, 'Open Folder','','Folder')
		self.folder_name1 = str(folder1)
		self.label_folder1.setText(self.folder_name1)

        def open_render2(self):
                QFileDialog.setFileMode(QFileDialog.Directory)

                self.folder_name2=str(QFileDialog.getExistingDirectory())
		self.label_folder2.setText(self.folder_name2)

	def change1(self):
		self.mode1 += 1
		if self.mode1 == 3:
			self.mode1 = 0
		self.myshow()
	def change2(self):
		self.mode2 += 1
		if self.mode2 == 3:
			self.mode2 = 0
		self.myshow()

        def start_render(self):
		self.m = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
		self.m2 = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
		self.pkl_paths = []
		self.pkl_paths2 = []
		self.op_joints = []
		self.op_joints2 = []
		self.ind = 0
		self.ind2 = 0

		self.rotx = 0.0
		self.roty = 0.0
		self.rotz = 0.0
		self.rotx2 = 0.0
		self.roty2 = 0.0
		self.rotz2 = 0.0

		self.res = None
		self.res2 = None
		self.img = None
		self.img2 = None

		self.w = None
		self.h = None
		self.w2 = None
		self.h2 = None
		self.mode1 = 0
		self.mode2 = 0

		sel1 = self.listView1.currentIndex().row()
		sel2 = self.listView2.currentIndex().row()

		if sel1 == -1 or sel2 == -1:
			QMessageBox.information(QWidget(), "Error", "Select Two Items")
		elif sel1 == sel2:
			QMessageBox.information(QWidget(), "Error", "Select Different Items")
		else:
			body = np.zeros(10)
			body2 = np.zeros(10)
			bodyIndex = np.zeros([10,20])
			bodyIndex2 = np.zeros([10,20])
			bodyTemp = np.zeros([10,20])
			bodyTemp2 = np.zeros([10,20])
			indexMax = 0
			indexMax2 = 0
			valueMax = 0.0
			valueMax2 = 0.0
		
			hyunhopkl_path = self.file_paths[sel1] + '/hyunho.pkl'
			with open(hyunhopkl_path, 'rb') as f:
				op_datas = pickle.load(f)
				self.op_joints = op_datas['op_joints']
			hyunhopkl_path2 = self.file_paths[sel2] + '/hyunho.pkl'
			with open(hyunhopkl_path2, 'rb') as f:
				op_datas = pickle.load(f)
				self.op_joints2 = op_datas['op_joints']

			self.pkl_paths = sorted(glob(self.file_paths[sel1] + '/*[0-9].pkl'))
			self.pkl_paths2 = sorted(glob(self.file_paths[sel2] + '/*[0-9].pkl'))

			for ind, pkl_path in enumerate(self.pkl_paths):
				if self.w == None:
					temp_img = cv2.imread(pkl_path[:-4] + '.png')
					self.w = temp_img.shape[1]
					self.h = temp_img.shape[0]
				
				with open(pkl_path,'r') as f:
					self.res = pickle.load(f)
				for i in range(10):
					for j in range(20):
						if self.res['betas'][i] >= j-10 and self.res['betas'][i] < j-9:
							bodyIndex[i][j] += 1
							bodyTemp[i][j] += self.res['betas'][i]
			for ind, pkl_path2 in enumerate(self.pkl_paths2):
				if self.w2 == None:
					temp_img = cv2.imread(pkl_path2[:-4] + '.png')
					self.w2 = temp_img.shape[1]
					self.h2 = temp_img.shape[0]

				with open(pkl_path2,'r') as f2:
					self.res2 = pickle.load(f2)
				for i in range(10):
					for j in range(20):
						if self.res2['betas'][i] >= j-10 and self.res2['betas'][i] < j-9:
							bodyIndex2[i][j] += 1
							bodyTemp2[i][j] += self.res2['betas'][i]


			for i in range(10):
				for j in range(20):
					if bodyIndex[i][j] > indexMax:
						indexMax = bodyIndex[i][j]
						valueMax = bodyTemp[i][j]
					if bodyIndex2[i][j] > indexMax2:
						indexMax2 = bodyIndex2[i][j]
						valueMax2 = bodyTemp2[i][j]
				body[i] = valueMax / (indexMax+sys.float_info.epsilon)
				body2[i] = valueMax2 / (indexMax2+sys.float_info.epsilon)
				indexMax = 0
				indexMax2 = 0

			self.m.betas[:] = body[:]
			self.m2.betas[:] = body2[:]

			self.compared = True
			self.myshow()

	def show_video(self):
		return

	def myshow(self):
		if not self.compared:
			return
		if self.mode1 == 0:
			with open(self.pkl_paths[self.ind],'r') as f:
				self.res = pickle.load(f) 
			ff = self.res['f']
			tt = self.res['cam_t']
			self.m.pose[:] = self.res['pose']
			self.img = render_model(self.m, self.m.f, self.w, self.h, np.array([self.rotx, self.roty,  self.rotz]), tt, ff)
			# convert to QImage
			img = 255 * self.img
			img = img.astype(np.uint8)
			img_RGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
		elif self.mode1 == 1:
			temp_paths = self.pkl_paths[self.ind][:-8]
			img_path = temp_paths + 'img/' + self.pkl_paths[self.ind][-8:-4] + '.png'
			print img_path
			self.img = cv2.imread(img_path)
			img_RGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
		elif self.mode1 == 2:
			with open(self.pkl_paths[self.ind],'r') as f:
				self.res = pickle.load(f) 
			ff = self.res['f']
			tt = self.res['cam_t']
			self.m.pose[:] = self.res['pose']
			self.img = render_model(self.m, self.m.f, self.w, self.h, np.array([self.rotx, self.roty,  self.rotz]), tt, ff)
			# convert to QImage
			img = 255 * self.img
			img = img.astype(np.uint8)
			img_RGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

			temp_paths = self.pkl_paths[self.ind][:-8]
			img_path = temp_paths + 'img/' + self.pkl_paths[self.ind][-8:-4] + '.png'
			self.img = cv2.imread(img_path)
			img_RGB2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

			rows, cols, channels = img_RGB.shape
			roi = img_RGB2[:rows, :cols]

			img2gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
			ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
			mask2 = cv2.bitwise_not(mask)
			
			bg = cv2.bitwise_and(roi, roi, mask=mask)
			fg = cv2.bitwise_and(img_RGB, img_RGB, mask=mask2)
			img_RGB = cv2.add(bg, fg)

			
		qImg = QImage(img_RGB.data, img_RGB.shape[1], img_RGB.shape[0], img_RGB.shape[1] * 3, QImage.Format_RGB888)
		qpImg = QPixmap(qImg)
		qpImg = qpImg.scaledToHeight(360)
		self.label_show1.setPixmap(qpImg)
		self.label_show1.setScaledContents(False)
		self.label_show1.show()

		if self.mode2 == 0:
			with open(self.pkl_paths2[self.ind2],'r') as f2:
				self.res2 = pickle.load(f2)
			ff2 = self.res2['f']
			tt2 = self.res2['cam_t']
			self.m2.pose[:] = self.res2['pose']
			self.img2 = render_model(self.m2, self.m2.f, self.w2, self.h2, np.array([self.rotx2, self.roty2, self.rotz2]), tt2, ff2)
			# convert to QImage
			img = 255 * self.img2
			img = img.astype(np.uint8)
			img_RGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
			self.img2 = img_RGB
		elif self.mode2 == 1:
			temp_paths = self.pkl_paths2[self.ind2][:-8]
			img_path = temp_paths + 'img/' + self.pkl_paths2[self.ind2][-8:-4] + '.png'
			self.img2 = cv2.imread(img_path)
			img_RGB = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
			self.img2 = img_RGB
		elif self.mode2 == 2:
			with open(self.pkl_paths2[self.ind2],'r') as f:
				self.res2 = pickle.load(f) 
			ff2 = self.res2['f']
			tt2 = self.res2['cam_t']
			self.m2.pose[:] = self.res2['pose']
			self.img2 = render_model(self.m2, self.m2.f, self.w2, self.h2, np.array([self.rotx2, self.roty2,  self.rotz2]), tt2, ff2)
			# convert to QImage
			img = 255 * self.img2
			img = img.astype(np.uint8)
			img_RGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

			temp_paths = self.pkl_paths2[self.ind2][:-8]
			img_path = temp_paths + 'img/' + self.pkl_paths2[self.ind2][-8:-4] + '.png'
			self.img2 = cv2.imread(img_path)
			img_RGB2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

			rows, cols, channels = img_RGB.shape
			roi = img_RGB2[:rows, :cols]

			img2gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
			ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
			mask2 = cv2.bitwise_not(mask)
			
			bg = cv2.bitwise_and(roi, roi, mask=mask)
			fg = cv2.bitwise_and(img_RGB, img_RGB, mask=mask2)
			img_RGB = cv2.add(bg, fg)
			self.img2 = img_RGB
			
			
		qImg = QImage(img_RGB.data, img_RGB.shape[1], img_RGB.shape[0], img_RGB.shape[1] * 3, QImage.Format_RGB888)
		qpImg = QPixmap(qImg)
		qpImg = qpImg.scaledToHeight(360)
		self.label_show2.setPixmap(qpImg)
		self.label_show2.setScaledContents(False)
		self.label_show2.show()
		self.label_sim_title.hide()
		self.label_sim.hide()


	def keyPressEvent(self, e):
		if self.compared:
			k = e.key()
			if k == ord('A'): # 원본 왼쪽
				self.roty += 0.1
				self.myshow()
			elif k == ord('D'): # 원본 오른쪽
				self.roty -= 0.1
				self.myshow()
			elif k == ord('W'): # 원본 위로
				self.rotx -= 0.1
				self.myshow()
			elif k == ord('S'): # 원본 아래로
				self.rotx += 0.1
				self.myshow()
			elif k == ord('Q'): # 원본 왼쪽 대각선
				self.rotz -= 0.1
				self.myshow()
			elif k == ord('E'): # 원본 오른쪽 대각선
				self.rotz += 0.1
				self.myshow()
			elif k == ord('R'): # 둘다 reset
				self.rotx2 = self.roty2 = self.rotz2 = self.rotx = self.roty = self.rotz = 0.0
				self.myshow()
			elif k == ord('B'): # 둘다 back
				self.roty += np.pi
				self.roty2 += np.pi
				self.myshow()
			elif k == ord('J'):
				self.roty2 += 0.1
				self.myshow()
			elif k == ord('L'):
				self.roty2 -= 0.1
				self.myshow()
			elif k == ord('I'):
				self.rotx2 -= 0.1
				self.myshow()
			elif k == ord('K'):
				self.rotx2 += 0.1
				self.myshow()
			elif k == ord('U'):
				self.rotz2 -= 0.1
				self.myshow()
			elif k == ord('O'):
				self.rotz2 += 0.1
				self.myshow()
			elif k == ord('Z'): # 원본 이전 프레임
				if self.ind > 0: # 0보다 안작아지게
					self.ind -= 1
					self.myshow()
			elif k == ord('X'): # 원본 다음 프레임
				if self.ind < len(self.pkl_paths): # 원본 프레임 수보다 안커지게
					self.ind += 1
					self.myshow()
			elif k == ord('N'): # 비교 이전 프레임
				if self.ind2 > 0: # 0보다 안작아지게
					self.ind2 -= 1
					self.myshow()
			elif k == ord('M'): # 비교 다음 프레임
				if self.ind2 < len(self.pkl_paths2): # 비교 프레임 수보다 안커지게
					self.ind2 += 1
					self.myshow()
			elif k == ord('Y'): # 유사도 분석
				sim = similarity(self.res, self.res2, self.ind, self.ind2, self.img2, self.op_joints, self.op_joints2)

				img_RGB = self.img2
				qImg = QImage(img_RGB.data, img_RGB.shape[1], img_RGB.shape[0], img_RGB.shape[1] * 3, QImage.Format_RGB888)
				qpImg = QPixmap(qImg)
				qpImg = qpImg.scaledToHeight(360)
				self.label_show2.setPixmap(qpImg)
				self.label_show2.setScaledContents(False)
				self.label_show2.show()
				self.label_sim_title.show()
				self.label_sim.setText(str(format(sim, '.2f')) + '%')
				self.label_sim.show()
				
			elif k == ord('V'): # double frame
				self.ind += 1
				self.ind2 += 1
				self.myshow()

class Worker(QObject):
	refresh_cur_frame = pyqtSignal(int)
	finish_process = pyqtSignal()

	def __init__(self, out_name):
		super(self.__class__, self).__init__(None)
		self.out_name = out_name
		self.cur = 0

	def process(self):
		print "@@@ worker process", self.cur
		res, isFinish = mmain(video=self.out_name, ui=True, num=self.cur, female=True, opFilter=True, gamma=1.0)
		self.cur = res

		if isFinish:
			self.finish_process.emit()
		else:
			self.refresh_cur_frame.emit(res)


