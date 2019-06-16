import cv2
import pyopenpose as op

cap = cv2.VideoCapture('user4.mp4')

video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cnt = 0

out_video_dir = 'user4_op/'

while cap.isOpened():
	out_path = '%s/%04d.png' % (out_video_dir, cnt)

	_, img = cap.read()

	params = dict()
	params["model_folder"] = "../models/"
	params["number_people_max"] = 1;
	
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	datum = op.Datum()
	datum.cvInputData = img
	opWrapper.emplaceAndPop([datum])

	img = datum.cvOutputData

	cv2.imwrite(out_path,img)

	cnt += 1;

	if cnt == frame_cnt or cnt == 1200:
		break;
