import cv2
import pyopenpose as op
import numpy as np
import pickle



def adjust_gamma(image, gamma=1.0):
    image = image / 255.0
    image = cv2.pow(image, 1.0 / gamma)
    return np.uint8(image * 255)

cap = cv2.VideoCapture('user4.mp4')

video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cnt = 0

ans = []
while cap.isOpened():
	_, img = cap.read()

	params = dict()
	params["model_folder"] = "../models/"
	params["number_people_max"] = 1;
	
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	img = adjust_gamma(img, 0.3)
	datum = op.Datum()
	datum.cvInputData = img
	opWrapper.emplaceAndPop([datum])

	img = datum.cvOutputData
#	cv2.imshow("img", img)
#	cv2.waitKey(0)

	mydata = datum.poseKeypoints[0]
	conf = np.zeros(24)

	conf[0] = 1.1
	conf[1] = mydata[12, 2]
	conf[2] = mydata[9, 2]
	conf[3] = 1.1
	conf[4] = mydata[13, 2]
	conf[5] = mydata[10, 2]
	conf[6] = 1.1
	conf[7] = mydata[14, 2]
	conf[8] = mydata[11, 2]
	conf[9] = 1.1
	conf[10] = mydata[19, 2]
	conf[11] = mydata[22, 2]
	conf[12] = mydata[1, 2]
	conf[13] = 1.1
	conf[14] = 1.1
	conf[15] = mydata[15, 2]
	conf[16] = mydata[5, 2]
	conf[17] = mydata[2, 2]
	conf[18] = mydata[6, 2]
	conf[19] = mydata[3, 2]
	conf[20] = mydata[7, 2]
	conf[21] = mydata[4, 2]
	conf[22] = 1.1
	conf[23] = 1.1

	ans.append(conf)

	print cnt
	cnt += 1
	if cnt == frame_cnt or cnt == 1200:
		break

with open("hyunho.pkl", 'w') as outf:
	pickle.dump(ans, outf)



