import cv2
import sys
import os

if len(sys.argv) < 2:
	print "ERROR: file name"
	exit()

file_name = sys.argv[1]

cap = cv2.VideoCapture(file_name + '.mp4')

video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cnt = 0

out_video_dir = file_name+ '/'

if not os.path.exists(out_video_dir):
	os.makedirs(out_video_dir)

while cap.isOpened():
	out_path = '%s/%04d.png' % (out_video_dir, cnt)

	_, img = cap.read()
	cv2.imwrite(out_path,img)

	cnt += 1;

	if cnt == frame_cnt or cnt == 1200:
		break;
