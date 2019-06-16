import cv2

cap = cv2.VideoCapture('model2.mp4')
cap.set(cv2.CAP_PROP_FPS, 12)

video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('model2_out.avi', fourcc, 12.0, (video_w, video_h))

cnt = 0
while cap.isOpened():
	print cnt
	_, img = cap.read()
	cv2.imshow("img", img)
	cv2.waitKey(0)

	cnt += 1
