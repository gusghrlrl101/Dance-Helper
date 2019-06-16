import cv2
import os
from glob import glob

# codec
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('result2.avi', fourcc, 24.0, (1280, 720))

# video
#cap = cv2.VideoCapture('../../../image/user4.mp4')

# image
i = 0
img_paths = sorted(glob('*[0-9].png'))
img_paths_original = sorted(glob('img/*[0-9].png'))
for ind, img_path in enumerate(img_paths):
	img = cv2.imread(img_path)

	rows, cols, channels = img.shape
	temp = cv2.imread(img_paths_original[ind])
	roi = temp[:rows, :cols]

	img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
	mask2 = cv2.bitwise_not(mask)

	bg = cv2.bitwise_and(roi, roi, mask=mask)
	fg = cv2.bitwise_and(img, img, mask=mask2)

	dst = cv2.add(bg, fg)
	cv2.imwrite("img_after/" + str(i) + ".png", dst)
	i += 1
#	out.write(dst)
#out.release()
