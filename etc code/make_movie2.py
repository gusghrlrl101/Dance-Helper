import cv2
import os
from glob import glob

# image
img_paths = sorted(glob('*[0-9].png'))
img_paths_original = sorted(glob('img/*[0-9].png'))

img = cv2.imread(img_paths[0])
rows, cols, channels = img.shape

# codec
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result.avi', fourcc, 10.0, (cols, rows))

for ind, img_path in enumerate(img_paths):
	img = cv2.imread(img_path)

	roi = img_paths_original[ind][:rows, :cols]

	img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
	mask2 = cv2.bitwise_not(mask)

	bg = cv2.bitwise_and(roi, roi, mask=mask)
	fg = cv2.bitwise_and(img, img, mask=mask2)

	dst = cv2.add(bg, fg)

	out.write(dst)
out.release()
