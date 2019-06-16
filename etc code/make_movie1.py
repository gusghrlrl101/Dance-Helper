import cv2
import os
from glob import glob

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result1.avi', fourcc, 24.0, (1280, 720))
img_paths = sorted(glob('*[0-9].png'))
for ind, img_path in enumerate(img_paths):
	img = cv2.imread(img_path)
	out.write(img)
out.release()
