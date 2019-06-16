# -*- Encoding:UTF-8 -*- #
import cPickle as pickle
import numpy as np
import sys
import os
from glob import glob
import cv2
import math
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
	print "ERROR: input path1, path2"
	exit()

path1 = sys.argv[1]
path2 = sys.argv[2]

pkl_paths = sorted(glob('result/' + path1 + '/*[0-9].pkl'))
pkl_paths2 = sorted(glob('result/' + path2 + '/*[0-9].pkl'))

total = 0.0
count = 0

x = []
ls = []
for ind, (pkl_path, pkl_path2) in enumerate(zip(pkl_paths, pkl_paths2)):
	with open(pkl_path,'r') as f:
		res = pickle.load(f)
	with open(pkl_path2,'r') as f2:
		res2 = pickle.load(f2)

	vectors = [[60, 54], [54, 48], [63, 57], [57, 51], [21, 12], [12, 3], [24, 15], [15, 6]]
	result = []
	for vector in vectors:
		v1 = []
		v2 = []
		for i in range(3):
			t1 = res['pose'][vector[0] + i] - res['pose'][vector[1] + i]
			t2 = res2['pose'][vector[0] + i] - res2['pose'][vector[1] + i]
			v1.append(t1)
			v2.append(t2)
		np_v1 = np.array(v1)
		np_v2 = np.array(v2)
		cos_theta = sum(np_v1 * np_v2) / math.sqrt(sum(np_v1 ** 2) * sum(np_v2 ** 2))
		rad = abs(math.acos(cos_theta))
		if rad > math.pi:
			rad -= math.pi
		result.append(1 - 5 * rad / math.pi)
		#result.append(1 / (1 + rad / math.pi))
	similarity = sum(result) / len(result) * 100
#	print(str(similarity)+"%")
	ls.append(similarity)
	x.append(ind)
	"""
	if ind == 1020:
		source = cv2.imread("./result/"+path1+"/"+str(ind)+".png")
		compare = cv2.imread("./result/"+path2+"/"+str(ind-17)+".png")
		cv2.imshow("source", source)
		cv2.imshow("compare", compare)
		cv2.waitKey(0)
	"""

print "similarity:", sum(ls) / len(ls), "%"

"""
# plot
plt.plot(x, ls, '-r')
plt.show()
"""
