# -*- Encoding:UTF-8 -*- #
import sys
import os
from glob import glob
import cPickle as pickle
import numpy as np
import math

path1 = sys.argv[1]
path2 = sys.argv[2]

body = np.zeros(10)
bodyIndex = np.zeros([10,20]) # -10~-9부터 9~10 구간 20개
bodyTemp = np.full([10,20], 0.0)
 

pkl_paths = sorted(glob('./opmlify/result/'+path1+'/*[0-9].pkl'))
pkl_paths2 = sorted(glob('./opmlify/result/'+path2+'/*[0-9].pkl'))
# -10~-9, -9~-9, ... 9~10 20구간으로 나눔
for ind, pkl_path in enumerate(pkl_paths):
	with open(pkl_path,'r') as f:
		res = pickle.load(f)
	#body += res['betas']
	for i in range(10):
#		if res['betas'][i] > bodyMax[i]:
#			bodyMax[i] = res['betas'][i]
#		elif res['betas'][i] < bodyMin[i]:
#			bodyMin[i] = res['betas'][i]
		for j in range(20):
			if res['betas'][i] >= j-10 and res['betas'][i] < j-9:
				bodyIndex[i][j]+=1
				bodyTemp[i][j]+=res['betas'][i]
indexMax=0
valueMax=0.0

for i in range(10):
	for j in range(20):
		if bodyIndex[i][j] > indexMax:
			indexMax = bodyIndex[i][j]
			valueMax = bodyTemp[i][j]
	print(indexMax)
	body[i]=valueMax/indexMax
	indexMax=0
print(str(body))
