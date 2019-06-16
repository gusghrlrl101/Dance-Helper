# -*- Encoding:UTF-8 -*- #
import pickle
import sys
import json
import os
from glob import glob

num = []
txt_paths = sorted(glob.glob('*[0-9].pkl'))
for ind, txt_path in enumerate(txt_paths):
	#file isn't exist
	if(os.path.isfile(txt_path[:-4] + '.txt') == False):
		with open(txt_path,'rb') as f:
			data = pickle.load(f)
			# file 1 write
			f2 = open(txt_path[:-4] + '.txt', 'w')
			line=str(data)
			pose3='{"pose": '
			pose=line.partition("'pose':")[2]
			pose2=pose.partition("])")[0]
			pose3+=pose2.partition("array(")[2]
			pose3+="]}"
			f2.write(str(pose3))
			f2.close()

	# parsing
	with open(txt_path[:-4] + '.txt') as json_file:
		json_data = json.load(json_file)
		json_string = json_data["pose"]
		num.append(int(json_string[0]))

print (num)
