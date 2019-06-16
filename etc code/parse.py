# -*- Encoding:UTF-8 -*- #
import pickle
import sys
import json
import os

# use -> python similarity.py 0000 0000

#TODO 1 : adjust pkl output format
#TODO 2 : adapt weight for each position

#file isn't exist
if(os.path.isfile('./'+str(sys.argv[1])+'parse.pkl')==False):
    with open('./'+str(sys.argv[1])+'.pkl','rb') as f:
        data = pickle.load(f)
        # file 1 write
        f2 = open('./'+str(sys.argv[1])+'parse.pkl', 'wb')
	line=str(data)
	result=''
	cam3='{"cam_t": '
	cam=line.partition("'cam_t':")[2]
	cam2=cam.partition("])")[0]
	cam3+=cam2.partition("array(")[2]
	cam3+="]"
	pose3=', "poses": '
	pose=line.partition("'pose':")[2]
	pose2=pose.partition("])")[0]
	pose3+=pose2.partition("array(")[2]
	pose3+="]"
	betas3=', "betas": '
	betas=line.partition("'betas':")[2]
	betas2=betas.partition("])")[0]
	betas3+=betas2.partition("array(")[2]
	betas3+="]"
	f3=', "f": '
	f=line.partition("'f':")[2]
	ff2=f.partition("])")[0]
	f3+=ff2.partition("array(")[2]
	f3+="]}"

	result=str(cam3)+str(pose3)+str(betas3)+str(f3)
#        f2.write(str(result))
	pickle.dump(result, f2)
	print(result)
 #       f2.close()

