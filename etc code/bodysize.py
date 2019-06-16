# -*- Encoding:UTF-8 -*- #
import pickle
import sys
import json
import os

# use -> python bodysize.py 0000


#file isn't exist
if(os.path.isfile('./'+str(sys.argv[1])+'body.txt')==False):
    with open('./'+str(sys.argv[1])+'.pkl','rb') as f:
        data = pickle.load(f)
	print(1)
	print(str(data))
        # file 1 write
        f2 = open('./'+str(sys.argv[1])+'body.txt', 'w')
	line=str(data)
	print(line)
	body3='{"betas": '
	body=line.partition("'betas':")[2]
	body2=body.partition("])")[0]
	body3+=body2.partition("array(")[2]
	body3+="]}"
	print(str(body3))
        f2.write(str(body3))
        f2.close()
# parsing
with open('./'+str(sys.argv[1])+'body.txt') as json_file:
	json_data = json.load(json_file)
	json_string = json_data["betas"]

print(str(json_string))


