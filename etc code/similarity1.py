# -*- Encoding:UTF-8 -*- #
import pickle
import sys
import json
import os

# use -> python similarity.py 0000 0000

#TODO 1 : adjust pkl output format
#TODO 2 : adapt weight for each position

#file isn't exist
if(os.path.isfile('./'+str(sys.argv[1])+'pose.txt')==False):
    with open('./'+str(sys.argv[1])+'.pkl','rb') as f:
        data = pickle.load(f)
        # file 1 write
        f2 = open('./'+str(sys.argv[1])+'pose.txt', 'w')
	line=str(data)
	pose3='{"pose": '
	pose=line.partition("'pose':")[2]
	pose2=pose.partition("])")[0]
	pose3+=pose2.partition("array(")[2]
	pose3+="]}"
        f2.write(str(pose3))
        f2.close()
if(os.path.isfile('./'+str(sys.argv[2])+'pose.txt')==False):
    with open('./'+str(sys.argv[2])+'.pkl', 'rb') as f3:
        data2 = pickle.load(f3)
        # file 2 write
	line2=str(data2)
	pose6='{"pose": '
	pose4=line2.partition("'pose':")[2]
	pose5=pose4.partition("])")[0]
	pose6+=pose5.partition("array(")[2]
	pose6+="]}"

        f4 = open('./'+str(sys.argv[2])+'pose.txt', 'w')
        f4.write(str(pose6))
        f4.close()

# parsing
with open('./'+str(sys.argv[1])+'pose.txt') as json_file:
    json_data = json.load(json_file)
    json_string = json_data["pose"]

with open('./'+str(sys.argv[2])+'pose.txt', 'r') as json_file2:
    json_data2 = json.load(json_file2)
    json_string2 = json_data2["pose"]
result=0
indiResult=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
indiSimil=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
indiInfo=['Hip L', 'Hip R', 'Lower Spine', 'Knee L', 'Knee R', 'Mid Spine', 'Ankle L', 'Ankle R', 'Upper Spine', 'Toes L', 'Toes R', 'Neck', 'Clavicle L', 'Clavicle R', 'Head',
'Shoulder L', 'Shoulder R', 'Elbow L', 'Elbow R', 'Wrist L', 'Wrist R']

for i in range(3, 62):
    result+=abs(json_string[i]-json_string2[i])
for i in range(1, 21):
    indiX = abs(json_string[i*3]-json_string2[i*3])
    indiY = abs(json_string[i*3+1]-json_string2[i*3+1])
    indiZ = abs(json_string[i*3+1]-json_string2[i*3+1])
    indiResult[i]+=(indiX+indiY+indiZ)
    indiSimil[i] = 1 / (1+indiResult[i])
    indiSimil[i] *= 100
    print(str(indiInfo[i])+" : "+str(indiSimil[i])+"%")
    if(indiSimil[i]<80):
        print(str(indiInfo[i])+"가 틀렸네요.")
        if(json_string2[i*3]-json_string[i*3]>0):
            print(str(indiInfo[i])+"를 왼쪽으로 옮기세요.")
        else:
            print(str(indiInfo[i])+"를 오른쪽으로 옮기세요.")
        if(json_string2[i*3+1]-json_string[i*3]>0):
            print(str(indiInfo[i])+"를 아래쪽으로 옮기세요.")
        else:
            print(str(indiInfo[i])+"를 위쪽으로 옮기세요.")
        if(json_string2[i*3+2]-json_string[i*3+2]>0):
            print(str(indiInfo[i])+"를 뒤쪽으로 옮기세요.")
        else:
            print(str(indiInfo[i])+"를 앞쪽으로 옮기세요.")
    
similarity = 1 / (1+result)
similarity *= 100
print(str(similarity)+"%")
import cv2
source = cv2.imread("./"+str(sys.argv[1])+".png")
compare = cv2.imread("./"+str(sys.argv[2])+".png");
cv2.imshow("source", source)
cv2.imshow("compare", compare)

cv2.waitKey(0)
