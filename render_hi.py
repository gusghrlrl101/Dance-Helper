# -*- Encoding:UTF-8 -*- #

import sys
import os
from glob import glob
import cPickle as pickle
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
from matplotlib import pyplot
import cv2
import math

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}

rotx=0.0
roty=0.0
rotz=0.0
rotx2=0.0
roty2=0.0
rotz2=0.0
path1 = sys.argv[1]
path2 = sys.argv[2]

op_pickle_path = (path1+'/hyunho.pkl')
op_pickle_path2 = (path2+'/hyunho.pkl')
with open(op_pickle_path,'rb') as f3:
	op_datas = pickle.load(f3)
	op_joints = op_datas['op_joints']
with open(op_pickle_path2, 'rb') as f4:
	op_datas2 = pickle.load(f4)
	op_joints2 = op_datas2['op_joints']

def _create_renderer(w=640,
                     h=480,
                     rt=np.array([rotx,roty,rotz]),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, rt, t, f, near=0.5, far=100, img=None):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=rt, t=t, f=f)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img


    imtmp = simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp

def similarity(res, res2, ind, ind2, img2): # ind는 model 인덱스, ind2는 compare 인덱스
	result=0
	indiSimil=np.zeros(12)
	indiResult=np.zeros(12)
	index=0 # 관절별 인덱스
	#vectors = [60, 54, 48, 63, 57, 51, 21, 12, 3, 24, 15, 6]
	vectors = [24,15,6,3,12,21,63,57,51,48,54,60]
	distance = [int(op_joints2[ind2][13][0]-op_joints[ind][13][0]),int(op_joints2[ind2][13][1]-op_joints[ind][13][1])]
	for vector in vectors:
		for i in range(3):
			indiResult[index]+=abs(res['pose'][vector+i]-res2['pose'][vector+i])
		indiSimil[index]=1 / (1+indiResult[index])
		result+=indiResult[index]
		indiSimil[index]*=100
		if((indiSimil[index]<80 and index >=7)or(indiSimil[index]<60)):
			cv2.arrowedLine(img2,(int(op_joints2[ind2][index][0]),int(op_joints2[ind2][index][1])),(int(op_joints[ind][index][0])+distance[0],int(op_joints[ind][index][1])+distance[1]),(0,0,128),1)
			cv2.circle(img2,(int(op_joints2[ind2][index][0]),int(op_joints2[ind2][index][1])),10,(0,0,255))
			print(str(vector)+"wrong")
		index+=1

	similarity = sum(indiSimil)
	similarity /= len(indiSimil)
	print(str(similarity)+"%")

def similarity2(res, res2):
	vectors = [[60, 54], [54, 48], [63, 57], [57, 51], [21, 12], [12, 3], [24, 15], [15,6]]
	similarity = 0.0
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
		result.append(1 - rad / math.pi)
		print(str(1-rad/math.pi))
	similarity = sum(result) / len(result) * 100
	print(str(similarity)+"%")


## Load SMPL model (here we load the female model)
m = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
m2 = load_model('./models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

## Assign attributes to renderer
w, h = (1280, 720)
color = colors['pink']

# calculate average body parameter
body = np.zeros(10)
body2 = np.zeros(10)
bodyIndex = np.zeros([10,20])
bodyIndex2 = np.zeros([10,20])
bodyTemp = np.zeros([10,20])
bodyTemp2 = np.zeros([10,20])
pkl_paths = sorted(glob(path1+'/*[0-9].pkl'))
pkl_paths2 = sorted(glob(path2+'/*[0-9].pkl'))

for ind, pkl_path in enumerate(pkl_paths):
	with open(pkl_path,'r') as f:
		res = pickle.load(f)
	for i in range(10):
		for j in range(20):
			if res['betas'][i] >= j-10 and res['betas'][i] < j-9:
				bodyIndex[i][j]+=1
				bodyTemp[i][j]+=res['betas'][i]
for ind, pkl_path2 in enumerate(pkl_paths2):
	with open(pkl_path2,'r') as f2:
		res2 = pickle.load(f2)
	for i in range(10):
		for j in range(20):
			if res2['betas'][i] >= j-10 and res2['betas'][i] < j-9:
				bodyIndex2[i][j]+=1
				bodyTemp2[i][j]+=res2['betas'][i]

indexMax=0
indexMax2=0
valueMax=0.0
valueMax2=0.0

for i in range(10):
	for j in range(20):
		if bodyIndex[i][j] > indexMax:
			indexMax = bodyIndex[i][j]
			valueMax = bodyTemp[i][j]
		if bodyIndex2[i][j] > indexMax2:
			indexMax2 = bodyIndex2[i][j]
			valueMax2 = bodyTemp2[i][j]
	body[i] = valueMax/(indexMax+sys.float_info.epsilon)
	body2[i] = valueMax2/(indexMax2+sys.float_info.epsilon)
	indexMax=0
	indexMax2=0
m.betas[:] = body[:]
m2.betas[:] = body2[:]


ind=0
ind2=0
sim = False
# render model
while True:
	with open(pkl_paths[ind],'r') as f:
		res = pickle.load(f)
	with open(pkl_paths2[ind2],'r') as f2:
		res2 = pickle.load(f2)
	print ind
	print ind2
	print pkl_paths[ind]
	print pkl_paths2[ind2]
	ff = res['f']
	tt = res['cam_t']
	m.pose[:] = res['pose']
	ff2 = res2['f']
	tt2 = res2['cam_t']
	m2.pose[:] = res2['pose']
	
	while True :
		if sim==False:
			img = render_model(m, m.f, w, h, np.array([rotx,roty,rotz]), tt, ff)
			img2 = render_model(m2, m2.f, w, h, np.array([rotx2,roty2,rotz2]), tt2, ff2)
		## Show it using OpenCV
			resize(img,img,Size(320,640))
			resize(img2,img2,Size(320,640))
			cv2.imshow('origin', img)
			cv2.imshow('compare', img2)

			print img
			print img2
		sim=False
		
		k = cv2.waitKey(0)
		if k == 27: # ESC
			break
		elif k == ord('a'): # 원본 왼쪽
			print("left")
			roty += 0.1
		elif k == ord('d'): # 원본 오른쪽
			print("right")
			roty -= 0.1
		elif k == ord('w'): # 원본 위로
			print("up")
			rotx -= 0.1
		elif k == ord('s'): # 원본 아래로
			print("down")
			rotx += 0.1
		elif k == ord('q'): # 원본 왼쪽 대각선
			print("left plane")
			rotz -= 0.1
		elif k == ord('e'): # 원본 오른쪽 대각선
			print("right plane")
			rotz += 0.1
		elif k == ord('r'): # 둘다 reset
			print("reset")
			rotx2 = roty2 = rotz2 = rotx = roty = rotz = 0.0
		elif k == ord('b'): # 둘다 back
			print("back")
			roty += np.pi
			roty2 += np.pi
		elif k == ord('j'):
			print("left2")
			roty2 += 0.1
		elif k == ord('l'):
			print("right2")
			roty2 -= 0.1
		elif k == ord('i'):
			print("up2")
			rotx2 -= 0.1
		elif k == ord('k'):
			print("down2")
			rotx2 += 0.1
		elif k == ord('u'):
			print("left plane2")
			rotz2 -= 0.1
		elif k == ord('o'):
			print("right plane2")
			rotz2 += 0.1
		elif k == ord('z'): # 원본 이전 프레임
			print("original prev")
			if ind > 0: # 0보다 안작아지게
				ind -= 1
			break
		elif k == ord('x'): # 원본 다음 프레임
			print("original next")
			if ind < len(pkl_paths): # 원본 프레임 수보다 안커지게
				ind += 1
			break
		elif k == ord('n'): # 비교 이전 프레임
			print("compare prev")
			if ind2 > 0: # 0보다 안작아지게
				ind2 -= 1
			break
		elif k == ord('m'): # 비교 다음 프레임
			print("compare next")
			if ind2 < len(pkl_paths2): # 비교 프레임 수보다 안커지게
				ind2 += 1
			break
		elif k == ord('y'): # 유사도 분석
			print("similarity")
			similarity(res,res2,ind,ind2,img2)
			sim=True
			cv2.imshow("compare",img2)
		else:
			ind += 1
			ind2 += 1
			break
	if k == 27:
		cv2.destroyAllWindows()
		break


		
