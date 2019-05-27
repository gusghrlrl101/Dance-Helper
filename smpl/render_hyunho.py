'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''
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


colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}

rotx=0.0
roty=0.0
rotz=0.0




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


def render_model(verts, faces, w, h, rt, t, f, near=0.5, far=25, img=None):
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



## Load SMPL model (here we load the female model)
m = load_model('../models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

## Assign attributes to renderer
w, h = (1280, 720)
color = colors['pink']

# open conf pkl
#with open('hyunho.pkl','r') as f:
#	res = pickle.load(f)

# calculate average body parameter
body = [0 for _ in range(10)]
pkl_paths = sorted(glob('*[0-9].pkl'))
for ind, pkl_path in enumerate(pkl_paths):
	with open(pkl_path,'r') as f:
		res = pickle.load(f)
	body += res['betas']
body /= 1200
m.betas[:] = body

# render model
for ind, pkl_path in enumerate(pkl_paths):
	print ind
	print pkl_path
	with open(pkl_path,'r') as f:
		res = pickle.load(f)
	ff = res['f']
	tt = res['cam_t']
	m.pose[:] = res['pose']
	while True :
		img = render_model(m, m.f, w, h, np.array([rotx,roty,rotz]), tt, ff)

	## Show it using OpenCV
		cv2.imshow('render_SMPL', img)
		k = cv2.waitKey(0)
		if k == 27:
			break
		elif k == ord('a'):
			print("left")
			roty += 0.1
		elif k == ord('d'):
			print("right")
			roty -= 0.1
		elif k == ord('w'):
			print("up")
			rotx -= 0.1
		elif k == ord('s'):
			print("down")
			rotx += 0.1
		elif k == ord('q'):
			print("left plane")
			rotz -= 0.1
		elif k == ord('e'):
			print("right plane")
			rotz += 0.1
		elif k == ord('r'):
			print("reset")
			rotx = roty = rotz = 0.0
		elif k == ord('b'):
			print("back")
			roty += np.pi
		else:
			break
	if k == 27:
		cv2.destroyAllWindows()
		break
		
