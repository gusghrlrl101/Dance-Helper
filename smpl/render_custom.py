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
import cPickle as pickle
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
from matplotlib import pyplot

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9,.9,.8],
    'capsule': [.7,.75,.5],
    'yellow': [.5, .7, .75],
}

## Load SMPL model (here we load the female model)
#m = load_model('../models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
#m = load_model('../opmlify/result/0021.pkl')
m = load_model('../opmlify/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

## Assign random pose and shape parameters
with open('../opmlify/result/'+str(sys.argv[1])+'.pkl','r') as f:
    res = pickle.load(f)
#m.pose[:] = np.random.rand(m.pose.size) * .2
#m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose[:] = res['pose']
m.betas[:] = res['betas']
m.pose[0] = np.pi

## Create OpenDR renderer
rn = ColoredRenderer()

rotx=0.0
roty=0.0
rotz=0.0

## Assign attributes to renderer
w, h = (640, 480)
color = colors['pink']
rn.camera = ProjectPoints(v=m, rt=np.array([rotx,roty,rotz]), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5)) # rt is parameter for rotate
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, vc=color, bgcolor=np.ones(3))
albedo = rn.vc
# Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=albedo,
    light_color=np.array([1., 1., 1.]))


## Show it using OpenCV
import cv2
cv2.imshow('render_SMPL', rn.r)
#img = cv2.cvtColor(rn.r, cv2.CV_32FC3, 255)
#img = rn.v.astype('uint8')
#cv2.imwrite('./render/'+str(sys.argv[1])+'.png', img)
pyplot.imsave('./render/'+str(sys.argv[1])+'.png',rn.r)
print ('..Press ESC key for exit')
k = cv2.waitKey(0)
if k == 27:
	cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()