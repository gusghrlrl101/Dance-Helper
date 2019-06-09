"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

About this Script:
============
This is a demo version of the algorithm implemented in the paper,
which fits the SMPL body model to the image given the joint detections.
The code is organized to be run on the LSP dataset.
See README to see how to download images and the detected joints.
"""

from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch
import pyopenpose as op

from multiprocessing import Process

from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
from render_model import render_model

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added
POSE = ["R ankle", "R knee", "R hip", "L hip", "L knee", "L ankle", "R wrist", "R elbow", "R shoulder", "L shoulder", "L elbow", "L wrist", "Neck", "head"]

# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d, init_pose):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=5000.,
                      pix_thsh=25.):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image 
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be considered,
              3D vector for the body orientation
    """
    # optimize camera translation and body orientation based on torso joints
    # LSP torso ids:
    # 2=right hip, 3=left hip, 8=right shoulder, 9=left shoulder
    torso_cids = [2, 3, 8, 9]
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    # initialize camera rotation
    rt = ch.zeros(3)
    # initialize camera translation
    print 'initializing translation via similar triangles'
    init_t = guess_init(model, flength, j2d, init_pose)
    t = ch.array(init_t)

    # check how close the shoulder joints are
    try_both_orient = np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # initialize the camera
    cam = ProjectPoints(
        f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)

    # we are going to project the SMPL joints
    cam.v = Jtr
    on_step = None

    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        # maxiter: 100, e_3: .0001
        options={'maxiter': 100,
                 'e_3': .0001,
                 # disp set to 1 enables verbose output from the optimizer
                 'disp': 0})

    return (cam, try_both_orient, opt_pose[:3].r)


# --------------------Core optimization --------------------
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       try_both_orient,
                       body_orient,
                       n_betas=10,
                       regs=None,
                       conf=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image 
    :param prior: mixture of gaussians pose prior
    :param try_both_orient: boolean, if True both body_orient and its flip are considered for the fit
    :param body_orient: 3D vector, initialization for the body orientation
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 14D vector storing the confidence values from the CNN
    :returns: a tuple containing the optimized model, its joints projected on image space, the camera translation
    """
    t0 = time()
    # define the mapping LSP joints -> SMPL joints
    # cids are joints ids for LSP:
    cids = range(12) + [13]
    # joint ids for SMPL
    # SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    # the vertex id for the joint corresponding to the head
    head_id = 411

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and LSP is significantly different so set
    # their weights to zero
    base_weights = np.array(
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    if try_both_orient:
        flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
            cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
        flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
        orientations = [body_orient, flipped_orient]
    else:
        orientations = [body_orient]

    if try_both_orient:
        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        errors = []

    svs = []
    cams = []
    for o_id, orient in enumerate(orientations):
        # initialize the shape to the mean shape in the SMPL training set
        betas = ch.zeros(n_betas)

        # initialize the pose by using the optimized body orientation and the
        # pose prior
        init_pose = np.hstack((orient, prior.weights.dot(prior.means)))

        # instantiate the model:
        # verts_decorated allows us to define how many
        # shape coefficients (directions) we want to consider (here, n_betas)
        sv = verts_decorated(
            trans=ch.zeros(3),
            pose=ch.array(init_pose),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs)

        # make the SMPL joints depend on betas
        Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                           for i in range(len(betas))])
        J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
            model.v_template.r)

        # get joint positions as a function of model pose, betas and trans
        (_, A_global) = global_rigid_transformation(
            sv.pose, J_onbetas, model.kintree_table, xp=ch)
        Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

        # add the head joint, corresponding to a vertex...
        Jtr = ch.vstack((Jtr, sv[head_id]))

        # ... and add the joint id to the list
        if o_id == 0:
            smpl_ids.append(len(Jtr) - 1)

        # update the weights using confidence values
        weights = base_weights * conf[
            cids] if conf is not None else base_weights

        # project SMPL joints on the image plane using the estimated camera
        cam.v = Jtr

        # data term: distance between observed and estimated joints in 2D
        obj_j2d = lambda w, sigma: (
            w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))

        # mixture of gaussians pose prior
        pprior = lambda w: w * prior(sv.pose)
        # joint angles pose prior, defined over a subset of pose parameters:
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        alpha = 10
        my_exp = lambda x: alpha * ch.exp(x)
        obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
                                                 58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

        on_step = None

        if regs is not None:
            # interpenetration term
            sp = SphereCollisions(
                pose=sv.pose, betas=sv.betas, model=model, regs=regs)
            sp.no_hands = True
        # weight configuration used in the paper, with joints + confidence values from the CNN
        # (all the weights used in the code were obtained via grid search, see the paper for more details)
        # the first list contains the weights for the pose priors,
        # the second list contains the weights for the shape prior
        opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                          [1e2, 5 * 1e1, 1e1, .5 * 1e1])

        # run the optimization in 4 stages, progressively decreasing the
        # weights for the priors
        for stage, (w, wbetas) in enumerate(opt_weights):
            print 'stage', stage
            objs = {}

            objs['j2d'] = obj_j2d(1., 100)

            objs['pose'] = pprior(w)

            objs['pose_exp'] = obj_angle(0.317 * w)

            objs['betas'] = wbetas * betas

            if regs is not None:
                objs['sph_coll'] = 1e3 * sp

            # HYUNHO
            start = time()
            # maxiter: 100, e_3: .0001
            ch.minimize(objs,
                x0=[sv.betas, sv.pose],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100,
                         'e_3': .0001,
                         'disp': 0})
            print time()-start

        t1 = time()
        print 'elapsed ', str(t1 - t0)
        if try_both_orient:
            errors.append((objs['j2d'].r**2).sum())
        svs.append(sv)
        cams.append(cam)

    if try_both_orient and errors[0] > errors[1]:
        choose_id = 1
    else:
        choose_id = 0

    return (svs[choose_id], cams[choose_id].r, cams[choose_id].t.r)


def run_single_fit(img,
                   j2d,
                   conf,
                   model,
                   regs=None,
                   n_betas=10,
                   flength=5000.,
                   pix_thsh=25.,
                   scale_factor=1,
                   do_degrees=None):
    """Run the fit for one specific image.
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """
    if do_degrees is None:
        do_degrees = []

    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))

    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor

    # estimate the camera parameters
    (cam, try_both_orient, body_orient) = initialize_camera(
        model,
        j2d,
        img,
        init_pose,
        flength=flength,
        pix_thsh=pix_thsh)

    # fit
    (sv, opt_j2d, t) = optimize_on_joints(
        j2d,
        model,
        cam,
        img,
        prior,
        try_both_orient,
        body_orient,
        n_betas=n_betas,
        conf=conf,
        regs=regs, )

    h = img.shape[0]
    w = img.shape[1]
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])

    images = []
    orig_v = sv.r
    
    for deg in do_degrees:
        if deg != 0:
            aroundy = cv2.Rodrigues(np.array([0, np.radians(deg), 0]))[0]
            center = orig_v.mean(axis=0)
            new_v = np.dot((orig_v - center), aroundy)
            verts = new_v + center
        else:
            verts = orig_v

        # now render
        im = (render_model(
            verts, model.f, w, h, cam, far=20 + dist) * 255.).astype('uint8')
        images.append(im)

    # return fit parameters
    params = {'cam_t': cam.t.r,
              'f': cam.f.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r,
              'conf': conf}

    return params, images


def openPose(img, gamma=1.0, ui=False):
    params = dict()
    model_folder = ('' if ui else '../') + 'models'
    params["model_folder"] = model_folder
    params["number_people_max"] = 1;

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    # hyunho gamma
    if gamma != 1.0:
        img2 = adjust_gamma(img, gamma=gamma)
        datum.cvInputData = img2
    else:
        datum.cvInputData = img

    opWrapper.emplaceAndPop([datum])

    mydata = datum.poseKeypoints[0]
    joints = np.zeros((14,2))
    conf = np.zeros(14)

    joints[0] = mydata[11, :2]
    joints[1] = mydata[10, :2]
    joints[2] = mydata[9, :2]
    joints[3] = mydata[12, :2]
    joints[4] = mydata[13, :2]
    joints[5] = mydata[14, :2]
    joints[6] = mydata[4, :2]
    joints[7] = mydata[3, :2]
    joints[8] = mydata[2, :2]
    joints[9] = mydata[5, :2]
    joints[10] = mydata[6, :2]
    joints[11] = mydata[7, :2]
    joints[12] = mydata[1, :2]
    joints[13] = mydata[0, :2]

    conf[0] = mydata[11, 2]
    conf[1] = mydata[10, 2]
    conf[2] = mydata[9, 2]
    conf[3] = mydata[12, 2]
    conf[4] = mydata[13, 2]
    conf[5] = mydata[14, 2]
    conf[6] = mydata[4, 2]
    conf[7] = mydata[3, 2]
    conf[8] = mydata[2, 2]
    conf[9] = mydata[5, 2]
    conf[10] = mydata[6, 2]
    conf[11] = mydata[7, 2]
    conf[12] = mydata[1, 2]
    conf[13] = mydata[0, 2]

    # head joint estimate
    # joints[13] += 0.5 * (joints[13] - joints[12])

    return joints, conf


def smplify(img, out_path, out_video_dir, op_joints, op_confs, cnt, genders, model_female, model_male, sph_regs_female, sph_regs_male, n_betas, flength, pix_thsh, do_degrees):
        print out_path
        if exists(out_path):
            out_path = out_path[:-4] + '_new.pkl'

        print 'Fitting 3D body saving to', out_path

        if img.ndim == 2:
            print "The image is grayscale!"
            img = np.dstack((img, img, img))

        # openpose
        joints, conf = op_joints[cnt], op_confs[cnt]

        # decide gender
        gender = 'male' if int(genders[cnt]) == 0 else 'female'
        if gender == 'female':
            model = model_female
            sph_regs = sph_regs_female
        elif gender == 'male':
            model = model_male
            sph_regs = sph_regs_male

        # female 
        model = model_female
        sph_regs = sph_regs_female

        # run
        params, vis = run_single_fit(
            img,
            joints,
            conf,
            model,
            regs=sph_regs,
            n_betas=n_betas,
            flength=flength,
            pix_thsh=pix_thsh,
            scale_factor=1,
            do_degrees=do_degrees)

        with open(out_path, 'w') as outf:
            params['op_joints'] = joints
            pickle.dump(params, outf)

        # This only saves the first rendering.
        if do_degrees is not None:
            cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])
#            out.write(vis[0])

def getFrames(temps, frame_cnt, ratio):
    frames = []
    cnt = 0
    if ratio > 1.0:
        for i, temp in enumerate(temps):
            if cnt < int(i * ratio):
                for _ in range(cnt, int(i * ratio)):
                    frames.append(temps[i - 1])
                    cnt += 1

                    if cnt == frame_cnt:
                        break

            if cnt == frame_cnt:
                break

            frames.append(temp)
            cnt += 1
    elif ratio < 1.0:
        for i, temp in enumerate(temps):
            if cnt > int(i * ratio):
                continue
                    
            frames.append(temp)
            cnt += 1
                    
            if cnt == frame_cnt:
                break
    else:
        frames = temps[:frame_cnt]
    return frames


def mmain(video="False",
         ui=False,
         female=True,
         num=-1,
         gamma=1.0):
    n_betas=10
    flength=5000.
    pix_thsh=25.

    # Set up paths & load models.
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')

    # Model paths:
    MODEL_NEUTRAL_PATH = join(MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    # paths to the npz files storing the regressors for capsules
    SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR, 'regressors_locked_normalized_hybrid.npz')
    SPH_REGS_FEMALE_PATH = join(MODEL_DIR, 'regressors_locked_normalized_female.npz')
    SPH_REGS_MALE_PATH = join(MODEL_DIR, 'regressors_locked_normalized_male.npz')

    img_dir = ('' if ui else '../') + 'image'
    data_dir = ('opmlify/' if ui else '') + 'results/lsp'
    out_dir = ('opmlify/' if ui else '') + 'result/'
    csv_dir = ('opmlify/' if ui else '') + 'lsp/lsp_gender.csv'
    est_dir = ('opmlify/' if ui else '') + 'lsp/est_joints.npz'

    if not exists(out_dir):
        makedirs(out_dir)

    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    sph_regs = None
    # File storing information about gender in LSP
    with open(csv_dir) as f:
        genders = f.readlines()
    model_female = load_model(MODEL_FEMALE_PATH)
    model_male = load_model(MODEL_MALE_PATH)
    sph_regs_male = np.load(SPH_REGS_MALE_PATH)
    sph_regs_female = np.load(SPH_REGS_FEMALE_PATH)

    # Load joints
    est = np.load(est_dir)['est_joints']

    # hyunho
    # Load video
    if video != "False":
        out_video_dir = out_dir + video[:-4]

        if not exists (out_video_dir):
            makedirs (out_video_dir)

        # first, openpose
        cap = cv2.VideoCapture(join(img_dir, video))

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ratio = 10.0 / fps

        # only 60 sec
        if frame_cnt > 60.0 * fps:
            frame_cnt = int(60.0 * fps)

        if ratio > 1.0:
            frame_cnt = int(frame_cnt * ratio)

        # get captures
        temps = []
        cnt = 0
        while cap.isOpened():
            _, img = cap.read()
            temps.append(img)

            cnt += 1
            if cnt == frame_cnt:
                break
        cap.release()

        # get frames
        frames = getFrames(temps, frame_cnt, ratio)

        if not exists(out_video_dir + '/img'):
            makedirs(out_video_dir + '/img')

        for cnt, img in enumerate(frames):
            if cnt <= num:
                continue
            out_path = '%s/%04d.png' % (out_video_dir + '/img', cnt)
            cv2.imwrite(out_path, img)
        return
        
        

        # make openpose pkl
        op_pickle_path = join(out_video_dir, 'hyunho.pkl')
        if not exists(op_pickle_path):
            temp_joints = []
            temp_confs = []

            # get openpose result
            for i, frame in enumerate(frames):
                print i
                temp_joint, temp_conf = openPose(frame, gamma, ui)
                temp_joints.append(temp_joint)
                temp_confs.append(temp_conf)

            # save to 'hyunho.pkl'
            with open(op_pickle_path, 'w') as outf:
                temp_params = dict()
                temp_params['op_joints'] = temp_joints
                temp_params['conf'] = temp_confs
                pickle.dump(temp_params, outf)

        # open openpose pkl
        with open(op_pickle_path, "rb") as f:
            op_datas = pickle.load(f)
            op_joints = op_datas['op_joints']
            op_confs = op_datas['conf']

        # view
        op_view = False
        if op_view:
            for cnt, img in enumerate(frames):
                for i, j in enumerate(op_joints[cnt]):
                    if op_confs[cnt][i] < 0.33:
                        temp_color = (0, 0, 255)
                    elif op_confs[cnt][i] > 0.66:
                        temp_color = (0, 255, 0)
                    else:    
                        temp_color = (255, 0, 0)

                    cv2.putText(img, str(int(op_confs[cnt][i] * 100)), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, temp_color)
                    cv2.circle(img, (int(j[0]), int(j[1])), 10, temp_color)

                cv2.imshow("img", img)
                k = cv2.waitKey(0)
                if k == 27:
                    return
                elif k == ord('q'):
                    break

        frame_cnt = len(op_joints)

        # openpose pose filter
        op_filter = True
        changed = []
        if op_filter:
            for cnt, img in enumerate(frames):
                for num, _ in enumerate(op_joints[cnt]):
                    joint = op_joints[cnt][num]
                    conf = op_confs[cnt][num]
                
                    if conf < 0.33:
                        finish_l = finish_r = False
   
                        # set range
                        for i in range(1, 3):
                            if not finish_l and 0 <= cnt - i:
                                if 0.66 < op_confs[cnt - i][num]:
                                    conf_l = op_confs[cnt - i][num]
                                    joint_l = op_joints[cnt - i][num]
                                    index_l = cnt - i
                                    finish_l = True
      
                            if not finish_r and cnt + i < frame_cnt:
                                if 0.66 < op_confs[cnt + i][num]:
                                    conf_r = op_confs[cnt + i][num]
                                    joint_r = op_joints[cnt + i][num]
                                    index_r = cnt + i
                                    finish_r = True
                                
                        if finish_l and finish_r:
                            if cnt not in changed:
                                changed.append(cnt)

                            conf_gradient = (conf_r - conf_l) / (index_r - index_l)
                            conf_bias = conf_l - conf_gradient * index_l
                            joint_gradient = (joint_r - joint_l) / (index_r - index_l)
                            joint_bias = joint_l - joint_gradient * index_l

                            for j in range(index_l + 1, index_r):
                                op_confs[j][num] = - (conf_gradient * j + conf_bias)
                                op_joints[j][num] = joint_gradient * j + joint_bias

	print changed


        # smplify
        max_proc = 5
        procs = []
        for cnt, img in enumerate(frames):
            if cnt <= num:
                continue

            print "video frame num: " + str(cnt)
            out_path = '%s/%04d.pkl' % (out_video_dir, cnt)

            if not exists(out_path) or (cnt in changed and not exists(out_path[:-4] + '_new.pkl')):
                if len(procs) < max_proc:
                    procs.append(Process(target=smplify, args=(img, out_path, out_video_dir, op_joints, op_confs, cnt, genders, model_female, model_male, sph_regs_female, sph_regs_male, n_betas, flength, pix_thsh, do_degrees)))

            if len(procs) == max_proc:
                for proc in procs:
                    proc.start()
                for proc in procs:
                    proc.join()

                procs[:] = []
                cap.release()

                return cnt, False

        if len(procs) > 0:
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()
            procs[:] = []

        return cnt, True

def adjust_gamma(image, gamma=1.0):
    image = image / 255.0
    image = cv2.pow(image, 1.0 / gamma)
    return np.uint8(image * 255)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        '--video',
        default="False",
        help="Input Video Path")
    args = parser.parse_args()

    mmain(args.video)
