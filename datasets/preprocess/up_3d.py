import os
import sys
import numpy as np
import scipy.misc
import argparse
import cPickle as pickle

def up_3d_extract(dataset_path, out_path, mode):

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_ = [], []

    # training/test splits
    if mode == 'trainval':
        txt_file = os.path.join(dataset_path, 'trainval.txt')
    elif mode == 'lsp_test':
        txt_file = 'data/namesUPlsp.txt'
    file = open(txt_file, 'r')
    txt_content = file.read()
    imgs = txt_content.split('\n')

    # go over all images
    for img_i in imgs:
        # skip empty row in txt
        if len(img_i) == 0:
            continue

        # image name 
        img_base = img_i[1:-10]
        img_name = '%s_image.png'%img_base

        # keypoints processing
        keypoints_file = os.path.join(dataset_path, '%s_joints.npy'%img_base)
        keypoints = np.load(keypoints_file)
        vis = keypoints[2]
        keypoints = keypoints[:2].T
        part = np.zeros([24,3])
        part[:14] = np.hstack([keypoints, np.vstack(vis)])

        # scale and center
        render_name = os.path.join(dataset_path, '%s_render_light.png' % img_base)
        I = scipy.misc.imread(render_name)
        ys, xs = np.where(np.min(I,axis=2)<255)
        bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

        # pose and shape
        pkl_file = os.path.join(dataset_path, '%s_body.pkl' % img_base)
        pkl = pickle.load(open(pkl_file, 'r'))
        pose = pkl['pose']
        shape = pkl['betas']

        # store data
        imgnames_.append(img_name)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        poses_.append(pose)
        shapes_.append(shape)

    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'up_3d_%s.npz' % mode)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       pose=poses_,
                       shape=shapes_)
