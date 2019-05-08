import os
import argparse
import numpy as np
import scipy.io as sio

def lsp_dataset_original_extract(dataset_path, out_path):

    # bbox expansion factor
    scaleFactor = 1.2

    # we use LSP dataset original for training
    imgs = range(1000)

    # structs we use
    imgnames_, scales_, centers_, parts_  = [], [], [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    # go over all the images
    for img_i in imgs:
        # image name
        imgname = 'images/im%04d.jpg' % (img_i+1)
        # read keypoints
        part14 = joints[:2,:,img_i].T
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])

        # store data
        imgnames_.append(imgname)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)

    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'lsp_dataset_original_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_)
