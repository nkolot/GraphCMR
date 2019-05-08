import os
from os.path import join
import sys
import json
import numpy as np

def coco_extract(dataset_path, out_path):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_ = [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'person_keypoints_train2014.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:,2]>0) < 12:
            continue
        # image name
        image_id = annot['image_id']
        img_name = join('train2014', str(imgs[image_id]['file_name']))
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        # store data
        imgnames_.append(img_name)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)

    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'coco_2014_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_)
