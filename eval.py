#!/usr/bin/python
"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python eval.py --checkpoint=data/models/model_checkpoint_h36m_up3d.pt --config=data/config.json --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. UP-3D ```--dataset=up-3d```
4. LSP ```--dataset=lsp```
"""
from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm

import config as cfg
from models import CMR, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils.mesh import Mesh

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'up-3d', 'lsp'], help='Choose evaluation dataset')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')

def run_evaluation(model, dataset_name, dataset, 
                   mesh, batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """
    
    renderer = PartRenderer()
    
    # Create SMPL model
    smpl = SMPL().cuda()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M)).float()
    
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Transfer model to the GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    eval_pose = False
    eval_shape = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2':
        eval_pose = True
    elif dataset_name == 'up-3d':
        eval_shape = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = cfg.DATASET_FOLDERS['upi-s1h']

    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl(gt_pose, gt_betas)
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        
        # Run inference
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, camera, pred_rotmat, pred_betas = model(images)
            
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)

            # Get 14 ground truth joints
            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_keypoints_3d = gt_keypoints_3d[:, cfg.J24_TO_J14, :-1]

            # Get 14 predicted joints from the non-parametic mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, cfg.H36M_TO_J14, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 
            # Get 14 predicted joints from the SMPL mesh
            pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices_smpl)
            pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0],:].clone()
            pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, cfg.H36M_TO_J14, :]
            pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl
            
            # Compute error metrics

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            error_smpl = torch.sqrt( ((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error
            mpjpe_smpl[step * batch_size:step * batch_size + curr_batch_size] = error_smpl

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error
            recon_err_smpl[step * batch_size:step * batch_size + curr_batch_size] = r_error_smpl

        # Shape evaluation (Mean per-vertex error)
        if eval_shape:
            se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            se_smpl = torch.sqrt(((pred_vertices_smpl - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            shape_err[step * batch_size:step * batch_size + curr_batch_size] = se
            shape_err_smpl[step * batch_size:step * batch_size + curr_batch_size] = se_smpl

        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE (NonParam): ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error (NonParam): ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('MPJPE (Param): ' + str(1000 * mpjpe_smpl[:step * batch_size].mean()))
                print('Reconstruction Error (Param): ' + str(1000 * recon_err_smpl[:step * batch_size].mean()))
                print()
            if eval_shape:
                print('Shape Error (NonParam): ' + str(1000 * shape_err[:step * batch_size].mean()))
                print('Shape Error (Param): ' + str(1000 * shape_err_smpl[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE (NonParam): ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error (NonParam): ' + str(1000 * recon_err.mean()))
        print('MPJPE (Param): ' + str(1000 * mpjpe_smpl.mean()))
        print('Reconstruction Error (Param): ' + str(1000 * recon_err_smpl.mean()))
        print()
    if eval_shape:
        print('Shape Error (NonParam): ' + str(1000 * shape_err.mean()))
        print('Shape Error (Param): ' + str(1000 * shape_err_smpl.mean()))
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    # Load model
    mesh = Mesh()
    model = CMR(mesh, options.num_layers, options.num_channels,
                      pretrained_checkpoint=args.checkpoint)
    # Setup evaluation dataset
    dataset = BaseDataset(options, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, mesh,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
