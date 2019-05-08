"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from utils import BaseTrainer, Mesh
from datasets import create_dataset
from models import GraphCNN, SMPLParamRegressor, SMPL
from models.geometric_layers import orthographic_projection, rodrigues
from utils.renderer import Renderer, visualize_reconstruction

class Trainer(BaseTrainer):
    """Trainer object.
    Inherits from BaseTrainer that sets up logging, saving/restoring checkpoints etc.
    """
    def init_fn(self):
        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options)

        # create Mesh object
        self.mesh = Mesh()
        self.faces = self.mesh.faces.to(self.device)

        # create GraphCNN
        self.graph_cnn = GraphCNN(self.mesh.adjmat,
                           self.mesh.ref_vertices.t(),
                           num_channels=self.options.num_channels,
                           num_layers=self.options.num_layers
                          ).to(self.device)
        
        # SMPL Parameter regressor
        self.smpl_param_regressor = SMPLParamRegressor().to(self.device)

        # Setup a joint optimizer for the 2 models
        self.optimizer = torch.optim.Adam(params=list(self.graph_cnn.parameters()) + list(self.smpl_param_regressor.parameters()),
                                           lr=self.options.lr,
                                           betas=(self.options.adam_beta1, 0.999),
                                           weight_decay=self.options.wd)

        # SMPL model
        self.smpl = SMPL().to(self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {'graph_cnn': self.graph_cnn, 'smpl_param_regressor': self.smpl_param_regressor}
        self.optimizers_dict = {'optimizer': self.optimizer}
        
        # Renderer for visualization
        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())

        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1,3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        """Training step."""
        self.graph_cnn.train()
        self.smpl_param_regressor.train()

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints']
        gt_keypoints_3d = input_batch['pose_3d']
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        has_smpl = input_batch['has_smpl']
        has_pose_3d = input_batch['has_pose_3d']
        images = input_batch['img']

        # Render vertices using SMPL parameters
        gt_vertices = self.smpl(gt_pose, gt_betas)
        batch_size = gt_vertices.shape[0]

        # Feed image in the GraphCNN
        # Returns subsampled mesh and camera parameters
        pred_vertices_sub, pred_camera = self.graph_cnn(images)
        
        # Upsample mesh in the original size
        pred_vertices = self.mesh.upsample(pred_vertices_sub.transpose(1,2))

        # Prepare input for SMPL Parameter regressor
        # The input is the predicted and template vertices subsampled by a factor of 4
        # Notice that we detach the GraphCNN
        x = pred_vertices_sub.transpose(1,2).detach()
        x = torch.cat([x, self.mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)], dim=-1)

        # Estimate SMPL parameters and render vertices
        pred_rotmat, pred_shape = self.smpl_param_regressor(x)
        pred_vertices_smpl = self.smpl(pred_rotmat, pred_shape)

        # Get 3D and projected 2D keypoints from the regressed shape
        pred_keypoints_3d = self.smpl.get_joints(pred_vertices)
        pred_keypoints_2d = orthographic_projection(pred_keypoints_3d, pred_camera)[:, :, :2]
        pred_keypoints_3d_smpl = self.smpl.get_joints(pred_vertices_smpl)
        pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, pred_camera.detach())[:, :, :2]

        # Compute losses

        # GraphCNN losses 
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, has_pose_3d)
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # SMPL regressor losses
        loss_keypoints_smpl = self.keypoint_loss(pred_keypoints_2d_smpl, gt_keypoints_2d)
        loss_keypoints_3d_smpl = self.keypoint_3d_loss(pred_keypoints_3d_smpl, gt_keypoints_3d, has_pose_3d)
        loss_shape_smpl = self.shape_loss(pred_vertices_smpl, gt_vertices, has_smpl)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)                                                

        # Add losses to compute the total loss
        loss = loss_shape_smpl + loss_keypoints_smpl + loss_keypoints_3d_smpl +\
               loss_regr_pose + 0.1 * loss_regr_betas + loss_shape + loss_keypoints + loss_keypoints_3d

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization in a list
        out_args = [pred_vertices, pred_vertices_smpl, pred_camera, 
                    pred_keypoints_2d, pred_keypoints_2d_smpl,
                    loss_shape, loss_shape_smpl, loss_keypoints, loss_keypoints_smpl,
                    loss_keypoints_3d, loss_keypoints_3d_smpl,
                    loss_regr_pose, loss_regr_betas, loss]
        out_args = [arg.detach() for arg in out_args]
        return out_args

    def train_summaries(self, input_batch,
                        pred_vertices, pred_vertices_smpl, pred_camera,
                        pred_keypoints_2d, pred_keypoints_2d_smpl,
                        loss_shape, loss_shape_smpl, loss_keypoints, loss_keypoints_smpl,
                        loss_keypoints_3d, loss_keypoints_3d_smpl,
                        loss_regr_pose, loss_regr_betas, loss):
        """Tensorboard logging."""
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
         
        rend_imgs = []
        rend_imgs_smpl = []
        batch_size = pred_vertices.shape[0]
        # Do visualization for the first 4 images of the batch
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1,2,0)
            # Get LSP keypoints from the full list of keypoints
            gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
            pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, self.to_lsp]
            pred_keypoints_2d_smpl_ = pred_keypoints_2d_smpl.cpu().numpy()[i, self.to_lsp]
            # Get GraphCNN and SMPL vertices for the particular example
            vertices = pred_vertices[i].cpu().numpy()
            vertices_smpl = pred_vertices_smpl[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            # Visualize reconstruction and detected pose
            rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, self.renderer)
            rend_img_smpl = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices_smpl, pred_keypoints_2d_smpl_, cam, self.renderer)
            rend_img = rend_img.transpose(2,0,1)
            rend_img_smpl = rend_img_smpl.transpose(2,0,1)
            rend_imgs.append(torch.from_numpy(rend_img))
            rend_imgs_smpl.append(torch.from_numpy(rend_img_smpl))
        rend_imgs = make_grid(rend_imgs, nrow=1)
        rend_imgs_smpl = make_grid(rend_imgs_smpl, nrow=1)

        # Save results in Tensorboard
        self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
        self.summary_writer.add_image('imgs_smpl', rend_imgs_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_shape', loss_shape, self.step_count)
        self.summary_writer.add_scalar('loss_shape_smpl', loss_shape_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_regr_pose', loss_regr_pose, self.step_count)
        self.summary_writer.add_scalar('loss_regr_betas', loss_regr_betas, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints', loss_keypoints, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_smpl', loss_keypoints_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d', loss_keypoints_3d, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d_smpl', loss_keypoints_3d_smpl, self.step_count)
        self.summary_writer.add_scalar('loss', loss, self.step_count)
