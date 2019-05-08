"""
This file provides a wrapper around GraphCNN and SMPLParamRegressor and is useful for inference since it fuses both forward passes in one.
It returns both the non-parametric and parametric shapes, as well as the camera and the regressed SMPL parameters.
"""
import torch
import torch.nn as nn

from models import GraphCNN, SMPLParamRegressor, SMPL

class CMR(nn.Module):

    def __init__(self, mesh, num_layers, num_channels, pretrained_checkpoint=None):
        super(CMR, self).__init__()
        self.graph_cnn = GraphCNN(mesh.adjmat, mesh.ref_vertices.t(),
                                  num_layers, num_channels)
        self.smpl_param_regressor = SMPLParamRegressor()
        self.smpl = SMPL()
        self.mesh = mesh
        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint)
            try:
                self.graph_cnn.load_state_dict(checkpoint['graph_cnn'])
            except KeyError:
                print('Warning: graph_cnn was not found in checkpoint')
            try:
                self.smpl_param_regressor.load_state_dict(checkpoint['smpl_param_regressor'])
            except KeyError:
                print('Warning: smpl_param_regressor was not found in checkpoint')

    def forward(self, image, train_graph_cnn=True, train_smpl_param_regressor=True, detach=True):
        """Fused forward pass for the 2 networks
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed non-parametric shape: size = (B, 6890, 3)
            Regressed SMPL shape: size = (B, 6890, 3)
            Weak-perspective camera: size = (B, 3)
            SMPL pose parameters (as rotation matrices): size = (B, 24, 3, 3)
            SMPL shape parameters: size = (B, 10)
        """
        batch_size = image.shape[0]
        if not train_graph_cnn:
            with torch.no_grad():
                pred_vertices_sub, camera = self.graph_cnn(image)
        else:
            pred_vertices_sub, camera = self.graph_cnn(image)
        pred_vertices = self.mesh.upsample(pred_vertices_sub.transpose(1, 2))
        if detach:
            x = pred_vertices_sub.transpose(1,2).detach()
        else:
            x = pred_vertices_sub.transpose(1,2)
        x = torch.cat([x, self.mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)], dim=-1)
        if not train_smpl_param_regressor:
            with torch.no_grad():
                pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        else:
            pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vertices_smpl = self.smpl(pred_rotmat, pred_betas)
        return pred_vertices, pred_vertices_smpl, camera, pred_rotmat, pred_betas
