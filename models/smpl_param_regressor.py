"""
Definition of SMPL Parameter Regressor used for regressing the SMPL parameters from the 3D shape
"""
from __future__ import division
import torch
import torch.nn as nn
from models.layers import FCBlock, FCResBlock

class SMPLParamRegressor(nn.Module):

    def __init__(self, use_cpu_svd=True):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        self.layers = nn.Sequential(FCBlock(1723 * 6, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 24 * 3 * 3 + 10))
        self.use_cpu_svd = use_cpu_svd

    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        betas = x[:, 24*3*3:].contiguous()
        rotmat = rotmat.view(-1, 3, 3).contiguous()
        orig_device = rotmat.device
        if self.use_cpu_svd:
            rotmat = rotmat.cpu()
        U, S, V = batch_svd(rotmat)

        rotmat = torch.matmul(U, V.transpose(1,2))
        det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        with torch.no_grad():
            for i in range(rotmat.shape[0]):
                det[i] = torch.det(rotmat[i])
        rotmat = rotmat * det
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        rotmat = rotmat.to(orig_device)
        return rotmat, betas

def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V
