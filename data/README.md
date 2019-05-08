# Extra data
Short description of the additional data files that our code requires to run smoothly.

### J_regressor_extra.npy
Joints regressor for joints or landmarks that are not included in the standard set of SMPL joints.

### J_regressor_h36m.npy
Joints regressor reflecting the Human3.6M joints. Used for evaluation.

### cube_parts.npy & vertex_texture.npy
Files allowing us to do the 6-body part evaluation on LSP. 

### mesh_downsampling.npz
Extra file with precomputed downsampling for the SMPL body mesh.

### namesUPlsp.txt
Evaluation files for the UP-3D dataset. Recovered from [BodyNet repo](https://github.com/gulvarol/bodynet) (with minor modifications to remove the absolute paths).

### train.h5
Annotations for the training set of the MPII human pose dataset. Recovered from the [Stacked Hourglass repo](https://github.com/princeton-vl/pose-hg-train).
