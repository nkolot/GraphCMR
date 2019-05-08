#!/bin/bash

# Download pretrained networks
wget https://seas.upenn.edu/~nkolot/data/cmr/models.tar --directory-prefix=data
tar -xvf data/models.tar -C data && rm data/models.tar

# Download SMPL model
wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl --directory-prefix=data
