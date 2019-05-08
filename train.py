#!/usr/bin/python
"""
This script is a wrapper for the training process

Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from UP-3D (full shape ground truth) and other in-the wild datasets (e.g., [LSP](http://sam.johnson.io/research/lsp.html), [MPII](http://human-pose.mpi-inf.mpg.de), [COCO](http://cocodataset.org/#home) that provide 2D keypoint ground truth). Again, make sure that you follow the [details for data preprocessing](https://github.com/nkolot/GraphCMR/blob/master/datasets/preprocess/README.md).

Example usage:
```
python train.py --name train_example --pretrained_checkpoint=data/models/model_checkpoint_h36m_up3d.pt --config=data/config.json
```
You can view the full list of command line options by running `python train.py --help`. The default values are the ones used to train the models in the paper.
Running the above command will start the training process. It will also create the folders `logs` and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look at the logs stored during training.
"""
from utils import TrainOptions
from train import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
