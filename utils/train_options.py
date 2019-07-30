import os
import json
import argparse
import numpy as np
from collections import namedtuple

class TrainOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained Graph CNN when starting training') 

        arch = self.parser.add_argument_group('Architecture')
        arch.add_argument('--model', default='graphcnn', choices=['graphcnn', 'baseline'], help='Number of filters in graph convolutions') 
        arch.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers') 
        arch.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN') 
        arch.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding it in the network') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--dataset', default='itw', choices=['itw', 'all'], help='Choose training dataset')
        train.add_argument('--num_epochs', type=int, default=50, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--noise_factor', type=float, default=0.4, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--scale_factor', type=float, default=0.25, help='rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]') 
        train.add_argument('--no_augmentation', dest='use_augmentation', default=True, action='store_false', help='Don\'t do augmentation') 
        train.add_argument('--no_augmentation_rgb', dest='use_augmentation_rgb', default=True, action='store_false', help='Don\'t do color jittering during training') 
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        optim.add_argument("--wd", type=float, default=0, help="Weight decay weight")
        return 

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
