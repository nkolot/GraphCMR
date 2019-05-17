#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess import h36m_extract,\
                                lsp_dataset_extract,\
                                lsp_dataset_original_extract,\
                                mpii_extract,\
                                coco_extract,\
                                up_3d_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH

    if args.train_files:
        # UP-3D dataset preprocessing (trainval set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')   
      
        # LSP dataset original preprocessing (training set)
        lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, out_path)

        # MPII dataset preprocessing
        mpii_extract(cfg.MPII_ROOT, out_path)

        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, out_path)

    if args.eval_files:
        # # Human3.6M preprocessing (two protocols)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)
        
        # LSP dataset preprocessing (test set)
        lsp_dataset_extract(cfg.LSP_ROOT, out_path)
        
        # UP-3D dataset preprocessing (lsp_test set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')
