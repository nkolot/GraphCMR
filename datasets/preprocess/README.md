## Data preparation
Besides the demo code, we also provide training and evaluation code for our approach. To use this functionality, you need to download the relevant datasets.
The datasets that our code supports are:
1. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
2. [UP-3D](http://files.is.tuebingen.mpg.de/classner/up/)
3. [LSP](http://sam.johnson.io/research/lsp.html)
4. [MPII](http://human-pose.mpi-inf.mpg.de)
5. [COCO](http://cocodataset.org/#home)

More specifically:
1. **Human3.6M**: Unfortunately, due to license limitations, we are not allowed to redistribute the MoShed data that we used for training. We only provide code to evaluate our approach on this benchmark. To download the relevant data, please visit the [website of the dataset](http://vision.imar.ro/human3.6m/description.php) and download the Videos, BBoxes MAT (under Segments) and 3D Positions Mono (under Poses) for Subjects S9 and S11. After downloading and uncompress the data, store them in the folder ```${Human3.6M root}```. The sructure of the data should look like this:
```
${Human3.6M root}
|-- S9
    |-- Videos
    |-- Segments
    |-- Bboxes
|-- S11
    |-- Videos
    |-- Segments
    |-- Bboxes
```
You also need to edit the file ```config.py``` to reflect the path ```${Human3.6M root}``` you used to store the data. 

2. UP-3D: We use this data both for training and evaluation. You need to download the [UP-3D zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip) (that provides images and 3D shapes for training and testing) and the [UPi-S1h zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/upi-s1h.zip) (which we will need for silhouette evaluation on the LSP dataset). After you unzip, please edit ```config.py``` to include the paths for the two datasets.

3. LSP: We again use LSP both for training and evaluation. You need to download the high resolution version of the dataset [LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip) (for training) and the low resolution version [LSP dataset](http://sam.johnson.io/research/lsp_dataset.zip) (for evaluation). After you unzip the dataset files, please complete the relevant root paths of the datasets in the file ```config.py```.

4. MPII: We use this dataset for training. You need to download the compressed file of the [MPII dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz). After uncompressing, please complete the root path of the dataset in the file ```config.py```.

5. COCO: We use this dataset for training. You need to download the [images](http://images.cocodataset.org/zips/train2014.zip) and the [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) for the 2014 training set of the dataset. After you unzip the files, the folder structure should look like:
```
${COCO root}
|-- train2014
|-- annotations
```
Then, you need to edit the ```config.py``` file with the ```${COCO root}``` path.

### Generate dataset files
After preparing the data, we continue with the preprocessing to produce the data/annotations for each dataset in the expected format. You need to run the file ```preprocess_datasets.py``` from the main folder of this repo that will do all this work automatically. Depending on whether you want to do evaluation or/and training, we provide two modes:

If you want to generate the files such that you can evaluate our pretrained models, you need to run:
```
python preprocess_datasets.py --eval_files
```
If you want to generate the files such that you can train using the supported datasets, you need to run:
```
python preprocess_datasets.py --train_files
```
