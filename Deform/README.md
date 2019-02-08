# Deformable GANs for Pose-based Human Image Generation.

### Requirment
* python2
* Numpy
* Scipy
* Skimage
* Pandas
* Tensorflow
* Keras
* tqdm

### Data preparation
Download pose estimator (conversion of this https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) [pose_estimator.h5](https://yadi.sk/d/blgmGpDi3PjXvK). Launch ```python compute_cordinates.py.``` It will compute human keypoints. Our estimations are given in ``DATA``, e.g. ``DATA/fashion-annotation-train.csv``.

Create pairs dataset with ```python create_pairs_dataset.py```. It define pairs for training or testing. Samples can be seen in ``DATA/train_pairs.csv``.

### Pose transfer testing
0. In order to do pose transfer comparisons, download model named ``generator-warp-maks-nn3-cl12.h5`` for market1501, ``generator-warp-maks-nn5-cl12.h5`` for DeepFashion from [pretrained models](https://yadi.sk/d/dxVvYxBw3QuUT9).
1. Run ```python test.py --generator_checkpoint path/to/generator/checkpoint``` (and same parameters as in train.py). It generate images and compute inception score, SSIM score and their masked versions.

### Warning
The version of our tensorflow is 1.4.0, the paths of ``annotations_file_train, images_dir_train, pairs_file_train`` in ``cmd.py`` should be specified correctly.