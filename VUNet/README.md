### Claim
This file is a simple instruction of doing pose transfer by VUNet. 
### Data preparation
Prepare the training/testing pairs using scripts stored in ``Deform/create_pairs_dataset.py``. Samples can be seen in ``DATA/train_pairs.csv``

Download pose estimator (conversion of this https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) [pose_estimator.h5](https://yadi.sk/d/blgmGpDi3PjXvK). Launch ``Deform/python compute_coordinates.py``. It will compute human keypoints. 
Our estimations are given in ``DATA``.

Then, create pickle file used for pose transfer. Script ``create_pickle.py`` is used to create pickle file. It contains an example. Just correctly specify the file/directory path.

### Do pose transfer
Download official models trained on Market1501 and DeepFashion from [VUNet](https://heibox.uni-heidelberg.de/d/71842715a8/?p=/vunet/pretrained_checkpoints&mode=list)

On Market1501:
```bash
CUDA_VISIBLE_DEVICES=0 python main_market.py --data_index PATH/TO/TEST_PICKLE_FILE --test_data_index PATH/TO/TEST_PICKLE_FILE --mode transfer --checkpoint ./model.ckpt-100000 --pairs_path PATH/TO/TEST_PAIRS --spatial_size 128 --batch_size 1
```

On DeepFashion:
```bash
CUDA_VISIBLE_DEVICES=0 python main_deepfashion.py --data_index PATH/TO/TEST_PICKLE_FILE --test_data_index PATH/TO/TEST_PICKLE_FILE --mode transfer --checkpoint ./model.ckpt-100000 --pairs_path PATH/TO/TEST_PAIRS --spatial_size 256 --batch_size 1
```