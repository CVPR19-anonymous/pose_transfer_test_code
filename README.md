### Code architecture
``DATA`` contains the training/testing pairs and pose estimations of our method.

``PG2`` denotes the codes of paper **Pose Guided Person Image Generation**

``Deform`` denotes the codes of paper **Deformable GANs for Pose-based Human Image Generation**

``VUNet`` denotes the codes of paper **A Variational U-Net for Conditional Appearance and Shape Generation**


### Reminders
* Download market dataset https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view. Unzip this file to a folder. Rename this folder to market-dataset. Rename bounding_box_test and bounding_box_train with test and train. 

* Download deep [fasion dataset in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). You will need to ask a pasword from dataset maintainers. Move img/ to data folder and rename it fasion/. Download key-point estimations from (https://yadi.sk/d/suymftBy3S7oKD) for fasion. Run script ```Deform/data/split_fasion_data.py```. 

* Data preparation for PG2 can be referred to the ``TF-record data preparation steps`` in ``PG2/README.md``

* To test a individual method, please read the ``README.md`` file in the corresponding folder first.