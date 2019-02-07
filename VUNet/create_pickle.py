import pickle
import os
import pandas as pd
import json
import numpy as np
import cv2

vunet_order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder',
               'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']
# joints: dictionary, key is the image name, value is array of size (18, 2), the coordinates are nomalized to [0, 1], occluded joints
# are represented by -1
# imgs: a dictionary, key is the image name, value is array of size (18, 2), containing train and test image paths under the same folder with index.p
# joint_order: vunet_order
# train: 1 indicates belongs to train images, 0 indicates belongs to test images.


def get_pairs(pairLst):
    pairs_file = pd.read_csv(pairLst)
    size = len(pairs_file)
    pairs = []
    print('Loading data pairs ...')
    for i in range(size):
        pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
        pairs.append(pair)
    print('Loading data pairs finished ...')
    return pairs


def load_pose_cords_from_strings(x_str, y_str, img_width, img_height, x_bound=0, y_bound=0):
    x_cords = json.loads(x_str)
    y_cords = json.loads(y_str)
    n_x_cords = []
    for x in x_cords:
        if x >= 0:
            n_x_cords.append((x + x_bound) / float(img_width))
        else:
            n_x_cords.append(x)
    n_y_cords = []
    for y in y_cords:
        if y >= 0:
            n_y_cords.append((y + y_bound) / float(img_height))
        else:
            n_y_cords.append(y)
    return np.concatenate([np.expand_dims(np.array(n_x_cords), -1),
                           np.expand_dims(np.array(n_y_cords), -1)], axis=1)


def load_keypoints(kp_df, to_search_file_name, image_width, image_height, x_bound=0, y_bound=0):
    kp = kp_df[kp_df['name'] == to_search_file_name].iloc[0]
    kp_coords = load_pose_cords_from_strings(kp['keypoints_x'], kp['keypoints_y'], image_width, image_height, x_bound, y_bound)
    return kp_coords


def create_pickle_for_vunet(img_dir, csv_annotation_file, save_path, image_width, image_height, x_bound=0, y_bound=0):
    all_img_names = os.listdir(img_dir)
    to_save_contents = dict()
    to_save_contents['joint_order'] = vunet_order
    to_save_contents['imgs'] = dict()
    to_save_contents['joints'] = dict()
    kp_df = pd.read_csv(csv_annotation_file, sep=':')
    for img_name in all_img_names:
        print(img_name)
        kp_coords = load_keypoints(kp_df, img_name, image_width, image_height, x_bound, y_bound)
        to_save_contents['imgs'][img_name] = img_name
        to_save_contents['joints'][img_name] = kp_coords
    output = open(save_path, 'wb')
    pickle.dump(to_save_contents, output)


def padding_image(img_dir, save_dir, padding_length):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        new_im = cv2.copyMakeBorder(img, 0, 0, padding_length, padding_length, cv2.BORDER_REPLICATE)
        cv2.imwrite(os.path.join(save_dir, img_name), new_im)


if __name__ == '__main__':
    # Specify the image dir, which only contains images
    img_dir = './test'
    # Specify the annotation file of the image dir
    csv_annotation_file = './market-annotation-test.csv'
    # Where to save the pickle file
    save_path = './test.p'
    # For market1501
    create_pickle_for_vunet(img_dir, csv_annotation_file, save_path, 128, 128, x_bound=32, y_bound=0)
    # For deepfashion
    # create_pickle_for_vunet(img_dir, csv_annotation_file, save_path, 256, 256, x_bound=0, y_bound=0)