import chainer
from chainer import functions as F
import argparse
from os import path
import glob
from PIL import Image
import numpy as np
import cupy
import chainercv
import cv2

import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('image_list', type=str)
    parser.add_argument('label_list', type=str)
    args = parser.parse_args()

    # Load list
    with open(args.image_list) as f:
        image_paths = [s.strip() for s in f.readlines()]

    with open(args.label_list) as f:
        label_paths = [s.strip() for s in f.readlines()]

    threshs = np.arange(0, 255, 5)
    iou_list = np.zeros([len(image_paths), threshs.size])
    dice_list = np.zeros_like(iou_list)
    acc_list = np.zeros_like(iou_list)
    prec_list = np.zeros_like(iou_list)
    recall_list = np.zeros_like(iou_list)
    spec_list = np.zeros_like(iou_list)

    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        # Load image and label
        image = Image.open(path.join(args.dataset_root, image_path)).convert('L').resize([128, 128], resample=2)
        label = Image.open(path.join(args.dataset_root, label_path)).convert('L').resize([128, 128], resample=0)

        # Convert image and label
        x = np.array(image, dtype='uint8')
        l = (np.array(label) == 255).astype('uint8')


        for thresh_idx, thresh in enumerate(threshs):
            # Predict shadow
            _, shadow_binary = cv2.threshold(x, thresh, 255, cv2.THRESH_BINARY)
            shadow_binary = shadow_binary == 255

            # Calc IoU
            iou = np.sum(shadow_binary * l) / np.sum((shadow_binary + l) > 0)
            # Calc DICE
            dice = 2 * np.sum(shadow_binary * l) / (np.sum(shadow_binary) + np.sum(l))
            # Calc confusion matrix
            tn = np.sum((shadow_binary == 0) * (l == 0))
            tp = np.sum((shadow_binary == 1) * (l == 1))
            fn = np.sum((shadow_binary == 0) * (l == 1))
            fp = np.sum((shadow_binary == 1) * (l == 0))
            acc = (tp + tn) / (tp + tn + fp + fn)
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            spec = tn / (fp + tn)

            iou_list[i, thresh_idx] = iou
            dice_list[i, thresh_idx] = dice
            acc_list[i, thresh_idx] = acc
            prec_list[i, thresh_idx] = prec
            recall_list[i, thresh_idx] = recall
            spec_list[i, thresh_idx] = spec

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print('Thresh, IoU avg, IoU std, DICE avg, DICE std, Acc mean, Acc std, Prec mean, Prec std, Recall mean, Recall std, Spec mean, Spec std')
    for line in np.vstack([threshs,
                           iou_list.mean(axis=0),
                           iou_list.std(axis=0),
                           dice_list.mean(axis=0),
                           dice_list.std(axis=0),
                           acc_list.mean(axis=0),
                           acc_list.std(axis=0),
                           prec_list.mean(axis=0),
                           prec_list.std(axis=0),
                           recall_list.mean(axis=0),
                           recall_list.std(axis=0),
                           spec_list.mean(axis=0),
                           spec_list.std(axis=0),
                           ]).T:
        print(', '.join(['{:4.3f}'.format(v) for v in line]))
    print('Max DICE is {} at thresh {}.'.format(dice_list.mean(axis=0).max(),
                                                threshs[np.argmax(dice_list.mean(axis=0))]))
