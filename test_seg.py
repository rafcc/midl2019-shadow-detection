import chainer
from chainer import functions as F
import argparse
import os
from os import path
import glob
from PIL import Image
import numpy as np
import cupy
import chainercv
import cv2

import nn
from chainercv.links import PixelwiseSoftmaxClassifier
from segnet import SegNetBasic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('image_list', type=str)
    parser.add_argument('label_list', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--segnet', action='store_true')
    args = parser.parse_args()

    # Load model
    if args.segnet:
        model = SegNetBasic(n_class=2)
        chainer.serializers.load_npz(args.model_path, model)
    else:
        model = nn.ShadowSplitter()
        chainer.serializers.load_npz(args.model_path, model,
                                     'updater/model:main/')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Load list
    with open(args.image_list) as f:
        image_paths = [s.strip() for s in f.readlines()]

    with open(args.label_list) as f:
        label_paths = [s.strip() for s in f.readlines()]

    threshs = np.arange(0, 1, 0.01)
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
        x = np.array(image, dtype='float32') / 255.
        x = x[None, None, :]
        l = (np.array(label) == 255).astype('uint8')
        l = l[None, None, :]
        if args.gpu >= 0:
            x = cupy.array(x, dtype='float32')

        # Predict shadow
        with chainer.using_config('train', False):
            if args.segnet:
                # Extract only shadow class (=1)
                shadow = 1 - F.softmax(model(x * 255), axis=1).data[:, 1][:, None, :, :]
            else:
                shadow = F.sigmoid(model.shadow_decoder(model.encoder(x))).data
        if args.gpu >= 0:
            x = cupy.asnumpy(x)
            shadow = cupy.asnumpy(shadow)

        for thresh_idx, thresh in enumerate(threshs):
            shadow_binary = shadow < thresh

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

        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)

            x = (x[0, 0] * 255).astype('uint8')
            shadow = (shadow[0, 0] * 255).astype('uint8')
            l = (255 - l[0, 0] * 255).astype('uint8')

            shadow_heat = cv2.applyColorMap(255 - shadow, cv2.COLORMAP_JET)
            l_heat = cv2.applyColorMap(255 - l, cv2.COLORMAP_JET)

            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
            shadow = cv2.cvtColor(shadow, cv2.COLOR_GRAY2RGB)
            l = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)

            overlayed = cv2.addWeighted(shadow_heat, 0.3, x, 1, 0)
            overlayed_gt = cv2.addWeighted(l_heat, 0.3, x, 1, 0)
            concat_img = cv2.hconcat([x, shadow, l, overlayed, overlayed_gt])

            cv2.imwrite(path.join(args.output_path, image_path.replace('/', '_')) + '.png', concat_img)

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
