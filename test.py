import chainer
from chainer import functions as F
import argparse
from os import path
import glob
from PIL import Image
import numpy as np
import cupy
import cv2

import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Load model
    model = nn.ShadowSplitter()
    chainer.serializers.load_npz(args.model_path, model,
                                 'updater/model:main/')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Make file list
    if path.isdir(args.data_path):
        filelist = glob.glob(path.join(args.data_path, '*.png'))
    else:
        filelist = [args.data_path]

    for fname in filelist:
        im = Image.open(fname).convert('L').resize([128, 128], resample=2)
        x = np.array(im, dtype='float32') / 255.
        x = x[None, None, :]
        if args.gpu >= 0:
            x = cupy.array(x, dtype='float32')
        shadow = F.sigmoid(model.shadow_decoder(model.encoder(x))).data
        if args.gpu >= 0:
            x = cupy.asnumpy(x)
            shadow = cupy.asnumpy(shadow)

        x = (x[0, 0] * 255).astype('uint8')
        shadow = (shadow[0, 0] * 255).astype('uint8')

        shadow_heat = cv2.applyColorMap(shadow, cv2.COLORMAP_JET)

        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        shadow = cv2.cvtColor(shadow, cv2.COLOR_GRAY2RGB)

        overlayed = cv2.addWeighted(shadow_heat, 0.3, x, 1, 0)
        concat_img = cv2.hconcat([x, shadow, overlayed])

        cv2.namedWindow('window')
        cv2.imshow('window', concat_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
