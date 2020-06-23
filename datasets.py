import chainer
import six
import os
from PIL import Image
import numpy as np
from chainercv import transforms


class AugmentedImageDataset(chainer.dataset.dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', dtype=None, train=True):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self.train = train

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        # Read image and convert to grayscale
        image = np.array(Image.open(path).convert('L'), dtype=self._dtype)
        # [0, 255] -> [0, 1]
        image /= 255.
        # Add channel
        image = image[None, :, :]

        if self.train:
            # Data augmentation
            # Random crop
            image = transforms.random_sized_crop(
                image, scale_ratio_range=(0.5, 1))
            # Random photometric-distortions (TODO)
            pass

        # Resize to 128x128
        image = transforms.resize(image, size=(128, 128))

        return image
