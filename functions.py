from chainer import functions as F
from scipy import special
import numpy as np
from scipy import ndimage


def log_beta_distribution(x, a, b):
    eps = 1e-5
    lnp = ((a - 1) * F.log(x + eps)
           + (b - 1) * F.log(1 - x + eps)
           - float(special.beta(a, b)))

    return lnp


def make_laplacian_of_gaussian_filter(sigma, ksize, angle):
    # Make laplacian of gaussian filter
    f = np.zeros([101, 101])
    x = np.arange(-50, 51)
    f[50, :] = (x**2 - sigma**2) / sigma**4 * np.exp(- x**2 / 2 / sigma**2)

    # Rotate, note that angle is in degree
    f = ndimage.rotate(f, angle, reshape=False, order=1)

    # Crop by ksize
    f = f[50 - int(ksize / 2):50 + int(ksize / 2) + 1,
          50 - int(ksize / 2):50 + int(ksize / 2) + 1]

    return f
