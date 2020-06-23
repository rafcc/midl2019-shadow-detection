import chainer
from chainer import functions as F
from chainer import links as L
from chainer import reporter
import numpy as np
import cv2
import functions


class ShadowSplitter(chainer.Chain):
    def __init__(self,
                 lambda_l2=1.,
                 lambda_l1=1.,
                 lambda_beta_shadow=1e-7,
                 lambda_beta_structure=1e-7,
                 lambda_ss=1.,
                 lambda_ssreg=1.,
                 ssreg_thresh=0.8,
                 lambda_edge=1):
        super(ShadowSplitter, self).__init__()

        with self.init_scope():
            self.encoder = Encoder()
            self.shadow_decoder = Decoder()
            self.structure_decoder = Decoder()

        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.lambda_beta_shadow = lambda_beta_shadow
        self.lambda_beta_structure = lambda_beta_structure
        self.lambda_ss = lambda_ss

        self.lambda_ssreg = lambda_ssreg
        self.ssreg_thresh = ssreg_thresh

        self.beta_shadow_a = 1.
        self.beta_shadow_b = 1.
        self.beta_structure_a = 10.
        self.beta_structure_b = 10.

    def __call__(self, x):
        xp = chainer.backends.cuda.get_array_module(x)

        # Create artificial shadow
        if chainer.config.train:
            a_shadows = []
            a_shadow_masks = []
            for i in range(x.shape[0]):
                a_shadow = np.zeros(x.shape[2:], dtype='uint8')

                # Shadow generation
                shadow_origin_x = int(x.shape[3] / 2)
                shadow_origin_y = 0
                shadow_height = int(np.random.uniform(0.2, 1, 1) * x.shape[2])
                shadow_small_radius = int(
                    np.random.uniform(
                        np.maximum(0, -shadow_origin_y),
                        x.shape[2] - shadow_origin_y, 1))
                shadow_width_rad = np.random.uniform(1e-8, 90, 1)
                shadow_angle = int(np.random.uniform(90 - 45, 90 + 45, 1))
                a_shadow_large = np.zeros(x.shape[2:], dtype='float')
                for i in np.linspace(-shadow_width_rad/10,
                                     shadow_width_rad/10, 10):
                    a_shadow_large += cv2.ellipse(
                        np.zeros(x.shape[2:], dtype='uint8'),
                        (shadow_origin_x, shadow_origin_y),
                        (shadow_small_radius + shadow_height,
                         shadow_small_radius + shadow_height), 0,
                        shadow_angle - shadow_width_rad / 2 + i,
                        shadow_angle + shadow_width_rad / 2 + i, 1, -1)
                a_shadow_large /= a_shadow_large.max()
                a_shadow_large *= 255
                a_shadow_small = cv2.ellipse(
                    np.zeros(x.shape[2:], dtype='uint8'),
                    (shadow_origin_x, shadow_origin_y),
                    (shadow_small_radius, shadow_small_radius), 0, 0, 360, 255,
                    -1)
                a_shadow = np.clip(
                    a_shadow_large.astype('float') -
                    a_shadow_small.astype('float'), 0, 255)
                a_shadow = cv2.GaussianBlur(a_shadow, (7, 7), 0)

                # [0, 255] -> [0, 1]
                a_shadow = xp.array(
                    a_shadow, dtype='float32')[None, :, :] / 255.
                # Leave strict shadow (note that shadow area == 1) as mask
                a_shadow_mask = (a_shadow == 1).astype('float32')
                # Make shadow semi-transparent
                a_shadow *= xp.random.uniform(0.5, 1, 1)

                a_shadows.append(a_shadow)
                a_shadow_masks.append(a_shadow_mask)
            a_shadow = xp.stack(a_shadows, axis=0)
            a_shadow_mask = xp.stack(a_shadow_masks, axis=0)
            # Swap fg and bg (black background -> white background)
            a_shadow = 1 - a_shadow
            # Add artificial shadow
            x *= a_shadow
        else:
            # Dummy for calculating ss_loss for eval
            a_shadow = xp.ones(x.shape, dtype='float32')
            a_shadow_mask = xp.ones(x.shape, dtype='float32')

        # Encode
        z = self.encoder(x)
        # Decode to shadow
        shadow = self.shadow_decoder(z)
        shadow = F.sigmoid(shadow)

        # Decode to structure
        structure = self.structure_decoder(z)
        structure = F.sigmoid(structure)

        # Reconstruction
        reconstruction = structure * shadow

        # Reconstruction loss
        l2_loss = F.mean_squared_error(x, reconstruction)
        l2_loss *= self.lambda_l2
        l1_loss = F.mean_absolute_error(x, reconstruction)
        l1_loss *= self.lambda_l1

        # Shadow augmented supervised loss
        ss_loss = F.mean(a_shadow_mask * (shadow - a_shadow)**2)
        ss_loss *= self.lambda_ss

        # Shadow regularization, defauling to no shadow
        ssreg_loss = F.mean(
            F.relu(self.ssreg_thresh - shadow)
        )
        ssreg_loss *= self.lambda_ssreg

        # Beta distribution loss
        if self.lambda_beta_shadow != 0:
            beta_shadow_loss = -functions.log_beta_distribution(
                F.flatten(shadow),
                self.beta_shadow_a,
                self.beta_shadow_b,
            )
            beta_shadow_loss = F.sum(beta_shadow_loss)
            beta_shadow_loss *= self.lambda_beta_shadow
        else:
            beta_shadow_loss = 0

        if self.lambda_beta_structure != 0:
            beta_structure_loss = -functions.log_beta_distribution(
                F.flatten(structure),
                self.beta_structure_a,
                self.beta_structure_b,
            )
            beta_structure_loss = F.sum(beta_structure_loss)
            beta_structure_loss *= self.lambda_beta_structure
        else:
            beta_structure_loss = 0

        loss = (l2_loss + l1_loss + beta_shadow_loss + beta_structure_loss +
                ss_loss + ssreg_loss)

        # For output intermediate result
        self.x = x
        self.shadow = shadow.data
        self.a_shadow = a_shadow
        self.structure = structure.data
        self.reconstruction = reconstruction.data

        reporter.report({
            'loss': loss,
            'l2_loss': l2_loss,
            'l1_loss': l1_loss,
            'beta_shadow_loss': beta_shadow_loss,
            'beta_structure_loss': beta_structure_loss,
            'ss_loss': ss_loss,
            'ssreg_loss': ssreg_loss,
        }, self)

        return loss


class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.conv3_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.conv4_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv5_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv5_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1_1(x))
        h = F.leaky_relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2)  # 256 -> 128
        h = F.leaky_relu(self.conv2_1(h))
        h = F.leaky_relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2)  # 128 -> 64
        h = F.leaky_relu(self.conv3_1(h))
        h = F.leaky_relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, ksize=2)  # 64 -> 32
        h = F.leaky_relu(self.conv4_1(h))
        h = F.leaky_relu(self.conv4_2(h))
        h = F.max_pooling_2d(h, ksize=2)  # 32 -> 16
        h = F.leaky_relu(self.conv5_1(h))
        h = F.leaky_relu(self.conv5_2(h))

        return h


class Decoder(chainer.Chain):
    def __init__(self):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.conv5_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv5_1 = L.Convolution2D(
                None, 256 * 4, ksize=3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv4_1 = L.Convolution2D(
                None, 256 * 4, ksize=3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.conv3_1 = L.Convolution2D(
                None, 128 * 4, ksize=3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(
                None, 64 * 4, ksize=3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.conv1_1 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv5_2(x))
        h = F.leaky_relu(self.conv5_1(h))
        h = F.depth2space(h, 2)  # 16 -> 32
        h = F.leaky_relu(self.conv4_2(h))
        h = F.leaky_relu(self.conv4_1(h))
        h = F.depth2space(h, 2)  # 32 -> 64
        h = F.leaky_relu(self.conv3_2(h))
        h = F.leaky_relu(self.conv3_1(h))
        h = F.depth2space(h, 2)  # 64 -> 128
        h = F.leaky_relu(self.conv2_2(h))
        h = F.leaky_relu(self.conv2_1(h))
        h = F.depth2space(h, 2)  # 128 -> 256
        h = F.leaky_relu(self.conv1_2(h))
        h = F.leaky_relu(self.conv1_1(h))

        return h
