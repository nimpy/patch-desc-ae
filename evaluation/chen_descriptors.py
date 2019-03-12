import numpy as np
from evaluation.chen_descriptor_training import sigmoid

# Chen et al. version for RGB images

def load_chen_16_rgb():
    input_size = 16 * 16 * 3
    hidden_size = 128

    theta = np.load('encoderChenEtAl_RGB_400it.npy')

    W1 = theta[0:hidden_size * input_size].reshape(hidden_size, input_size)
    W2 = theta[hidden_size * input_size:2 * hidden_size * input_size].reshape(input_size, hidden_size)
    b1 = theta[2 * hidden_size * input_size:2 * hidden_size * input_size + hidden_size]
    b2 = theta[2 * hidden_size * input_size + hidden_size:]

    return W1, b1


W1_rgb, b1_rgb = load_chen_16_rgb()


def chen_16_rgb(patch):
    """ Return descriptor for 16x16x3 patches. """
    patch_size = patch.shape[0]
    data = np.expand_dims(patch.reshape(patch_size * patch_size * 3), axis=1)
    z2 = W1_rgb.dot(data) + np.tile(b1_rgb, (1, 1)).transpose()
    patch_descr = sigmoid(z2)
    return patch_descr


# Chen et al. version for grayscale images

def load_chen_16():
    input_size = 16 * 16
    hidden_size = 128

    theta = np.load('encoderChenEtAl_400it.npy')

    W1 = theta[0:hidden_size * input_size].reshape(hidden_size, input_size)
    W2 = theta[hidden_size * input_size:2 * hidden_size * input_size].reshape(input_size, hidden_size)
    b1 = theta[2 * hidden_size * input_size:2 * hidden_size * input_size + hidden_size]
    b2 = theta[2 * hidden_size * input_size + hidden_size:]

    return W1, b1


W1, b1 = load_chen_16()


def chen_16(patch):
    """ Return descriptor for 16x16x3 patches. """
    patch_size = patch.shape[0]
    data = np.expand_dims(patch.reshape(patch_size * patch_size * 3), axis=1)
    z2 = W1.dot(data) + np.tile(b1, (1, 1)).transpose()
    patch_descr = sigmoid(z2)
    return patch_descr


# TODO
# # SIFT for grayscale images
#
# def cv_sift():
#     keypoint = cv.KeyPoint((patch_size - 1) / 2, (patch_size - 1) / 2, _size=patch_size)
#     keypoints = [keypoint]
#     sift = cv.xfeatures2d.SIFT_create()
