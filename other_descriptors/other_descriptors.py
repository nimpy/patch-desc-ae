import numpy as np
from other_descriptors.chen_descriptor_training import sigmoid
# import cv2 as cv


patch_size = 16

models_dir = '/home/niaki/Projects/patch-desc-ae/other_descriptors'

# Chen et al. version for RGB images

def init_chen_rgb():
    input_size = patch_size * patch_size * 3
    hidden_size = 128

    theta = np.load(models_dir + '/encoderChenEtAl_RGB_400it.npy')

    W1 = theta[0:hidden_size * input_size].reshape(hidden_size, input_size)
    W2 = theta[hidden_size * input_size:2 * hidden_size * input_size].reshape(input_size, hidden_size)
    b1 = theta[2 * hidden_size * input_size:2 * hidden_size * input_size + hidden_size]
    b2 = theta[2 * hidden_size * input_size + hidden_size:]

    return W1, b1


W1_rgb, b1_rgb = init_chen_rgb()


def compute_chen_rgb(patch):
    """ Return descriptor for a 16x16x3 patch. """
    patch_size = patch.shape[0]
    data = np.expand_dims(patch.reshape(patch_size * patch_size * 3), axis=1)
    z2 = W1_rgb.dot(data) + np.tile(b1_rgb, (1, 1)).transpose()
    patch_descr = sigmoid(z2)
    return patch_descr


# Chen et al. version for grayscale images

def init_chen():
    input_size = patch_size * patch_size
    hidden_size = 128

    theta = np.load(models_dir + '/encoderChenEtAl_400it.npy')

    W1 = theta[0:hidden_size * input_size].reshape(hidden_size, input_size)
    # W2 = theta[hidden_size * input_size:2 * hidden_size * input_size].reshape(input_size, hidden_size)
    b1 = theta[2 * hidden_size * input_size:2 * hidden_size * input_size + hidden_size]
    # b2 = theta[2 * hidden_size * input_size + hidden_size:]

    return W1, b1


W1, b1 = init_chen()


def compute_chen(patch):
    """ Return descriptor for a 16x16 grayscale patch. """
    patch_size = patch.shape[0]
    data = np.expand_dims(patch.reshape(patch_size * patch_size * 3), axis=1)
    z2 = W1.dot(data) + np.tile(b1, (1, 1)).transpose()
    patch_descr = sigmoid(z2)
    return patch_descr


# # SIFT for grayscale images
#
# def init_cv_sift():
#     keypoint = cv.KeyPoint((patch_size - 1) / 2, (patch_size - 1) / 2, _size=patch_size)
#     keypoints = [keypoint]
#     sift = cv.xfeatures2d.SIFT_create()
#     return sift, keypoints
#
#
# cv_sift_16, cv_keypoints_16 = init_cv_sift()
#
#
# def compute_cv_sift_16(patch):
#     _, patch_descr = cv_sift_16.compute(patch, cv_keypoints_16)
#     return patch_descr
