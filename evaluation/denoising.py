from utils.comparisons import calculate_ssd, calculate_psnr
from utils.noise import add_gaussian_noise
from ae_descriptor import init_IR_128, compute_IR, compute_descriptor_from_IR
from other_descriptors.other_descriptors import compute_chen_rgb

import numpy as np
import imageio
import sys

from operator import itemgetter

image_path = '/home/niaki/Downloads/house_modern.jpg'

noise_level = 30

patch_size = 16
half_patch_size = patch_size // 2

nr_similar_patches = 10
# query_stride = 1
# compare_stride = 1
stride = 1
comparing_window_size = 29
eps = 0.0001


image = imageio.imread(image_path)

# to be deleted
# image_crop_size = 64 #64
# image_crop_x_coord = 40 #40 # 30 #33
# image_crop_y_coord = 63 #63 # 22 #63
# image = image[image_crop_x_coord : image_crop_x_coord + image_crop_size, image_crop_y_coord : image_crop_y_coord + image_crop_size, :]

image_height = image.shape[0]
image_width = image.shape[1]

image_noisy = add_gaussian_noise(image, sigma=noise_level)
image_noisy = image_noisy / 255.

encoder_IR, encoder_mp = init_IR_128(image_height, image_width, patch_size)

image_noisy_IR = compute_IR(image_noisy, encoder_IR)



def make_kernel_from_half_size(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        value = 1 / (2 * d + 1) ** 2
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                kernel[f + 1 - i - 1, f + 1 - j - 1] = kernel[f + 1 - i - 1, f + 1 - j - 1] + value
    kernel = kernel / f
    return kernel

def make_kernel(size):
    if size % 2 == 0:
        f = size // 2 - 1
        kernel_temp = make_kernel_from_half_size(f)
        kernel = np.zeros((size, size))
        kernel[ : size - 1,  : size - 1] = kernel_temp
        for i in range(size):
            kernel[size - 1, i] = kernel[size - 2, i]
            kernel[i, size - 1] = kernel[i, size - 2]
        kernel[size - 1, size - 1] = kernel[size - 2, size - 2]
    else:
        f = size // 2
        kernel = make_kernel_from_half_size(f)
    return kernel


def find_similar_patches_with_IR():

    kernel1D = make_kernel(patch_size)
    kernel = np.repeat(kernel1D, 32, axis=1).reshape((patch_size, patch_size, 32))


    diffs = {}
    x_coords = {}
    y_coords = {}

    total_nr_patches = len(range(0, image_width - patch_size + 1, stride)) * len(
        range(0, image_height - patch_size + 1, stride))

    # initialise the dictionaries that store the differences
    counter_query_patches = 0
    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            diffs[counter_query_patches] = {}
            counter_query_patches += 1

    counter_query_patches = 0

    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            #         print(counter_query_patches)
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_patches))

            # query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            # query_patch = np.multiply(query_patch, kernel)
            # query_patch_descr = encoder_patch.predict(np.expand_dims(query_patch, axis=0))[0]

            # query_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_query, y_query, patch_size, encoder_mp)
            query_patch = image_noisy_IR[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            query_patch = np.multiply(query_patch, kernel)
            query_patch_descr = encoder_mp.predict(np.expand_dims(query_patch, axis=0))[0]

            x_coords[counter_query_patches] = x_query
            y_coords[counter_query_patches] = y_query

            #         diffs[counter_query_patches] = {}

            counter_compare_patches = 0

            for y_compare in range(0, image_width - patch_size + 1, stride):
                for x_compare in range(0, image_height - patch_size + 1, stride):

                    if counter_query_patches < counter_compare_patches and abs(
                            x_compare - x_query) < comparing_window_size and abs(
                            y_compare - y_query) < comparing_window_size:

                        # compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        # compare_patch = np.multiply(compare_patch, kernel)
                        # compare_patch_descr = encoder_patch.predict(np.expand_dims(compare_patch, axis=0))[0]

                        # compare_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_compare, y_compare, patch_size, encoder_mp)
                        compare_patch = image_noisy_IR[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        compare_patch = np.multiply(compare_patch, kernel)
                        compare_patch_descr = encoder_mp.predict(np.expand_dims(compare_patch, axis=0))[0]

                        query_compare_diff = calculate_ssd(query_patch_descr, compare_patch_descr)

                        diffs[counter_query_patches][counter_compare_patches] = query_compare_diff
                        diffs[counter_compare_patches][counter_query_patches] = query_compare_diff

                    counter_compare_patches += 1

            counter_query_patches += 1

    return diffs, x_coords, y_coords


def find_similar_patches_with_Chen():

    kernel1D = make_kernel(patch_size)
    kernel = np.repeat(kernel1D, 3, axis=1).reshape((patch_size, patch_size, 3))

    diffs = {}
    x_coords = {}
    y_coords = {}

    total_nr_patches = len(range(0, image_width - patch_size + 1, stride)) * len(
        range(0, image_height - patch_size + 1, stride))

    # initialise the dictionaries that store the differences
    counter_query_patches = 0
    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            diffs[counter_query_patches] = {}
            counter_query_patches += 1

    counter_query_patches = 0

    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            #         print(counter_query_patches)
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_patches))

            # query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            # query_patch = np.multiply(query_patch, kernel)
            # query_patch_descr = encoder_patch.predict(np.expand_dims(query_patch, axis=0))[0]

            # query_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_query, y_query, patch_size, encoder_mp)
            query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            query_patch = np.multiply(query_patch, kernel)
            query_patch_descr = compute_chen_rgb(query_patch)

            x_coords[counter_query_patches] = x_query
            y_coords[counter_query_patches] = y_query

            #         diffs[counter_query_patches] = {}

            counter_compare_patches = 0

            for y_compare in range(0, image_width - patch_size + 1, stride):
                for x_compare in range(0, image_height - patch_size + 1, stride):

                    if counter_query_patches < counter_compare_patches and abs(
                            x_compare - x_query) < comparing_window_size and abs(
                            y_compare - y_query) < comparing_window_size:

                        # compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        # compare_patch = np.multiply(compare_patch, kernel)
                        # compare_patch_descr = encoder_patch.predict(np.expand_dims(compare_patch, axis=0))[0]

                        # compare_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_compare, y_compare, patch_size, encoder_mp)
                        compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        compare_patch = np.multiply(compare_patch, kernel)
                        compare_patch_descr = compute_chen_rgb(compare_patch)

                        query_compare_diff = calculate_ssd(query_patch_descr, compare_patch_descr)

                        diffs[counter_query_patches][counter_compare_patches] = query_compare_diff
                        diffs[counter_compare_patches][counter_query_patches] = query_compare_diff

                    counter_compare_patches += 1

            counter_query_patches += 1

    return diffs, x_coords, y_coords


def find_similar_patches_with_exhaustive_search():

    kernel1D = make_kernel(patch_size)
    kernel = np.repeat(kernel1D, 3, axis=1).reshape((patch_size, patch_size, 3))

    diffs = {}
    x_coords = {}
    y_coords = {}

    total_nr_patches = len(range(0, image_width - patch_size + 1, stride)) * len(
        range(0, image_height - patch_size + 1, stride))

    # initialise the dictionaries that store the differences
    counter_query_patches = 0
    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            diffs[counter_query_patches] = {}
            counter_query_patches += 1

    counter_query_patches = 0

    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            #         print(counter_query_patches)
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_patches))

            # query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            # query_patch = np.multiply(query_patch, kernel)
            # query_patch_descr = encoder_patch.predict(np.expand_dims(query_patch, axis=0))[0]

            # query_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_query, y_query, patch_size, encoder_mp)
            query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            query_patch = np.multiply(query_patch, kernel)
            # query_patch_descr = compute_chen_rgb(query_patch)

            x_coords[counter_query_patches] = x_query
            y_coords[counter_query_patches] = y_query

            #         diffs[counter_query_patches] = {}

            counter_compare_patches = 0

            for y_compare in range(0, image_width - patch_size + 1, stride):
                for x_compare in range(0, image_height - patch_size + 1, stride):

                    if counter_query_patches < counter_compare_patches and abs(
                            x_compare - x_query) < comparing_window_size and abs(
                            y_compare - y_query) < comparing_window_size:

                        # compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        # compare_patch = np.multiply(compare_patch, kernel)
                        # compare_patch_descr = encoder_patch.predict(np.expand_dims(compare_patch, axis=0))[0]

                        # compare_patch_descr = compute_descriptor_from_IR(image_noisy_IR, x_compare, y_compare, patch_size, encoder_mp)
                        compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                        compare_patch = np.multiply(compare_patch, kernel)
                        # compare_patch_descr = compute_chen_rgb(compare_patch)

                        query_compare_diff = calculate_ssd(query_patch, compare_patch)

                        diffs[counter_query_patches][counter_compare_patches] = query_compare_diff
                        diffs[counter_compare_patches][counter_query_patches] = query_compare_diff

                    counter_compare_patches += 1

            counter_query_patches += 1

    return diffs, x_coords, y_coords



def create_denoised_image_from_similar_patches(diffs, x_coords, y_coords):

    image_denoised = np.zeros_like(image_noisy)

    total_nr_patches = len(range(0, image_width - patch_size + 1, stride)) * len(
        range(0, image_height - patch_size + 1, stride))
    counter_query_patches = 0

    for y_query in range(0, image_width - patch_size + 1, stride):
        for x_query in range(0, image_height - patch_size + 1, stride):
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_patches))

            compare_patches = np.ones((nr_similar_patches, patch_size, patch_size, 3))

            i = 0

            for id_compare, _ in sorted(diffs[counter_query_patches].items(), key=itemgetter(1), reverse=False):
                #             print(key, value)

                #         for i in range(nearest_patches_nb):

                x_compare = x_coords[id_compare]
                y_compare = y_coords[id_compare]
                compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]
                compare_patch1 = np.expand_dims(compare_patch, axis=0)
                compare_patches[i] = compare_patch1

                i += 1
                if i >= nr_similar_patches:
                    break

            compare_patches_mean = np.mean(compare_patches, axis=0)

            image_denoised[x_query + half_patch_size, y_query + half_patch_size, :] = compare_patches_mean[
                                                                                      half_patch_size, half_patch_size,
                                                                                      :]

            counter_query_patches += 1

    return image_denoised


def denoise_image_with_IR():
    diffs, x_coords, y_coords = find_similar_patches_with_IR()
    image_denoised = create_denoised_image_from_similar_patches(diffs, x_coords, y_coords)

    im1 = image_denoised[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im2 = image_noisy[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = image[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = im3 / 255.

    print("denoised and clean")
    psnr_denoised = calculate_psnr(im1, im3, 1)
    print("psnr:" + str(psnr_denoised))

    print("noisy and clean")
    psnr_noisy = calculate_psnr(im2, im3, 1)
    print("psnr:" + str(psnr_noisy))

    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_clean.jpg', im3)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_denoised' + str(noise_level) + '_AE128_PSNR_' + str(psnr_denoised) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im1)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_noisy' + str(noise_level) + '_PSNR_' + str(psnr_noisy) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im2)


def denoise_image_with_with_Chen():
    diffs, x_coords, y_coords = find_similar_patches_with_Chen()
    image_denoised = create_denoised_image_from_similar_patches(diffs, x_coords, y_coords)

    im1 = image_denoised[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im2 = image_noisy[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = image[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = im3 / 255.

    print("denoised and clean")
    psnr_denoised = calculate_psnr(im1, im3, 1)
    print("psnr:" + str(psnr_denoised))

    print("noisy and clean")
    psnr_noisy = calculate_psnr(im2, im3, 1)
    print("psnr:" + str(psnr_noisy))

    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_clean.jpg', im3)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_denoised' + str(noise_level) + '_chen_PSNR_' + str(psnr_denoised) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im1)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_noisy' + str(noise_level) + '_PSNR_' + str(psnr_noisy) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im2)


def denoise_image_with_exhaustive_search():
    diffs, x_coords, y_coords = find_similar_patches_with_exhaustive_search()
    image_denoised = create_denoised_image_from_similar_patches(diffs, x_coords, y_coords)

    im1 = image_denoised[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im2 = image_noisy[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = image[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size, :]
    im3 = im3 / 255.

    print("denoised and clean")
    psnr_denoised = calculate_psnr(im1, im3, 1)
    print("psnr:" + str(psnr_denoised))

    print("noisy and clean")
    psnr_noisy = calculate_psnr(im2, im3, 1)
    print("psnr:" + str(psnr_noisy))

    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_clean.jpg', im3)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_denoised' + str(noise_level) + '_exhsearch_PSNR_' + str(psnr_denoised) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im1)
    imageio.imwrite('/home/niaki/Downloads/house_modern_dnsingv4_kernel_noisy' + str(noise_level) + '_PSNR_' + str(psnr_noisy) + '_' + str(nr_similar_patches) + 'simpatch.jpg', im2)





def main():
    # denoise_image_with_IR()
    # denoise_image_with_with_Chen()
    denoise_image_with_exhaustive_search()

if __name__ == '__main__':
    main()