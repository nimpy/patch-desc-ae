from ae_descriptor import init_descr_32, init_descr_128, compute_descriptor
from other_descriptors.other_descriptors import compute_chen_rgb
from utils.comparisons import calculate_ssd
from utils.noise import add_gaussian_noise

import imageio
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib
import datetime


# image_path = '/home/niaki/Downloads/Lenna.png'
# image_path = '/scratch/data/panel13/panel13_cropped2.png'
image_path = '/home/niaki/Downloads/clean.jpg'

patch_size = 16
patch_width = patch_size
patch_height = patch_size

nr_similar_patches = 5
query_stride = 100
compare_stride = 8
eps = 0.0001

random_seed = 120


image = imageio.imread(image_path)
image_height = image.shape[0]
image_width = image.shape[1]
psnr_max_value = 255

noise_level = 10
image_noisy = add_gaussian_noise(image, sigma=noise_level)


encoder32 = init_descr_32(16)
encoder128 = init_descr_128(16)



###
image_noisy = image_noisy / 255.

image = image / 255.
psnr_max_value = 1


def generate_visualisation_for_3_descrs(x_queries, y_queries, results_patches_x_coords_0, results_patches_y_coords_0,
                                        results_patches_x_coords_1, results_patches_y_coords_1,
                                        results_patches_x_coords_2, results_patches_y_coords_2):
    y_offset_under = -0.2
    font_size = 18
    x_offset_left = -2.5
    y_offset_left = 15

    fig = plt.figure(figsize=(17, 8))

    total_nr_query_patches = len(x_queries)

    columns = nr_similar_patches + 1
    rows = total_nr_query_patches * 3

    counter_query_patches = 0

    for query_it in range(total_nr_query_patches):

        x_query = x_queries[query_it]
        y_query = y_queries[query_it]
        patch_query = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]

        ax = fig.add_subplot(rows, columns, (counter_query_patches * 3) * (nr_similar_patches + 1) + 1)
        ax.axis('off')
        ax.set_title('query', y=y_offset_under, fontsize=font_size)  # + str(query_it + 1)
        ax.imshow(patch_query)

        for i in range(nr_similar_patches):
            x_compare = results_patches_x_coords_0[counter_query_patches][i]
            y_compare = results_patches_y_coords_0[counter_query_patches][i]

            # psnr = calculate_psnr(image[x_query: x_query + patch_size, y_query: y_query + patch_size, :],
            #                       image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :],
            #                       max_value=psnr_max_value)

            patch_compare = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

            ax = fig.add_subplot(rows, columns, (counter_query_patches * 3) * (nr_similar_patches + 1) + 2 + i)
            ax.axis('off')
            if i == 0:
                # ax.text(x_offset_left, 1, 'proposed v128', rotation=90, fontsize=font_size)
                ax.text(x_offset_left, y_offset_left, 'proposed v128', rotation=90, fontsize=font_size)  # y_offset_left
            # ax.set_title("{:.2f} [dB]".format(psnr), y=y_offset_under, fontsize=font_size)
            ax.imshow(patch_compare)

        for i in range(nr_similar_patches):
            x_compare = results_patches_x_coords_1[counter_query_patches][i]
            y_compare = results_patches_y_coords_1[counter_query_patches][i]

            # psnr = calculate_psnr(image[x_query: x_query + patch_size, y_query: y_query + patch_size, :],
            #                       image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :],
            #                       max_value=psnr_max_value)

            patch_compare = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

            ax = fig.add_subplot(rows, columns, ((counter_query_patches * 3) + 1) * (nr_similar_patches + 1) + 2 + i)
            ax.axis('off')
            if i == 0:
                ax.text(x_offset_left, y_offset_left - 2, 'Chen et al.', rotation=90, fontsize=font_size)
            # ax.set_title("{:.2f} [dB]".format(psnr), y=y_offset_under, fontsize=font_size)
            ax.imshow(patch_compare)

        for i in range(nr_similar_patches):
            x_compare = results_patches_x_coords_2[counter_query_patches][i]
            y_compare = results_patches_y_coords_2[counter_query_patches][i]

            # psnr = calculate_psnr(image[x_query: x_query + patch_size, y_query: y_query + patch_size, :],
            #                       image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :],
            #                       max_value=psnr_max_value)

            patch_compare = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]

            ax = fig.add_subplot(rows, columns, ((counter_query_patches * 3) + 2) * (nr_similar_patches + 1) + 2 + i)
            ax.axis('off')
            if i == 0:
                ax.text(x_offset_left, y_offset_left - 2, 'exhaustive', rotation=90, fontsize=font_size)
            # ax.set_title("{:.2f} [dB]".format(psnr), y=y_offset_under, fontsize=font_size)
            ax.imshow(patch_compare)

        counter_query_patches += 1

    # fig.savefig("/home/niaki/PycharmProjects/patch-desc-ae/results/Visualisation_v128_chen_exhaustive_q_" + str(x_query) + "_" + str(y_query) + "_noise" + str(noise_level) + ".pdf", bbox_inches='tight')
    fig.savefig("/home/niaki/Downloads/Visualisation_v128_chen_exhaustive_q_" + str(x_query) + "_" + str(
        y_query) + "_noise" + str(noise_level) + "_" + datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S") + ".pdf", bbox_inches='tight')

    fig.show()

    plt.show(block=True)
    plt.interactive(False)


def retrieve_patches_for_queries_and_descr(x_queries, y_queries, which_desc):

    results_patches_diffs = {}
    results_patches_x_coords = {}
    results_patches_y_coords = {}
    results_patches_positions = {}

    counter_query_patches = 0

    total_nr_query_patches = len(x_queries)

    for query_it in range(total_nr_query_patches):

        x_query = x_queries[query_it]
        y_query = y_queries[query_it]

        sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_query_patches))

        query_patch = image_noisy[x_query: x_query + patch_size, y_query: y_query + patch_size, :]

        if which_desc == 0:
            query_patch_descr = compute_descriptor(query_patch, encoder32)
        elif which_desc == 1:
            query_patch_descr = compute_descriptor(query_patch, encoder128)
        elif which_desc == 2:
            query_patch_descr = compute_chen_rgb(query_patch)
        else:
            query_patch_descr = query_patch

        counter_compare_patches = 0

        patches_diffs = [1000000000]
        patches_x_coords = [-1]
        patches_y_coords = [-1]
        patches_positions = [-1]

        for y_compare in range(0, image_width - patch_size + 1, compare_stride):
            for x_compare in range(0, image_height - patch_size + 1, compare_stride):

                compare_patch = image_noisy[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

                if which_desc == 0:
                    compare_patch_descr = compute_descriptor(compare_patch, encoder32)
                elif which_desc == 1:
                    compare_patch_descr = compute_descriptor(compare_patch, encoder128)
                elif which_desc == 2:
                    compare_patch_descr = compute_chen_rgb(compare_patch)
                else:
                    compare_patch_descr = compare_patch

                diff = calculate_ssd(query_patch_descr, compare_patch_descr)

                if diff < eps:
                    counter_compare_patches += 1
                    continue

                # sorting
                for i in range(len(patches_diffs)):
                    if diff < patches_diffs[i]:
                        patches_diffs.insert(i, diff)
                        patches_x_coords.insert(i, x_compare)
                        patches_y_coords.insert(i, y_compare)
                        patches_positions.insert(i, counter_compare_patches)
                        break

                counter_compare_patches += 1

        results_patches_diffs[counter_query_patches] = patches_diffs[:nr_similar_patches]
        results_patches_x_coords[counter_query_patches] = patches_x_coords[:nr_similar_patches]
        results_patches_y_coords[counter_query_patches] = patches_y_coords[:nr_similar_patches]
        results_patches_positions[counter_query_patches] = patches_positions[:nr_similar_patches]

        counter_query_patches += 1

    return results_patches_x_coords, results_patches_y_coords



def main():
    x_queries = [1185] #[9, 58, 315, 26]
    y_queries = [570] #[12, 233, 101, 473]

    results_patches_x_coords_0, results_patches_y_coords_0 = retrieve_patches_for_queries_and_descr(x_queries, y_queries, 1)
    results_patches_x_coords_1, results_patches_y_coords_1 = retrieve_patches_for_queries_and_descr(x_queries, y_queries, 2)
    results_patches_x_coords_2, results_patches_y_coords_2 = retrieve_patches_for_queries_and_descr(x_queries, y_queries, 3)

    generate_visualisation_for_3_descrs(x_queries, y_queries, results_patches_x_coords_0, results_patches_y_coords_0,
                                        results_patches_x_coords_1, results_patches_y_coords_1,
                                        results_patches_x_coords_2, results_patches_y_coords_2)

if __name__ == '__main__':
    main()