from ae_descriptor import init_descr_32, init_descr_128, compute_descriptor
from other_descriptors.other_descriptors import compute_chen_rgb
from utils.comparisons import calculate_ssd, calculate_psnr

import imageio
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib


image_path = '/home/niaki/Code/Lenna.png'

nr_similar_patches = 40
patch_size = 16
patch_width = patch_size
patch_height = patch_size

query_stride = 32
compare_stride = 16
eps = 0.0001


image = imageio.imread(image_path)

image = image / 255.

image_height = image.shape[0]
image_width = image.shape[1]


encoder32 = init_descr_32(16)
encoder128 = init_descr_128(16)


def find_similar_patches_for_query_patches(which_desc):

    # we want all the possible patches, sorted by similarity to the query patch
    nearest_patches_nb = len(range(0, image_width - patch_size + 1, compare_stride)) * len(
        range(0, image_height - patch_size + 1, compare_stride))

    results_patches_diffs = {}
    results_patches_x_coords = {}
    results_patches_y_coords = {}
    results_patches_positions = {}

    counter_query_patches = 0

    # just for the sake of output
    total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

    for y_query in range(0, image_width - patch_size + 1, query_stride):
        for x_query in range(0, image_height - patch_size + 1, query_stride):
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_query_patches))

            query_patch = image[x_query: x_query + patch_size, y_query: y_query + patch_size, :]

            if which_desc == 0:
                query_patch_descr = compute_descriptor(query_patch, encoder32)
            elif which_desc == 1:
                query_patch_descr = compute_descriptor(query_patch, encoder128)
            elif which_desc == 2:
                query_patch_descr = compute_chen_rgb(query_patch)
            else:
                query_patch_descr = query_patch

            counter_compare_patches = 0
            compare_patches_scores = {}

            patches_diffs = [1000000000]
            patches_x_coords = [-1]
            patches_y_coords = [-1]
            patches_positions = [-1]

            for y_compare in range(0, image_width - patch_size + 1, compare_stride):
                for x_compare in range(0, image_height - patch_size + 1, compare_stride):

                    compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

                    if which_desc == 0:
                        compare_patch_descr = compute_descriptor(compare_patch, encoder32)
                    elif which_desc == 1:
                        compare_patch_descr = compute_descriptor(compare_patch, encoder128)
                    elif which_desc == 2:
                        compare_patch_descr = compute_chen_rgb(compare_patch)
                    else:
                        compare_patch_descr = compare_patch

                    diff = calculate_ssd(query_patch_descr, compare_patch_descr)
                    #                 diff = (query_patch_descr - compare_patch_descr)**2

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

            results_patches_diffs[counter_query_patches] = patches_diffs[:nearest_patches_nb - 1]
            results_patches_x_coords[counter_query_patches] = patches_x_coords[:nearest_patches_nb - 1]
            results_patches_y_coords[counter_query_patches] = patches_y_coords[:nearest_patches_nb - 1]
            results_patches_positions[counter_query_patches] = patches_positions[:nearest_patches_nb - 1]

            counter_query_patches += 1

    return results_patches_diffs, results_patches_positions



def classify_patches_and_save_to_file(results_patches_positions, results_descr_patches_positions, results_descr_patches_diffs, filename_descr):


    counter = 0

    ground_truths = []
    predictions = []

    for i in range(len(results_patches_positions)):

        denominator = results_descr_patches_diffs[i][-2] - results_descr_patches_diffs[i][0]
        offset = results_descr_patches_diffs[i][0] / denominator

        fraction_of_one = 1.0 / len(results_patches_positions[0])

        for j in range(len(results_patches_positions[0])):

            prediction = 1 - (results_descr_patches_diffs[i][j] / denominator - offset) #change this, base it on equidistant numbers between 0 and 1
            # prediction = 1.0 - j * fraction_of_one

            if results_descr_patches_positions[i][j] in results_patches_positions[i][:nr_similar_patches]:
                ground_truth = 1
            else:
                ground_truth = 0

            ground_truths.append(ground_truth)
            predictions.append(prediction)

            counter += 1

    np.save('/home/niaki/Downloads/ROC_thresh' + str(nr_similar_patches) + '_ground_truths.npy', np.array(ground_truths))
    np.save('/home/niaki/Downloads/ROC_thresh' + str(nr_similar_patches) + '_predictions' + filename_descr + '.npy', np.array(predictions))



def main():

    results_patches_diffs_v128, results_patches_positions_v128 = find_similar_patches_for_query_patches(1)
    results_patches_diffs_chen, results_patches_positions_chen = find_similar_patches_for_query_patches(2)
    results_patches_diffs_exhsrch, results_patches_positions_exhsrch = find_similar_patches_for_query_patches(3)

    classify_patches_and_save_to_file(results_patches_positions_exhsrch, results_patches_positions_v128,
                                      results_patches_diffs_v128, 'v128')
    classify_patches_and_save_to_file(results_patches_positions_exhsrch, results_patches_positions_chen,
                                      results_patches_diffs_chen, 'chen')


if __name__ == '__main__':
    main()