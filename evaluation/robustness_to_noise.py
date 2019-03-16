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

image_path = '/home/niaki/PycharmProjects/patch-desc-ae/images/Lenna.png'
patch_size = 16
nr_similar_patches = 40
query_stride = 100
compare_stride = 8
eps = 0.0001
results_dir = '/home/niaki/PycharmProjects/patch-desc-ae/results'


image = imageio.imread(image_path)
image_height = image.shape[0]
image_width = image.shape[1]

encoder32 = init_descr_32(16)
encoder128 = init_descr_128(16)

total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

#TODO add function calculate_ssd/calculate_psnr as a parameter to this function

def calculate_SSDs_for_desc_and_noise_level(which_desc, noise_level):

    image_noisy = add_gaussian_noise(image, sigma=noise_level)

    query_x_coords = []
    query_y_coords = []

    results_noisy_descr_patches_diffs = {}
    results_noisy_descr_patches_x_coords = {}
    results_noisy_descr_patches_y_coords = {}
    results_noisy_descr_patches_positions = {}

    counter_query_patches = 0

    # just for the sake of output
    total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

    for y_query in range(0, image_width - patch_size + 1, query_stride):
        for x_query in range(0, image_height - patch_size + 1, query_stride):
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_query_patches))

            query_x_coords.append(x_query)
            query_y_coords.append(y_query)

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

            results_noisy_descr_patches_diffs[counter_query_patches] = patches_diffs[:nr_similar_patches]
            results_noisy_descr_patches_x_coords[counter_query_patches] = patches_x_coords[:nr_similar_patches]
            results_noisy_descr_patches_y_coords[counter_query_patches] = patches_y_coords[:nr_similar_patches]
            results_noisy_descr_patches_positions[counter_query_patches] = patches_positions[:nr_similar_patches]

            counter_query_patches += 1

    ssds = []

    for q_it in range(total_nr_query_patches):
        for c_it in range(nr_similar_patches):

            # getting the query patch from the clean image
            x_query = query_x_coords[q_it]
            y_query = query_y_coords[q_it]
            query_patch = image[x_query: x_query + patch_size, y_query: y_query + patch_size, :]

            # getting the compare patch from the clean image
            x_compare = results_noisy_descr_patches_x_coords[q_it][c_it]
            y_compare = results_noisy_descr_patches_y_coords[q_it][c_it]
            compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

            # calculating the difference in the clean image
            actual_diff = calculate_ssd(query_patch, compare_patch)
            ssds.append(actual_diff)

    ssds = np.array(ssds)
    return ssds


def compute_SSDs(which_descs, noise_levels):

    ssds_by_model = {}
    which_descs = [0, 1, 2, 3]
    noise_levels = [0, 10, 20, 30, 40, 50]

    for which_desc in which_descs:

        print("which desc? " + str(which_desc))
        ssds_by_model[which_desc] = {}

        for noise_level in noise_levels:
            print('--- Noise level ' + str(noise_level) + ' ---')

            ssds_by_model[which_desc][noise_level] = calculate_SSDs_for_desc_and_noise_level(which_desc, noise_level)

    noise_levels_string = "_".join(str(noise_level) for noise_level in noise_levels)
    which_descs_string = "_".join(str(which_desc) for which_desc in which_descs)
    ssds_by_model_file_path = results_dir + '/ssds_noise_levels_' + noise_levels_string + '__descrs_' + which_descs_string + '__' + str(total_nr_query_patches) + '_query_patches.pkl'

    with open(ssds_by_model_file_path, 'wb') as f:
        pickle.dump(ssds_by_model, f)

    return ssds_by_model

def plot_SSDs_for_4_descrs(ssds_by_model, noise_levels, ssds_by_model_plot_file_path):
    font = {'family': 'sans-serif',
            'weight': 'medium',
            'size': 23}

    matplotlib.rc('font', **font)

    # noise_levels = [0, 10, 20, 30, 40, 50]

    # with open(ssds_by_model_file_path, 'rb') as f:
    #     ssds_by_model = pickle.load(f)

    fig = plt.figure(figsize=(25, 10))

    for noise_level in noise_levels:
        data = []
        labels = []
        data.append(ssds_by_model[0][noise_level])
        data.append(ssds_by_model[1][noise_level])
        data.append(ssds_by_model[2][noise_level])
        data.append(ssds_by_model[3][noise_level])
        labels.append('A')
        labels.append('B')
        labels.append('C')
        labels.append('D')
        ax = fig.add_subplot(1, 6, noise_level // 10 + 1)
        if noise_level == 0:
            ax.set_ylabel('SSD', fontsize=23)
        ax.set_yscale('log')
        ax.set_title('Noise σ = ' + str(noise_level))
        # ax.set_ylim([18000, 12000000])
        ax.boxplot(data, labels=labels)#, fontsize=18)

    fig.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.35, hspace=None)
    fig.savefig(ssds_by_model_plot_file_path + '.pdf', bbox_inches='tight')
    fig.savefig(ssds_by_model_plot_file_path + '.jpg', bbox_inches='tight')

    fig.show()

    plt.show(block=True)
    plt.interactive(False)


def compute_and_plot_SSDs():
    which_descs = [0, 1, 2, 3]
    noise_levels = [0, 10, 20, 30, 40, 50]

    ssds_by_model = compute_SSDs(which_descs, noise_levels)
    plot_SSDs_for_4_descrs(ssds_by_model, noise_levels)



def main():
    print()
    # compute_SSDs(0, 0)

    ssds_by_model_plot_file_path = '/home/niaki/PycharmProjects/patch-desc-ae/results/ssds_noise_levels_0_10_20_30_40_50__descrs_0_1_2_3__25_query_patches'

    noise_levels = [0, 10, 20, 30, 40, 50]
    ssds_by_model_file_path = '/home/niaki/PycharmProjects/patch-desc-ae/results/ssds_noise_levels_0_10_20_30_40_50__descrs_0_1_2_3__25_query_patches.pkl'
    with open(ssds_by_model_file_path, 'rb') as f:
        ssds_by_model = pickle.load(f)
    plot_SSDs_for_4_descrs(ssds_by_model, noise_levels, ssds_by_model_plot_file_path)


if __name__ == "__main__":
    main()
