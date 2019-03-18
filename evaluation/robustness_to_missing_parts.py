from ae_descriptor import init_descr_32, init_descr_128, compute_descriptor
from other_descriptors.other_descriptors import compute_chen_rgb
from utils.comparisons import calculate_ssd
from utils.partially_obscuring import mask_ising_model, mask_random_border_rectangle, mask_random_corner_rectangle, mask_of_specific_percentage

import imageio
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib

image_path = '/home/niaki/PycharmProjects/patch-desc-ae/images/Lenna.png'
patch_size = 16
nr_similar_patches = 40
query_stride = 32
compare_stride = 8
eps = 0.0001
results_dir = '/home/niaki/PycharmProjects/patch-desc-ae/results'
results_version = '4'

image = imageio.imread(image_path)
image_height = image.shape[0]
image_width = image.shape[1]

encoder32 = init_descr_32(16)
encoder128 = init_descr_128(16)

seeds_for_ising = {}
seeds_for_ising[0.1] = [22,39,180,333,398,486,502,553,561,588,646,674,773,786,811,849,925,954,989]
seeds_for_ising[0.2] = [0,6,16,18,38,41,56,61,62,64,73,91,93,99,103,110,127,143,145,149,158,169,170,188,206,215,227,249,255,266,283,288,294,295,296,300,305,306,319,339,355,360,372,373,379,385,390,393,394,404,410,418,421,424,426,443,447,456,457,458,460,463,466,467,468,473,476,493,495,501,515,537,538,544,565,579,590,591,595,603,626,627,631,635,647,653,656,658,662,670,684,708,713,730,736,740,761,770,774,778,795,805,807,822,823,836,853,863,865,879,891,894,898,912,926,927,938,944,945,957,965]
seeds_for_ising[0.3] = [1,2,5,13,21,27,30,37,45,50,52,55,57,63,66,67,68,70,75,80,81,87,89,92,98,100,106,111,114,115,121,128,135,136,141,142,146,150,152,157,159,161,162,168,174,177,183,186,190,192,197,205,207,219,229,230,232,233,235,236,240,241,245,247,256,261,265,273,274,280,281,282,287,291,304,307,313,315,317,321,324,327,330,337,340,346,348,349,352,359,366,374,376,399,401,406,408,409,420,425,434,436,439,446,451,452,454,469,479,483,490,496,499,500,504,510,511,512,516,519,520,522,523,525,527,530,535,539,540,541,550,556,559,569,572,573,574,583,594,596,597,601,606,607,609,610,613,619,623,633,634,637,638,640,645,655,657,663,668,669,677,680,682,687,691,692,697,700,707,709,714,718,720,721,724,727,731,735,738,743,745,746,755,765,768,780,781,783,787,788,792,798,802,804,820,828,837,839,841,842,843,847,848,851,854,855,857,861,867,868,870,872,875,876,877,880,885,892,897,902,904,911,913,914,915,918,931,932,936,939,946,949,950,951,955,967,968,975,976,977,986,990,994,998]
seeds_for_ising[0.4] = [4,8,10,12,14,17,20,23,24,25,26,28,31,32,42,44,46,48,49,51,53,54,58,69,72,74,77,78,86,96,101,105,107,108,109,112,117,118,119,122,124,125,126,129,130,132,133,134,137,139,140,144,151,156,160,163,164,167,172,173,175,178,179,181,184,185,187,189,191,193,194,196,200,203,204,209,212,216,217,220,221,223,224,226,231,234,237,238,239,242,243,244,248,250,251,252,253,254,259,260,262,263,264,267,269,276,278,284,293,299,301,309,311,320,322,325,326,328,331,338,342,343,344,351,353,354,357,361,362,364,365,367,368,371,375,377,378,381,382,383,384,387,388,389,391,392,395,407,411,412,413,414,415,416,417,419,423,427,429,430,433,437,438,441,442,444,445,449,450,461,462,464,465,472,477,480,482,484,487,488,489,491,494,497,498,503,505,506,508,509,514,518,526,528,533,542,547,548,551,552,554,555,562,563,566,570,571,575,576,577,578,580,581,584,585,586,589,593,598,599,602,604,605,608,611,614,615,620,622,624,628,629,632,639,641,643,644,649,651,652,654,659,660,664,665,667,672,673,675,676,678,681,683,688,693,694,695,696,698,702,705,706,711,712,717,722,725,728,733,734,737,739,741,742,744,748,749,751,752,753,756,757,760,763,764,766,769,771,772,775,776,777,779,782,785,789,793,794,799,800,801,806,808,812,816,818,819,821,824,825,826,827,829,833,834,835,838,840,844,845,846,852,856,864,866,871,873,874,878,881,883,887,888,890,893,896,899,900,903,905,907,908,910,920,921,922,923,924,929,930,934,935,937,940,942,943,947,948,953,956,958,960,963,964,966,969,974,978,979,981,984,985,987,988,991,995,997]
seeds_for_ising[0.5] = [7,9,11,15,19,29,33,34,35,36,40,43,47,59,60,65,71,76,79,82,83,84,85,88,90,94,95,97,102,104,113,116,120,123,131,138,147,148,153,154,155,165,166,171,176,182,195,198,199,201,202,208,210,211,213,214,218,222,225,228,246,257,258,268,270,271,272,275,277,279,285,286,289,290,292,297,298,302,303,308,310,312,314,316,318,323,329,332,334,335,336,341,345,347,350,356,358,363,369,370,380,386,396,397,400,402,403,405,422,428,431,432,435,440,448,453,455,459,470,471,475,478,481,485,492,507,513,517,521,524,529,531,532,534,536,543,545,546,549,557,558,560,564,567,568,582,587,592,600,612,616,617,618,621,625,630,636,642,648,650,661,666,671,679,685,686,689,690,699,701,703,704,710,715,716,719,723,726,729,732,747,750,754,758,759,762,767,784,790,791,796,797,803,809,810,813,814,815,817,830,831,832,850,858,859,860,862,869,882,884,886,889,895,901,906,909,916,917,919,928,933,941,952,959,961,962,970,971,972,973,980,982,983,992,993,996,999]

total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

#TODO add function calculate_ssd/calculate_psnr as a parameter to this function

def calculate_SSDs_for_desc_and_missing_level(which_desc, mask_perc):
    query_x_coords = []
    query_y_coords = []

    results_noisy_descr_patches_diffs = {}
    results_noisy_descr_patches_x_coords = {}
    results_noisy_descr_patches_y_coords = {}
    results_noisy_descr_patches_positions = {}

    counter_query_patches = 0
    seed_nr = 0

    # just for the sake of output
    total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

    for y_query in range(0, image_width - patch_size + 1, query_stride):
        for x_query in range(0, image_height - patch_size + 1, query_stride):
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_query_patches))

            query_x_coords.append(x_query)
            query_y_coords.append(y_query)

            # if there the missing part percentage is 0
            if mask_perc == 0:
                inverse_mask = np.ones((patch_size, patch_size, 3), dtype=np.uint8)
            else:
                # alternate between random corner, random border, and ising mask
                if counter_query_patches % 3 == 0:
                    seed = seeds_for_ising[mask_perc][seed_nr]
                    seed_nr += 1
                    if seed_nr == (len(seeds_for_ising[mask_perc])):
                        seed_nr = 0
                    np.random.seed(seed * 10)
                    mask = mask_ising_model(patch_size=patch_size)

                if counter_query_patches % 3 == 1:
                    mask = mask_of_specific_percentage(mask_perc - 0.05, mask_perc + 0.05, mask_random_border_rectangle)
                if counter_query_patches % 3 == 2:
                    mask = mask_of_specific_percentage(mask_perc - 0.05, mask_perc + 0.05, mask_random_corner_rectangle)

                inverse_mask = np.repeat((1 - mask), 3, axis=1).reshape((patch_size, patch_size, 3))

            query_patch = image[x_query: x_query + patch_size, y_query: y_query + patch_size, :]
            query_patch = query_patch * inverse_mask

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
                    compare_patch = compare_patch * inverse_mask

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


def compute_SSDs(which_descs, mask_percs):

    ssds_by_model = {}
    which_descs = [0, 1, 2, 3]
    mask_percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for which_desc in which_descs:

        print("which desc? " + str(which_desc))
        ssds_by_model[which_desc] = {}

        for mask_perc in mask_percs:
            print('--- Mask percentage ' + str(mask_perc) + ' ---')

            ssds_by_model[which_desc][mask_perc] = calculate_SSDs_for_desc_and_missing_level(mask_perc=mask_perc,
                                                                                       which_desc=which_desc)

    mask_percs_string = "_".join(str(mask_perc) for mask_perc in mask_percs)
    which_descs_string = "_".join(str(which_desc) for which_desc in which_descs)
    ssds_by_model_file_path = results_dir + '/ssds_missing_percentages_' + mask_percs_string + '__descrs_' + which_descs_string + '__' + str(total_nr_query_patches) + '_query_patches__' + results_version + '.pkl'

    with open(ssds_by_model_file_path, 'wb') as f:
        pickle.dump(ssds_by_model, f)

    return ssds_by_model

def plot_SSDs_for_4_descrs(ssds_by_model, mask_percs, ssds_by_model_plot_file_path):
    font = {'family': 'sans-serif',
            'weight': 'medium',
            'size': 23}

    matplotlib.rc('font', **font)

    # mask_percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # with open(ssds_by_model_file_path, 'rb') as f:
    #     ssds_by_model = pickle.load(f)

    fig = plt.figure(figsize=(25, 10))

    for mask_perc in mask_percs:
        data = []
        labels = []
        data.append(ssds_by_model[0][mask_perc])
        data.append(ssds_by_model[1][mask_perc])
        data.append(ssds_by_model[2][mask_perc])
        data.append(ssds_by_model[3][mask_perc])
        labels.append('A')
        labels.append('B')
        labels.append('C')
        labels.append('D')
        ax = fig.add_subplot(1, 6, mask_perc * 10 + 1)
        if mask_perc == 0:
            ax.set_ylabel('SSD', fontsize=23)
        ax.set_yscale('log')
        ax.set_title('Missing ' + str(int(mask_perc * 100)) + '%')
        ax.set_ylim([7500, 20000000])
        ax.boxplot(data, labels=labels)#, fontsize=18)

    fig.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.35, hspace=None)
    fig.savefig(ssds_by_model_plot_file_path + '.pdf', bbox_inches='tight')
    fig.savefig(ssds_by_model_plot_file_path + '.jpg', bbox_inches='tight')

    fig.show()

    plt.show(block=True)
    plt.interactive(False)


def compute_and_plot_SSDs():
    which_descs = [0, 1, 2, 3]
    mask_percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    ssds_by_model = compute_SSDs(which_descs, mask_percs)
    plot_SSDs_for_4_descrs(ssds_by_model, mask_percs)


def main():
    # compute_SSDs(0, 0)

    ssds_by_model_plot_file_path = '/home/niaki/PycharmProjects/patch-desc-ae/results/ssds_missing_percentages_0_0.1_0.2_0.3_0.4_0.5__descrs_0_1_2_3__256_query_patches_3'

    mask_percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ssds_by_model_file_path = '/home/niaki/PycharmProjects/patch-desc-ae/results/ssds_missing_percentages_0_0.1_0.2_0.3_0.4_0.5__descrs_0_1_2_3__256_query_patches_3.pkl'
    with open(ssds_by_model_file_path, 'rb') as f:
        ssds_by_model = pickle.load(f)
    plot_SSDs_for_4_descrs(ssds_by_model, mask_percs, ssds_by_model_plot_file_path)


if __name__ == "__main__":
    main()
