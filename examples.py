import imageio
from ae_descriptor import init_descr_32, compute_descriptor, init_IR, compute_IR, compute_descriptor_from_IR


def print_descr_3dims(descr):
    for k in range(descr.shape[2]):
        for j in range(descr.shape[1]):
            for i in range(descr.shape[0]):
                print(round(descr[i, j, k], 2), end=' ')
            print()
        print()
        print()
    print()


def example_use_descrs():
    print("Example usage of descriptors...")

    patch_size = 16
    image = imageio.imread('/home/niaki/Downloads/Lenna.png')
    image = image / 255.
    x_coord = 26
    y_coord = 473
    patch = image[x_coord : x_coord + patch_size, y_coord : y_coord + patch_size, :]


    encoder = init_descr_32(patch_size=patch_size)
    patch_descr1 = compute_descriptor(patch, encoder)

    print("patch descr shape", patch_descr1.shape)
    print("patch descr:")
    print_descr_3dims(patch_descr1)


    encoder_IR, encoder_mp = init_IR(image.shape[0], image.shape[1], patch_size)
    image_IR = compute_IR(image, encoder_IR)
    print("patch IR")
    print_descr_3dims(image_IR[x_coord: x_coord + patch_size, y_coord: y_coord + patch_size, :])
    patch_descr2 = compute_descriptor_from_IR(image_IR, x_coord, y_coord, patch_size, encoder_mp)

    print("patch descr shape", patch_descr2.shape)
    print("patch descr from IR:")
    print_descr_3dims(patch_descr2)


    print("diff")
    print(patch_descr1 - patch_descr2)
    print("print descr")
    print_descr_3dims(patch_descr1 - patch_descr2)


def main():
    example_use_descrs()


if __name__ == "__main__":
    main()