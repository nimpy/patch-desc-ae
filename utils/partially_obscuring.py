import numpy as np


def mask_random_rectangle(patch_size=16, mask_percentage_per_axis_mu=0.2, mask_percentage_per_axis_sigma=0.1):
    """Generate a mask with a rectangle somewhere on it."""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    mask_percentage_per_x_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_x_axis = np.clip(mask_percentage_per_x_axis, 0, 1)
    mask_percentage_per_y_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_y_axis = np.clip(mask_percentage_per_y_axis, 0, 1)

    x_centre = np.random.randint(patch_size)
    y_centre = np.random.randint(patch_size)

    x_mask_start = x_centre - int(mask_percentage_per_x_axis * patch_size)
    x_mask_end = x_centre + int(mask_percentage_per_x_axis * patch_size)

    y_mask_start = y_centre - int(mask_percentage_per_y_axis * patch_size)
    y_mask_end = y_centre + int(mask_percentage_per_y_axis * patch_size)

    x_mask_start = np.clip(x_mask_start, 0, patch_size)
    x_mask_end = np.clip(x_mask_end, 0, patch_size)
    y_mask_start = np.clip(y_mask_start, 0, patch_size)
    y_mask_end = np.clip(y_mask_end, 0, patch_size)

    for y in range(y_mask_start, y_mask_end):
        for x in range(x_mask_start, x_mask_end):
            mask[x, y] = 1
    return mask


def mask_random_corner_rectangle(patch_size=16, mask_percentage_per_axis_mu=0.25, mask_percentage_per_axis_sigma=0.3):
    """Generate a mask with a rectangle on one of the corners."""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    mask_percentage_per_x_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_x_axis = np.clip(mask_percentage_per_x_axis, 0, 1)

    mask_percentage_per_y_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_y_axis = np.clip(mask_percentage_per_y_axis, 0, 1)

    corner = np.random.randint(4)

    if corner % 2:
        x_centre = 0
        x_mask_start = x_centre
        x_mask_end = x_centre + int(mask_percentage_per_x_axis * patch_size)
    else:
        x_centre = patch_size
        x_mask_end = x_centre
        x_mask_start = x_centre - int(mask_percentage_per_x_axis * patch_size)

    if corner < 2:
        y_centre = 0
        y_mask_start = y_centre
        y_mask_end = y_centre + int(mask_percentage_per_y_axis * patch_size)
    else:
        y_centre = patch_size
        y_mask_end = y_centre
        y_mask_start = y_centre - int(mask_percentage_per_y_axis * patch_size)

    x_mask_start = np.clip(x_mask_start, 0, patch_size)
    x_mask_end = np.clip(x_mask_end, 0, patch_size)
    y_mask_start = np.clip(y_mask_start, 0, patch_size)
    y_mask_end = np.clip(y_mask_end, 0, patch_size)

    for y in range(y_mask_start, y_mask_end):
        for x in range(x_mask_start, x_mask_end):
            mask[x, y] = 1
    return mask


def mask_random_border_rectangle(patch_size=16, mask_percentage_per_axis_mu=0.25, mask_percentage_per_axis_sigma=0.3):
    """Generate a mask with a rectangle on one of the borders."""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    mask_percentage_per_x_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_x_axis = np.clip(mask_percentage_per_x_axis, 0, 1)

    mask_percentage_per_y_axis = np.random.normal(mask_percentage_per_axis_mu, mask_percentage_per_axis_sigma)
    mask_percentage_per_y_axis = np.clip(mask_percentage_per_y_axis, 0, 1)

    corner = np.random.randint(4)
    x_mask_start = 0
    x_mask_end = patch_size

    y_mask_start = 0
    y_mask_end = patch_size

    if corner == 0:
        x_centre = 0
        x_mask_start = x_centre
        x_mask_end = x_centre + int(mask_percentage_per_x_axis * patch_size)
    elif corner == 1:
        x_centre = patch_size
        x_mask_end = x_centre
        x_mask_start = x_centre - int(mask_percentage_per_x_axis * patch_size)
    elif corner == 2:
        y_centre = 0
        y_mask_start = y_centre
        y_mask_end = y_centre + int(mask_percentage_per_y_axis * patch_size)
    else:
        y_centre = patch_size
        y_mask_end = y_centre
        y_mask_start = y_centre - int(mask_percentage_per_y_axis * patch_size)

    x_mask_start = np.clip(x_mask_start, 0, patch_size)
    x_mask_end = np.clip(x_mask_end, 0, patch_size)
    y_mask_start = np.clip(y_mask_start, 0, patch_size)
    y_mask_end = np.clip(y_mask_end, 0, patch_size)

    for y in range(y_mask_start, y_mask_end):
        for x in range(x_mask_start, x_mask_end):
            mask[x, y] = 1
    return mask


def mcmove(config, N, beta):
    ''' This is to execute the Monte Carlo moves using
    Metropolis algorithm such that detailed
    balance condition is satisified'''
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[(a - 1) % N, b] + config[a, (b - 1) % N]
            cost = 2 * s * nb
            if cost < 0:
                s *= -1
            elif np.random.rand() < np.exp(-cost * beta):
                s *= -1
            config[a, b] = s
    return config


def mask_ising_model(patch_size=16, iteration=4):
    """Generate a mask using the Ising model."""
    N, temp = patch_size, .4
    config = 2 * np.random.randint(2, size=(N, N)) - 1

    msrmnt = 1001

    for i in range(msrmnt):
        mcmove(config, N, 1.0 / temp)
        if i == iteration:
            mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            for y in range(patch_size):
                for x in range(patch_size):
                    if config[x, y] == 1:
                        mask[x, y] = 0
                    else:
                        mask[x, y] = 1

            if np.count_nonzero(mask) > ((patch_size * patch_size) // 2):
                mask = 1 - mask

    return mask