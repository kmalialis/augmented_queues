import numpy as np
from scipy.interpolate import CubicSpline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    """
    https://github.com/yu4u/cutout-random-erasing
    :param p: The probability that random erasing is performed
    :param s_l: Minimum proportion of erased area against input image
    :param s_h: Maximum proportion of erased area against input image
    :param r_1: Minimum aspect ratio of erased area
    :param r_2: Maximum aspect ratio of erased area
    :param v_l: Minimum value for erased area
    :param v_h: Maximum value for erased area
    :param pixel_level: Pixel-level randomization for erased area
    :return: Augmented image
    """

    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


def apply_data_augmentations(x, y, configs, num_aug_per_config=10):

    # Required for grayscale images
    if len(x.shape) < 3:
        current_dim = int(np.sqrt(x.shape[-1]))
        x = x.reshape(current_dim, current_dim, -1)

    sample = np.expand_dims(x, axis=0)
    aug_imgs = []
    aug_y = []
    for config in configs:
        datagen = ImageDataGenerator(**config)
        it = datagen.flow(sample, batch_size=1)

        for i in range(num_aug_per_config):
            batch = it.next()
            image = batch[0].astype('uint8')
            aug_imgs.append(np.array(image))
            aug_y.append(y)

    return aug_imgs, aug_y


def get_augmentations(X, y, num_aug_per_config=10, combinatorial_augmentation=True):

    if combinatorial_augmentation:
        # Note: When combinatorial augmentation is enabled, this will result in 1 * num_aug_per_config augmentations
        aug_params = [
            {
                'brightness_range': (0.1, 2.0),
                'zoom_range': (0.5, 1.5),
                'rotation_range': 20,
                'width_shift_range': (-0.1, 0.1),
                'height_shift_range': (-0.1, 0.1),
                'preprocessing_function': get_random_eraser(pixel_level=True),
            }
        ]
    else:
        # Note: When combinatorial augmentation is disabled, this will result in len(aug_params) * num_aug_per_config augmentations
        # Each dictionary is an augmentation that will be applied to every image, dictionaries with multiple key-value pairs are "stacked augmentations"
        aug_params = [
            {'brightness_range': (0.1, 2.0)},
            {'width_shift_range': (-0.1, 0.1)},
            {'height_shift_range': (-0.1, 0.1)},
            {'zoom_range': (0.5, 1.5)},
            {'rotation_range': 20},
            {'width_shift_range': (-0.1, 0.1), 'height_shift_range': (-0.1, 0.1)},
            {'preprocessing_function': get_random_eraser(pixel_level=False)},
        ]

    aug_X = []
    aug_y = []
    print('Applying augmentations...')
    for i in range(len(X)):
        current_X_augs, curr_y = apply_data_augmentations(x=X[i],
                                                          y=y[i],
                                                          configs=aug_params,
                                                          num_aug_per_config=num_aug_per_config)
        aug_X.extend(current_X_augs)
        aug_y.extend(curr_y)

    aug_X = np.asarray(aug_X)
    aug_y = np.asarray(aug_y)
    aug_y_encoded = to_categorical(aug_y, num_classes=10, dtype='float32')

    return aug_X, aug_y, aug_y_encoded


def get_TimeSeries_augmentations(X, y_sample, y_enc_sample, num_aug_per_config=10):
    # time series transformation
    aug_X = []
    aug_Y = []
    aug_Y_enc = []
    # apply 2 types of time series transformation, equal number of transformations
    for _ in range(0, int(num_aug_per_config/2)):
        # time wrap transform
        aumData = time_warp(X)
        aug_X.append(aumData)
        # window slice transform
        aumData = window_slice(X)
        aug_X.append(aumData)
        # extend y
        aug_Y.append(y_sample)
        aug_Y.append(y_sample)
        aug_Y_enc.append(y_enc_sample)
        aug_Y_enc.append(y_enc_sample)

    return aug_X, aug_Y, aug_Y_enc
