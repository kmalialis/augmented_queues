# -*- coding: utf-8 -*-
import sys

import numpy as np
from collections import deque

from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from aux_augment import get_augmentations, get_TimeSeries_augmentations

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

########
# Data #
########


# Get samples in dict of queues
def get_sample(params_env, time_step, flag_init):
    # get pairs
    d_xs = {}
    if flag_init:
        arr = params_env['data_init'].values
        for i in range(params_env['num_classes']):
            idx_cls = np.where(arr[:, -1] == i)[0]
            arr_cls = arr[idx_cls]
            q_cls = deque(maxlen=params_env['memory_size'])
            for j in range(len(idx_cls)):
                q_cls.append(arr_cls[j, :])
            d_xs[i] = q_cls
    else:
        arr_cls = params_env['data'].values[time_step, :]
        q_cls = deque(maxlen=1)
        q_cls.append(arr_cls)
        cls = int(arr_cls[-1])
        d_xs[cls] = q_cls

    return d_xs


# reshape img for VGG
def x_reshape_vgg(x):
    X_num_columns = x.shape[0]
    if X_num_columns == 3072:
        channels = 3
        X_num_columns = X_num_columns / channels        # Get number of columns for a single channel (3072/3 = 1024)
        X_dim = int(np.sqrt(X_num_columns))             # Get height/width (32 pixels)
        x = x.reshape((-1, X_dim, X_dim, channels))     # Convert to 32x32x3
    return x


# reshape TS for VGG
def x_reshape_vggTS(x):
    nb_dims = 1
    nb_timeSteps = int(x.shape[0] / nb_dims)
    input_shape = (nb_timeSteps, nb_dims)
    x = x.reshape((-1, 1, input_shape[0], input_shape[1]))

    return x


##########################
# Prequential evaluation #
##########################


def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric

##################
# Model training #
##################


# data prep for ActiQ training
def fc_prep_training(d, params_env, show_sample_augmentation=False):
    # unfold dict
    xy = [a for _, q in d.items() for a in q]
    xy = np.vstack(xy)

    # features
    x = xy[:, :-1]

    # target
    y = xy[:, -1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')
    y = np.reshape(y, (y.shape[0], 1))

    # if TS data then reshape in order to apply augmentations
    if params_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
        nb_dims = 1
        nb_timesteps = int(x.shape[1] / nb_dims)
        input_shape = (nb_timesteps, nb_dims)
        x = x.reshape((-1, input_shape[0], input_shape[1]))

    # Apply TS augmentations
    if params_env['num_augmentations'] > 0 and params_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
        tempX = x
        tempy = y
        tempyEnc = y_encoded
        for i in range(len(x)):
            aug_x, aug_y, aug_y_enc = get_TimeSeries_augmentations(x[i:i+1], y[i], y_encoded[i],
                                                                   num_aug_per_config=params_env['num_augmentations'])
            input_shape = (aug_x[0].shape[1], 1)
            for tAugm in range(len(aug_x)):
                aug_x[tAugm] = aug_x[tAugm].reshape((input_shape[0], input_shape[1]))
            # Stack original and augmented data
            tempX = np.vstack([tempX, aug_x])
            tempy = np.vstack([tempy, aug_y])
            tempyEnc = np.vstack([tempyEnc, aug_y_enc])

        x = tempX
        y = tempy
        y_encoded = tempyEnc

    # else apply img augmentations
    elif params_env['num_augmentations'] > 0:
        aug_x = x * 255
        aug_x, aug_y, aug_y_encoded = get_augmentations(aug_x, y, num_aug_per_config=params_env['num_augmentations'],
                                                        combinatorial_augmentation=True)
        aug_x = aug_x.astype('float32')
        aug_x /= 255
        aug_x = aug_x.reshape(-1, 784)

        if show_sample_augmentation:
            # Show the first augmented image from the list
            plt.imshow(aug_x[0], interpolation='nearest')
            plt.show()

        # Stack original and augmented data
        x = np.vstack([x, aug_x])
        y = np.vstack([y, aug_y])
        y_encoded = np.vstack([y_encoded, aug_y_encoded])

    return x, y, y_encoded


# data prep for VGG-ActiQ training
def vgg_prep_training(d, params_env, show_sample_augmentation=True):
    # unfold dict
    xy = [a for _, q in d.items() for a in q]
    xy = np.vstack(xy)

    # features
    x = xy[:, :-1]

    # reshape data for VGG
    if params_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
        nb_dims = 1
        nb_timeSteps = int(x.shape[1] / nb_dims)
        input_shape = (nb_timeSteps, nb_dims)
        x = x.reshape(-1, 1, input_shape[0], input_shape[1])

    else:
        x = [x_reshape_vgg(x=elem) for elem in x]
        x = np.vstack(x)


    # target
    y = xy[:, -1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')
    y = np.reshape(y, (y.shape[0], 1))

    if params_env['num_augmentations'] > 0 and params_env['data_source'] in ['ElectricDevices', 'ChlorineConcentration',
                                                                             'StarLightCurves', 'TwoPatterns',
                                                                             'UWaveGestureLibraryZ']:
        tempX = x
        tempy = y
        tempyenc = y_encoded
        for sample, y_sample, y_enc_sample in zip(x, y, y_encoded):
            aug_x, aug_y, aug_y_enc = get_TimeSeries_augmentations(sample, y_sample, y_enc_sample,
                                                                   num_aug_per_config=params_env['num_augmentations'])
            tempX = np.vstack([tempX, aug_x])
            tempy = np.vstack([tempy, aug_y])
            tempyenc = np.vstack([tempyenc, aug_y_enc])

        x = tempX
        y = tempy
        y_encoded = tempyenc

    elif params_env['num_augmentations'] > 0:
        aug_x = x * 255
        aug_x, aug_y, aug_y_encoded = get_augmentations(aug_x, y, num_aug_per_config=params_env['num_augmentations'], combinatorial_augmentation=True)
        aug_x = aug_x.astype('float32')
        aug_x /= 255

        if show_sample_augmentation:
            # Show the first augmented image from the list
            plt.imshow(aug_x[0], interpolation='nearest')
            plt.show()

        # Stack original and augmented data
        x = np.vstack([x, aug_x])
        y = np.vstack([y, aug_y])
        y_encoded = np.vstack([y_encoded, aug_y_encoded])

    # shuffle data (not really needed)
    # x, y, y_encoded = unison_shuffled_copies(params_env, x, y, y_encoded)
    # x = x.reshape(-1, x.shape[2], 1)  # added for 1d pooling
    return x, y, y_encoded


# data prep for Incremental training
def incr_fc_prep_training(xy, params_env, show_sample_augmentation=True):

    # features
    x = xy[:-1]

    # target
    y = xy[-1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')

    # reshape
    x = np.reshape(x, (1, x.shape[0]))
    y_encoded = np.reshape(y_encoded, (1, y_encoded.shape[0]))
    y = np.reshape(y, (1, 1))

    if params_env['num_augmentations'] > 0:
        aug_x = x * 255
        aug_x, aug_y, aug_y_encoded = get_augmentations(aug_x, y, num_aug_per_config=params_env['num_augmentations'],
                                                        combinatorial_augmentation=True)
        aug_x = aug_x.astype('float32')
        aug_x /= 255
        aug_x = aug_x.reshape(-1, 784)

        if show_sample_augmentation:
            # Show the first augmented image from the list
            tmp = aug_x[0].reshape(1, 784)
            plt.imshow(aug_x[0], interpolation='nearest')
            plt.show()

        # Stack original and augmented data
        x = np.vstack([x, aug_x])
        y = np.vstack([y, aug_y])
        y_encoded = np.vstack([y_encoded, aug_y_encoded])

    return x, y, y_encoded


# data prep for Incremental training
def incr_vgg_prep_training(xy, params_env):
    # features
    x = xy[:-1]

    # target
    y = xy[-1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')

    # reshape
    x = x_reshape_vgg(x=x)
    y_encoded = np.reshape(y_encoded, (1, y_encoded.shape[0]))
    y = np.reshape(y, (1, 1))

    return x, y, y_encoded


# train model (single classifier)
def prep_and_train(d, xy, params_env):
    # get x and y
    x = None
    y = None

    if params_env['method'] == 'rvus':  # y is y_encoded here
        if params_env['architecture'] == 'nn':
            x, _, y = incr_fc_prep_training(xy, params_env)
        elif params_env['architecture'] == 'vgg':
            x, _, y = incr_vgg_prep_training(xy, params_env)

    elif params_env['method'] == 'actiq':  # y is y_encoded here
        if params_env['architecture'] == 'nn':
            x, _, y = fc_prep_training(d, params_env)
        elif params_env['architecture'] == 'vgg':
            x, _, y = vgg_prep_training(d, params_env)

    # print('Final x shape before train:', x.shape)
    # train
    params_env['nn'].train(x, y)

###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(params_env):

    ######################
    # Init preq. metrics #
    ######################

    # general accuracy
    preq_general_accs = []
    preq_general_acc_n = 0.0
    preq_general_acc_s = 0.0

    # class accuracies
    keys = range(params_env['num_classes'])
    preq_class_accs = {k: [] for k in keys}
    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

    # gmean
    preq_gmeans = []

    ####################
    # Init AL strategy #
    ####################

    active_threshold = 1.0
    budget_current = 0.0
    budget_u = 0.0
    d_counter = {k: 0 for k in keys}  # counter for num of labels requested per class

    #############
    # Init data #
    #############

    d_xy = get_sample(params_env, -1, flag_init=True)

    #########
    # Start #
    #########

    d_mis = {key: 0 for key in range(params_env['num_classes'])}  # mis-classifications

    for t in tqdm(range(0, params_env['time_steps'])):

        if t % 1000 == 0:
            print('Time step: ', t)

        ###############
        # Get example #
        ###############

        # get example
        d_temp = get_sample(params_env, t, flag_init=False)
        xy = [list(i) for i in d_temp.values()][0][0]

        xy = np.reshape(xy, (1, len(xy)))

        x = xy[0, :-1]
        y = xy[0, -1]

        # reshape here once to avoid reshaping multiple times later on
        if params_env['architecture'] == 'nn':
            x = np.reshape(x, (1, x.shape[0]))
        elif params_env['architecture'] == 'vgg':
            if params_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
                x = x_reshape_vggTS(x)
            else:
                x = x_reshape_vgg(x)

        xy = np.reshape(xy, (xy.shape[1],))

        ###################
        # Predict example #
        ###################
        # Output:
        # y_pred_max: will be used by the AL strategy
        # pred_class: will be used to determine correctness (evaluation)

        pred_class = None
        y_pred_max = None

        if params_env['method'] in ['rvus', 'actiq']:
            _, y_pred_max, pred_class = params_env['nn'].predict(x)

        ###############
        # Correctness #
        ###############

        correct = 1 if y == pred_class else 0  # check if prediction was correct

        if not correct:  # update mis-classifications
            d_mis[int(y)] = d_mis[int(y)] + 1

        ########################
        # Update preq. metrics #
        ########################

        # update general accuracy
        preq_general_acc_s, preq_general_acc_n, preq_general_acc = \
            update_preq_metric(preq_general_acc_s, preq_general_acc_n, correct, params_env['preq_fading_factor'])
        preq_general_accs.append(preq_general_acc)

        # update class accuracies & gmean
        preq_class_acc_s[y], preq_class_acc_n[y], preq_class_acc[y] = update_preq_metric(
            preq_class_acc_s[y], preq_class_acc_n[y], correct, params_env['preq_fading_factor'])

        lst = []
        for k, v in preq_class_acc.items():
            preq_class_accs[k].append(v)
            lst.append(v)

        gmean = np.power(np.prod(lst), 1.0 / len(lst))
        preq_gmeans.append(gmean)

        ###################
        # Online learning #
        ###################
        # NOTE: This is different from setting the budget = 1.0 in active learning below

        if params_env['flag_learning'] == 'online':
            d_counter[y] += 1  # increase counter
            d_xy[y].append(xy)  # append new example
            prep_and_train(d_xy, xy, params_env)  # data prep and training

        ###################
        # Active learning #
        ###################

        elif params_env['flag_learning'] == 'active':
            labelling = 0

            if budget_current < params_env['active_budget_total']:
                rnd = params_env['random_state'].normal(1.0, params_env['active_delta'])
                threshold = active_threshold * rnd

                if y_pred_max <= threshold:
                    labelling = 1  # set flag
                    d_counter[y] += 1  # increase counter
                    d_xy[y].append(xy)  # append to queues

                    # data prep and training
                    prep_and_train(d_xy, xy, params_env)

                    # reduce AL threshold
                    active_threshold *= (1.0 - params_env['active_threshold_update'])
                else:
                    # increase AL threshold
                    active_threshold *= (1.0 + params_env['active_threshold_update'])

            # update budget
            budget_u = labelling + budget_u * params_env['active_budget_lambda']
            budget_current = budget_u / params_env['active_budget_window']

    # number of labels per class (this is to ensure order)
    num_labels = np.zeros(len(keys))
    for k in keys:
        num_labels[k] = d_counter[k]

    return preq_general_accs, preq_class_accs, preq_gmeans, num_labels

