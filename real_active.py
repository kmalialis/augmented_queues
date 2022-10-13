# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import gray2rgb
from tqdm import tqdm
from real_run_active import run
from class_nn_standard import StandardNN
from class_vgg import VGG

# hide warnings (before importing Keras)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

#######
# I/O #
#######


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()


# Write array to a row in the given file
def write_to_file(filename, arr):
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')


############
# Datasets #
############


def load_dataset(d_env, dataset_name):
    d_env['data_init'] = pd.read_csv(f"data/{dataset_name}/{dataset_name}_init_mem{d_env['memory']}.csv", header=None)
    if d_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
        d_env['data'] = pd.read_csv(f"data/{dataset_name}/{dataset_name}_arr_mem{d_env['memory']}.csv",
                                    header=None)
    else:
        if d_env['imbalance_ratio'] == 1:
            d_env['data'] = pd.read_csv(f"data/{dataset_name}/{dataset_name}_arr_mem{d_env['memory']}_balanced.csv", header=None)
        else:
            d_env['data'] = pd.read_csv(f"data/{dataset_name}/{dataset_name}_arr_mem{d_env['memory']}_imbalance{d_env['imbalance_ratio']}.csv", header=None)

    return d_env


def add_dataset_nn(d_env, d_nn, show_sample_image=False):
    # Load data
    d_env = load_dataset(d_env=d_env, dataset_name=d_env['data_source'])

    # derived
    d_env['num_classes'] = len(d_env['data_init'].iloc[:, -1].unique())
    d_env['memory_size'] = int(d_env['data_init'].shape[0] / d_env['num_classes'])
    d_env['num_features'] = d_env['data_init'].shape[1] - 1

    if show_sample_image:
        plt.imshow(d_env[0], interpolation='nearest')
        plt.show()

    # Set hyperparameters for each model
    if d_env['data_source'] == 'mnist':
        # params_nn
        d_nn['learning_rate'] = 0.0001
        d_nn['layer_dims'] = [d_env['num_features'], 1024, 1024]  # [n_x, n_h1, .., n_hL] i.e. it does not contain n_y
        d_nn['minibatch_size'] = 128
        d_nn['activation'] = 'leaky_relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0

    elif d_env['data_source'] == 'UWaveGestureLibraryZ':
        # params_nn
        d_nn['learning_rate'] = 0.001
        d_nn['layer_dims'] = [d_env['num_features'], 512, 512]  # [n_x, n_h1, .., n_hL] i.e. it does not contain n_y
        d_nn['minibatch_size'] = 128
        d_nn['activation'] = 'leaky_relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0

    elif d_env['data_source'] == 'TwoPatterns':
        # params_nn
        d_nn['learning_rate'] = 0.0001
        d_nn['layer_dims'] = [d_env['num_features'], 512, 512]  # [n_x, n_h1, .., n_hL] i.e. it does not contain n_y
        d_nn['minibatch_size'] = 128
        d_nn['activation'] = 'leaky_relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0


def add_dataset_vgg(d_env, d_nn, show_sample_image=False):
    # Load data
    d_env = load_dataset(d_env=d_env, dataset_name=d_env['data_source'])

    X_init, y_init = np.asarray(d_env['data_init'].iloc[:, :-1]), np.asarray(d_env['data_init'].iloc[:, -1])
    X, y = np.asarray(d_env['data'].iloc[:, :-1]), np.asarray(d_env['data'].iloc[:, -1])

    # Resize data and derive useful info for later use
    if d_env['data_source'] in ['TwoPatterns', 'UWaveGestureLibraryZ']:
        y_init = y_init.reshape(-1, 1)
        y = y.reshape(-1, 1)

        data_init = pd.DataFrame(np.hstack((X_init, y_init)))
        data = pd.DataFrame(np.hstack((X, y)))

        # Update datasets to resized versions
        d_env['data_init'] = data_init
        d_env['data'] = data

        # derived
        d_env['num_classes'] = len(d_env['data_init'].iloc[:, -1].unique())
        d_env['memory_size'] = int(d_env['data_init'].shape[0] / d_env['num_classes'])
        d_env['num_features'] = d_env['data_init'].shape[1] - 1

        d_nn['input_shape'] = (1, d_env['num_features'], 1)

    else:
        if d_env['data_source'] == 'mnist':
            RESIZE_DIM = 32  # Resize images to 32x32 for VGG compatibility (MaxPooling on 5th block)
            current_dim = int(np.sqrt(X_init.shape[-1]))  # Get the squared root of the number of columns to define the current dimensions of the image e.g. (784 -> 28x28)

            X_init, y_init = X_init.reshape(-1, current_dim, current_dim), y_init.reshape(-1, 1)
            X, y = X.reshape(-1, current_dim, current_dim), y.reshape(-1, 1)

            print(f"Resizing images from {current_dim}x{current_dim} to {RESIZE_DIM}x{RESIZE_DIM}")

            X_init = np.array([resize(img, (RESIZE_DIM, RESIZE_DIM)) for img in tqdm(X_init)])  # Resize images to 32x32
            X = np.array([resize(img, (RESIZE_DIM, RESIZE_DIM)) for img in tqdm(X)])            # Resize images to 32x32

            X_init = gray2rgb(X_init)   # Convert to RGB format (32x32x3)
            X = gray2rgb(X)             # Convert to RGB format (32x32x3)

            if show_sample_image:
                plt.imshow(X[0], interpolation='nearest')
                plt.show()

            channels = 3
            X_init = X_init.reshape(-1, (RESIZE_DIM ** 2) * channels)  # 32^2 = 1024 * 3 = 3072 columns
            X = X.reshape(-1, (RESIZE_DIM ** 2) * channels)

            data_init = pd.DataFrame(np.hstack((X_init, y_init)))
            data = pd.DataFrame(np.hstack((X, y)))

            # Update datasets to resized versions
            d_env['data_init'] = data_init
            d_env['data'] = data

            # derived
            d_env['num_classes'] = len(d_env['data_init'].iloc[:, -1].unique())
            d_env['memory_size'] = int(d_env['data_init'].shape[0] / d_env['num_classes'])
            d_env['num_features'] = d_env['data_init'].shape[1] - 1

        d_nn['input_shape'] = (32, 32, 3)

    # Set hyperparameters for each model
    if d_env['data_source'] == 'mnist':
        # params for mnist
        d_nn['num_blocks'] = d_env['num_vgg_blocks']
        d_nn['init_filters'] = 64

        d_nn['learning_rate'] = 0.001
        d_nn['batch_size'] = 128
        d_nn['activation'] = 'relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0

    elif d_env['data_source'] == 'TwoPatterns':
        # params for mnist
        d_nn['num_blocks'] = d_env['num_vgg_blocks']
        d_nn['init_filters'] = 64

        d_nn['learning_rate'] = 0.0001
        d_nn['batch_size'] = 128
        d_nn['activation'] = 'relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0
    elif d_env['data_source'] == 'UWaveGestureLibraryZ':
        # params for mnist
        d_nn['num_blocks'] = d_env['num_vgg_blocks']
        d_nn['init_filters'] = 64

        d_nn['learning_rate'] = 0.001
        d_nn['batch_size'] = 128
        d_nn['activation'] = 'relu'
        d_nn['l2_reg'] = 0.0
        d_nn['dropout'] = 0.0


######
# NN #
######
def create_nn_fc(params_env, params_nn, layer_dims, seed):
    return StandardNN(
        layer_dims=layer_dims + [params_env['num_classes']],
        learning_rate=params_nn['learning_rate'],
        num_epochs=params_nn['num_epochs'],
        minibatch_size=params_nn['minibatch_size'],
        l2_reg=params_nn['l2_reg'],
        dropout=params_nn['dropout'],
        seed=seed)


######
# VGG #
######
def create_nn_vgg(params_env, params_nn, seed):
    return VGG(
        num_classes=params_env['num_classes'],
        input_shape=params_nn['input_shape'],
        num_epochs=params_nn['num_epochs'],
        batch_size=params_nn['batch_size'],
        learning_rate=params_nn['learning_rate'],
        num_blocks=params_nn['num_blocks'],
        init_filters=params_nn['init_filters'],
        seed=seed,
    )


# create model
def create_nn_single(params_env, params_nn):
    nn = None

    if params_env['method'] in ['rvus', 'actiq']:
        if params_env['architecture'] == 'nn':
            nn = create_nn_fc(params_env, params_nn, layer_dims=params_nn['layer_dims'], seed=params_env['seed'])

        elif params_env['architecture'] == 'vgg':
            nn = create_nn_vgg(params_env, params_nn, seed=params_env['seed'])

    return nn

#################
# safety checks #
#################


def run_safety_checks(params_env):
    if params_env['architecture'] not in ['nn', 'vgg']:
        raise Exception(f'Incorrect architecture, {params_env["architecture"]} is not implemented.')

    if params_env['flag_learning'] not in ['online', 'active']:
        raise Exception('Incorrect learning paradigm entered.')

    if params_env['method'] not in ['rvus', 'actiq']:
        raise Exception('Incorrect learning method entered.')

    if params_env['data_source'] not in ['mnist', 'TwoPatterns', 'UWaveGestureLibraryZ']:
        raise Exception('Incorrect dataset entered.')

    if params_env['method'] == 'actiq' and params_env['memory_size'] < 1:
        raise Exception('Neural network requires memory size >= 1')

    if params_env['active_budget_total'] < 0.0 or params_env['active_budget_total'] > 1.0:
        raise Exception('Budget must be in [0,1].')

    if params_env['num_augmentations'] < 0:
        raise Exception('Number of augmentations must be greater or equal to 0.')


###########################################################################################
#                                         Main                                            #
###########################################################################################


def main(params_env):

    ######################
    # Settings: required #
    ######################

    # nn parameters
    params_nn = {'num_epochs': 1}  # NOTE: fixed

    if params_env['architecture'] == 'nn':
        add_dataset_nn(params_env, params_nn)
    elif params_env['architecture'] == 'vgg':
        add_dataset_vgg(params_env, params_nn)

    ###################
    # Settings: fixed #
    ###################
    # NOTE: Keep these parameters fixed to replicate the paper's results

    # fixed - suggested by their authors
    params_env['active_threshold_update'] = 0.01
    params_env['active_budget_window'] = 300
    params_env['active_budget_lambda'] = 1.0 - (1.0 / params_env['active_budget_window'])
    params_env['active_delta'] = 1.0  # N(1, delta) - no randomisation if set to 0

    # fixed
    params_env['seed'] = 0
    params_env['preq_fading_factor'] = 0.99
    params_env['flag_store'] = 1

    # derived
    params_env['random_state'] = np.random.RandomState(params_env['seed'])
    params_env['time_steps'] = int(params_env['data'].shape[0])
    print("Time steps:", params_env['time_steps'])

    # safety checks for the inserted settings
    run_safety_checks(params_env)

    ################
    # Output files #
    ################

    # file directory and names
    out_method = params_env['method']

    out_dir = params_env['out_dir']
    out_dir_name = '{}_{}{}_{}{}x{}'.format(params_env['architecture'],
                                            params_env['data_source'], params_env['imbalance_ratio'],
                                            out_method, params_env['memory_size'], params_env['num_augmentations'])
    out_file_name = '{}_{}{}_{}{}x{}_{}'.format(params_env['architecture'],
                                                params_env['data_source'], params_env['imbalance_ratio'],
                                                out_method, params_env['memory_size'], params_env['num_augmentations'],
                                                params_env['active_budget_total'])
    out_path = out_dir + out_dir_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # files to store g-mean
    filename_acc = os.path.join(os.getcwd(), out_path, out_file_name + '_preq_acc.txt')
    filename_gmean = os.path.join(os.getcwd(), out_path, out_file_name + '_preq_gmean.txt')
    filename_counter = os.path.join(os.getcwd(), out_path, out_file_name + '_counter.txt')

    if params_env['flag_store']:
        create_file(filename_acc)
        create_file(filename_gmean)
        create_file(filename_counter)

    #########
    # Start #
    #########

    for r in range(params_env['repeats']):
        print('Repetition: ', r)

        # create nn
        params_env['nn'] = create_nn_single(params_env, params_nn)

        # start
        preq_general_accs, _, preq_gmeans, num_labels = run(params_env)

        # store
        if params_env['flag_store']:
            write_to_file(filename_acc, preq_general_accs)
            write_to_file(filename_gmean, preq_gmeans)
            write_to_file(filename_counter, num_labels)
