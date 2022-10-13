# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow .keras.datasets import mnist


# load data
(x, y), _ = mnist.load_data()
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # reshape

# rescale to from [0,255] to [0,1]
x = x.astype('float32')
x /= 255


# prepare datasets
def create_arr(x, y, size_minority, size_majority, random_state, class_majority=0):
    arrs = []
    lst_idx = []

    for i in range(10):
        arr_x = x[np.where(y == i)[0]]
        arr_size = size_majority if i == class_majority else size_minority
        idx = random_state.choice(arr_x.shape[0], size=arr_size, replace=False)
        arr_x = arr_x[idx, :]
        lst_idx.append(idx)

        arr_y = i * np.ones(arr_x.shape[0])
        arr_y = arr_y.astype(np.uint8)  # same type as y
        arr_y = np.reshape(arr_y, (len(arr_y), 1))

        z = np.hstack((arr_x, arr_y))
        arrs.append(z)

    return np.vstack(arrs), np.hstack(lst_idx)


def create_data(imbalance_ratio, memory_size, random_state, store=0):
    # init data
    mnist_init, idx_init = create_arr(x, y, memory_size, memory_size, random_state)
    df_init = pd.DataFrame(mnist_init)
    name_init = 'mnist_init_mem' + str(memory_size) + '.csv'

    # arriving data
    new_x = np.delete(x, idx_init, axis=0)
    new_y = np.delete(y, idx_init, axis=0)

    num_samples_majority = 5000
    num_samples_minority = int(imbalance_ratio * num_samples_majority)
    print("Generating {} samples of the majority class & {} samples of minority class with imbalance ratio {}".format(num_samples_majority,
                                                                                                                      num_samples_minority,
                                                                                                                      imbalance_ratio))
    mnist_arr, _ = create_arr(new_x, new_y, num_samples_minority, num_samples_majority, random_state)
    dataset_balance_ratio_title = ('imbalance' + str(imbalance_ratio) if (float(imbalance_ratio) != 1.0) else 'balanced')


    df_arr = pd.DataFrame(mnist_arr)
    df_arr = df_arr.sample(frac=1).reset_index(drop=True)
    name_arr = 'mnist_arr_mem' + str(memory_size) + '_' + dataset_balance_ratio_title + '.csv'

    # store
    if store:
        df_init.to_csv(name_init, index=False, header=False)
        df_arr.to_csv(name_arr, index=False, header=False)


# generate data
random_state = np.random.RandomState(0)
memory_size = 10
imbalance_ratio = 1
create_data(imbalance_ratio, memory_size, random_state, store=1)
