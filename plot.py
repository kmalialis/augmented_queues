# -*- coding: utf-8 -*-

import os.path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

########
# Plot #
########

def create_plots(list_filenames, list_legend_names=[], loc='lower right'):
    size = 42
    size_legend = 25
    fig = plt.figure(figsize=(12, 12))

    for i in range(len(list_filenames)):
        arr = np.loadtxt(list_filenames[i], delimiter=', ')         # load data

        print(arr.shape)

        means = np.mean(arr, axis=0)                                # y-axis values
        x_axis = np.arange(means.shape[0])                          # x-axis values
        se = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])    # standard error (ddof=1 for sample)

        # markers_on = range(0, arr.shape[1], 500)
        # plt.plot(x_axis, means, marker=markers[i], markevery=markers_on, label=str(list_legend_names[i]))

        plt.plot(x_axis, means, label=str(list_legend_names[i]), linewidth=3.0)
        plt.fill_between(x_axis, means - se, means + se, alpha=0.2)

    # x-axis
    plt.xlim(0,arr.shape[1])

    plt.xlabel('Time Step', fontsize=size, weight='bold')
    plt.xticks(fontsize=size)
    plt.xticks(np.arange(0.0, arr.shape[1] + 100, 5000), fontsize=size)

    # y-axis
    plt.ylabel('G-mean', fontsize=size, weight='bold')
    plt.yticks(np.arange(0.0, 1.000001, 0.2), fontsize=size)
    plt.ylim(0.0, 1.0)

    # legend
    if 1:
        leg = plt.legend(ncol=1, loc=loc, fontsize=size_legend)
        leg.get_frame().set_alpha(0.9)

    # grid
    plt.grid(linestyle='dotted')

    # plot
    plt.show()

    # save
    # fig.savefig(out_dir + 'circles10.pdf', bbox_inches='tight')



########
# test #
########

out_dir = 'exps/'
filenames = [
    out_dir + 'vgg_TwoPatterns1_actiq10x0_0.1_preq_gmean.txt',
    out_dir + 'vgg_TwoPatterns1_actiq10x10_0.1_preq_gmean.txt',
    out_dir + 'vgg_TwoPatterns1_actiq100x0_0.1_preq_gmean.txt',
]

legend = ['ActiQ - VGG (3 Blocks, M=10)', 'ActiQ - VGG (3 Blocks, M=10, N=10)', 'ActiQ - VGG (3 Blocks, M=100)']
create_plots(filenames, list_legend_names = legend, loc='lower right')
