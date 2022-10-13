# -*- coding: utf-8 -*-

#######################################################################################################################
# PAPER                                                                                                               #
#                                                                                                                     #
# You can get a *free* copy of the pre-print version from arXiv or Zenodo. Alternatively, you can get the published   #
# version from the publisher’s website (behind a paywall). Please check the README file for the links.                #
#                                                                                                                     #
# CITATION REQUEST                                                                                                    #
#                                                                                                                     #
# If you have found our paper and / or part of our code useful, please cite our work as follows:                      #
#                                                                                                                     #
# K. Malialis, D. Papatheodoulou, S. Filippou, C. G. Panayiotou, M. M. Polycarpou. Data Augmentation On-the-fly and   #
# Active Learning in Data Stream Classification. In IEEE Symposium Series on Computational Intelligence (SSCI), 2022. #
#                                                                                                                     #
# REQUIREMENTS                                                                                                        #
#                                                                                                                     #
# Python 3.7. Also, please check the “requirements.txt” file for the necessary libraries and packages.                #
#                                                                                                                     #
# INSTRUCTIONS                                                                                                        #
#                                                                                                                     #
# The user must specify the parameters in this file (main.py) and described below. For example, if you run main.py    #
# as it is, it will generate the three text files found in the "exps" folder (which is automatically created).        #
# You can then use the function provided in “plot.py” to plot the results, and obtain Fig. 5b in the paper.           #
#                                                                                                                     #
#   * Memory Size: MNIST {10, 50}, TwoPatterns and UWaveGestureLibraryZ {10, 100}. E.g. memory_sizes = [100].         #
#   * Imbalance Ratio: MNIST {0.1, 0.001, 1}. The value 1 refers to balanced scenarios.                               #
#                      It only applies to MNIST. E.g. imbalance_ratios = [1].                                         #
#   * Budgets: The active learning budget in the range of 0.0 < B < 1.0. E.g. budgets = [0.1].                        #
#   * Repeats: Number of repeats to avg the results. E.g. repeats = 20.                                               #
#   * VGG Blocks: Number of VGG blocks in the range {1, 2, 3, 4, 5}. E.g. num_vgg_blocks = 3.                         #
#   * Augmentations: Number of augmentations *PER* sample. E.g. augmentations = [10].                                 #
#   * Data Source: Name of the dataset {'mnist', 'TwoPatterns', 'UWaveGestureLibraryZ'}. E.g. data_source = 'mnist'.  #
#   * Method: The method to be used {'rvus', 'actiq'}. E.g. method = 'actiq'.                                         #
#   * Learning Flag: The learning type {'online', 'active'}. E.g. flag_learning='active'.                             #
#                    Note that online learning /= active learning with 100% budget.                                   #
#   * Architecture: The architecture type for the current run {'vgg', 'nn'}. E.g. architecture = 'vgg'.               #
#   * Output Directory: The directory to be automatically created for storing results. E.g. out_dir = 'exps/'.        #
#######################################################################################################################

from real_active import main

if __name__ == '__main__':

    configs = []

    # Generic configurations
    memory_sizes = [100, 10]
    imbalance_ratios = [1]  # NOTE: applies only for MNIST dataset
    budgets = [0.1]
    repeats = 5  # 20 repetitions in the paper
    data_source = 'TwoPatterns'
    method = 'actiq'
    flag_learning = 'active'
    architecture = 'vgg'
    out_dir = 'exps/'

    # Model specific configurations
    num_vgg_blocks = 3  # in [1,5]; ignored if other architecture is selected

    for memory in memory_sizes:

        if memory == 10:
            augmentations = [10, 0]
        elif memory == 100:
            augmentations = [0]

        for imb_ratio in imbalance_ratios:
            for num_augs in augmentations:
                for budget in budgets:
                    current_config = {'repeats': repeats,
                                      'data_source': data_source,
                                      'imbalance_ratio': imb_ratio,
                                      'num_augmentations': num_augs,
                                      'memory': memory,
                                      'method': method,
                                      'flag_learning': flag_learning,
                                      'active_budget_total': budget,  # NOTE: applies only if flag_learning = active
                                      'architecture': architecture,
                                      'out_dir': out_dir,
                                      }

                    if current_config["architecture"] == "vgg":
                        current_config["num_vgg_blocks"] = num_vgg_blocks

                    configs.append(current_config)

    for conf in configs:
        main(conf)
