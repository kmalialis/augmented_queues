# Augmented Queues
There is an emerging need for predictive models to be trained on-the-fly, since in numerous machine learning applications data are arriving in an online fashion. A critical challenge encountered is that of limited availability of ground truth information (e.g., labels in classification tasks) as new data are observed one-by-one online, while another significant challenge is that of class imbalance. This work introduces the novel Augmented Queues method, which addresses the dual-problem by combining in a synergistic manner online active learning, data augmentation, and a multi-queue memory to maintain separate and balanced queues for each class. We perform an extensive experimental study using image and time-series augmentations, in which we examine the roles of the active learning budget, memory size, imbalance level, and neural network type. We demonstrate two major advantages of Augmented Queues. First, it does not reserve additional memory space as the generation of synthetic data occurs only at training times. Second, learning models have access to more labelled data without the need to increase the active learning budget and / or the original memory size. Learning on-the-fly poses major challenges which, typically, hinder the deployment of learning models. Augmented Queues significantly improves the performance in terms of learning quality and speed.

# Paper
You can get a free copy of the pre-print version from arXiv (link TBA) or Zenodo (link TBA).

Alternatively, you can get the published version from the publisher’s website (link TBA).

# Instructions
Please check the “main.py” file.

# Requirements
Python 3.7. Please also check the “requirements.txt” file for the necessary libraries and packages.

# Citation request
If you have found our paper and / or part of our code useful, please cite our work as follows:

K. Malialis, D. Papatheodoulou, S. Filippou, C. G. Panayiotou, M. M. Polycarpou. Data Augmentation On-the-fly and Active Learning in Data Stream Classification. IEEE Symposium Series on Computational Intelligence (SSCI), 2022.
