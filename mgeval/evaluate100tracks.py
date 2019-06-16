# -*- coding: utf-8 -*-
"""
# Metrics available from core library
    total_used_pitch
    bar_used_pitch
    total_used_note
    bar_used_note
    total_pitch_class_histogram
    bar_pitch_class_histogram
    pitch_class_transition_matrix
    pitch_range
    avg_pitch_shift
    avg_IOI
    note_length_hist
    note_length_transition_matrix
"""
import midi
import sys
import glob
import numpy as np
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

"""## Absolute measurement: statistic analysis

Assign dataset path
"""

set1 = glob.glob('./Dataset1/*.mid')
set2 = glob.glob('./Dataset2/*.mid')
"""construct empty dictionary to fill in measurement across samples"""

num_samples = 10
set1_eval = {'total_used_pitch':np.zeros((num_samples,1)),                      #working
                # 'bar_used_pitch':np.zeros((num_samples,1)),
#                'total_used_note':np.zeros((num_samples,1)),                    #working
                # 'bar_used_note':np.zeros((num_samples,1)),
#                'total_pitch_class_histogram':np.zeros((num_samples,12)),#working
                # 'bar_pitch_class_histogram':np.zeros((num_samples,12)),
#                'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),#working
                # 'pitch_range':np.zeros((num_samples,1)),
#                'avg_pitch_shift':np.zeros((num_samples,1)),#working
#                'avg_IOI':np.zeros((num_samples,1)),#working
#                'note_length_hist':np.zeros((num_samples,12)),#working
#                'note_length_transition_matrix':np.zeros((num_samples,12,12))#working
            }


set2_eval = {'total_used_pitch':np.zeros((num_samples,1)),#working
                # 'bar_used_pitch':np.zeros((num_samples,1)),
#                'total_used_note':np.zeros((num_samples,1)),#working
                # 'bar_used_note':np.zeros((num_samples,1)),
#                'total_pitch_class_histogram':np.zeros((num_samples,12)),#working
                # 'bar_pitch_class_histogram':np.zeros((num_samples,12)),
#                'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),#working
                # 'pitch_range':np.zeros((num_samples,1)),
#                'avg_pitch_shift':np.zeros((num_samples,1)),#working
#                'avg_IOI':np.zeros((num_samples,1)),#working
#                'note_length_hist':np.zeros((num_samples,12)),#working
#                'note_length_transition_matrix':np.zeros((num_samples,12,12))#working
            }

metrics_list = list(set1_eval.keys())
print("set1 size: " + str(len(set1)))
print("set2 size: " + str(len(set2)))

printonce = False;

for i in range(0, num_samples):
    feature1 = core.extract_feature(set1[i])
    feature2 = core.extract_feature(set2[i])

    # print("Track1: " + set1[i] + " Track2: " + set2[i]

    for metric_name in metrics_list:
        if not printonce:
            print("evaluating " +  metric_name)
        set1_eval[metric_name][i] = getattr(core.metrics(), metric_name)(feature1)
        set2_eval[metric_name][i] = getattr(core.metrics(), metric_name)(feature2)

    printonce = True




"""statistic analysis: absolute measurement"""

for i in range(0, len(metrics_list)):

    print('\n------------------------------------------------')
    print((metrics_list[i] + ':'))
    print('------------------------------------------------')
    print(' train_set')
    print(('  mean: ', np.mean(set1_eval[metrics_list[i]], axis=0)))
    # print(('  std: ', np.std(set1_eval[metrics_list[i]], axis=0)))

    # print('------------------------')
    print(' generated_test')
    print(('  mean: ', np.mean(set2_eval[metrics_list[i]], axis=0)))
    # print(('  std: ', np.std(set2_eval[metrics_list[i]], axis=0)))
    from sklearn.preprocessing import normalize
    if(metrics_list[i] is 'note_length_transition_matrix'):
        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        # ax.set_aspect('equal')
        plt.imshow( np.mean(set1_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.bone)


        ax = fig.add_subplot(2,2,2)
        # ax.set_aspect('equal')
        normalized = normalize(np.mean(set1_eval[metrics_list[i]], axis=0),axis=1, norm='l1')
        plt.imshow(normalized, interpolation='nearest', cmap=plt.cm.bone)

        a2 = fig.add_subplot(2,2,3)
        # a2.set_aspect('equal')
        plt.imshow( np.mean(set2_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.bone)

        ax = fig.add_subplot(2,2,4)
        # ax.set_aspect('equal')
        normalized = normalize(np.mean(set2_eval[metrics_list[i]], axis=0),axis=1, norm='l1')
        plt.imshow(normalized , interpolation='nearest', cmap=plt.cm.bone)

        plt.colorbar()
        plt.show()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.set_aspect('equal')
        # plt.imshow( np.std(set1_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.ocean)
        # plt.colorbar()
        # plt.show()

'''

"""## Relative measurement: generalizes the result among features with various dimensions

the features are sum- marized to
- the intra-set distances
- the difference of intra-set and inter-set distances.

exhaustive cross-validation for intra-set distances measurement
"""

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
        set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

"""exhaustive cross-validation for inter-set distances measurement"""

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])

"""visualization of intra-set and inter-set distances"""

plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)
for i in range(0,len(metrics_list)):
    sns.kdeplot(plot_set1_intra[i], label='intra_set1')
    sns.kdeplot(plot_sets_inter[i], label='inter')
    sns.kdeplot(plot_set2_intra[i], label='intra_set2')

    plt.title(metrics_list[i])
    plt.xlabel('Euclidean distance')
    plt.show()

"""the difference of intra-set and inter-set distances."""

for i in range(0, len(metrics_list)):
    print((metrics_list[i] + ':'))
    print('------------------------')
    print(' demo_set1')
    print(('  Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])))
    print(('  Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])))

    print(' demo_set2')
    print(('  Kullback–Leibler divergence:',utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])))
    print(('  Overlap area:', utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])))
'''
