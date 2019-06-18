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

song1 = str(sys.argv[1])
# set1 = glob.glob('./Song1/*.mid')

num_samples = 1
song1_eval = {'total_used_pitch':np.zeros((num_samples,1)),                      #working
               'bar_used_pitch':np.zeros((num_samples,1)),
               'total_used_note':np.zeros((num_samples,1)),                    #working
                'bar_used_note':np.zeros((num_samples,1)),
               'total_pitch_class_histogram':np.zeros((num_samples,12)),#working
                #'bar_pitch_class_histogram':np.zeros((num_samples,12)),
               'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),#working
               'pitch_range':np.zeros((num_samples,1)),#working
               'avg_pitch_shift':np.zeros((num_samples,1)),#working
               'avg_IOI':np.zeros((num_samples,1)),#working
               'note_length_hist':np.zeros((num_samples,12)),#working
               'note_length_transition_matrix':np.zeros((num_samples,12,12))#working
            }


metrics_list = list(song1_eval.keys())

printonce = False;


feature = core.extract_feature(song1)

for metric_name in metrics_list:
    # if not printonce:
    #     print("evaluating " +  metric_name)

    if (metric_name is 'pitch_class_transition_matrix') or (metric_name is 'note_length_transition_matrix'):
        song1_eval[metric_name] = getattr(core.metrics(), metric_name)(feature,normalize = 2)
    else:
        song1_eval[metric_name] = getattr(core.metrics(), metric_name)(feature)

printonce = True

for i in range(0, len(metrics_list)):
    print metrics_list[i]
    print song1_eval[metrics_list[i]]
    print "---------------------\n"


def absolute_niceplot (metrics):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('song1.pdf')

    pitches = ["C ","C#","D ","D#","E ","F ","F#","G ","G#","A ","Bb","B "]
    notelengths = ["1","1/2","1/4","1/8","1/16","3/4","3/8","3/16","3/32","1t","1/2t","1/4t"]



    if 'total_used_pitch' in list(metrics.keys()):
        print("Total used pitch")
        fig = plt.figure()
        fig.clf()
        txt = 'Total used pitch: ' + str(metrics['total_used_pitch'])
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'bar_used_pitch' in list(metrics.keys()):
        print("bar_used_pitch")
        metric = metrics['bar_used_pitch']
        fig = plt.figure()
        fig.clf()
        txt = 'Average Bar used pitch: ' + str(round(np.mean(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'total_used_note' in list(metrics.keys()):
        print("total_used_note")
        metric = metrics['total_used_note']
        fig = plt.figure()
        fig.clf()
        txt = 'Total used note: ' + str(round(np.mean(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'bar_used_note' in list(metrics.keys()):
        print("bar_used_note")
        metric = metrics['bar_used_note']
        fig = plt.figure()
        fig.clf()
        txt = 'Average Bar used note: ' + str(round(np.mean(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'pitch_range' in list(metrics.keys()):
        print("pitch_range")
        fig = plt.figure()
        fig.clf()
        txt = 'Pitch range: ' + str(metrics['pitch_range'])
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'avg_pitch_shift' in list(metrics.keys()):
        print("avg_pitch_shift")
        fig = plt.figure()
        fig.clf()
        txt = 'Average Pitch Shift: ' + str(round(metrics['avg_pitch_shift'],3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'avg_IOI' in list(metrics.keys()):
        print("avg_IOI")
        fig = plt.figure()
        fig.clf()
        txt = 'Average inter-onset-intervals: ' + str(round(metrics['avg_IOI'],3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'note_length_hist' in list(metrics.keys()):
        print("note_length_hist")
        metric = metrics['note_length_hist']
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(notelengths)))
        ax.set_xticklabels(notelengths)
        plt.bar(range(len(metric)),metric,color=(0,0.498, 0.4, 1))
        plt.xlabel('Note Length')
        plt.title('Note Length Histogram')
        pp.savefig()

    if 'note_length_transition_matrix' in list(metrics.keys()):
        print("note_length_transition_matrix")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(notelengths)))
        ax.set_yticks(np.arange(len(notelengths)))
        ax.set_xticklabels(notelengths, fontsize=8)
        ax.set_yticklabels(notelengths, fontsize=8)
        plt.imshow( metrics['note_length_transition_matrix'] , interpolation='nearest', cmap=plt.cm.summer)
        plt.xlabel('Note length')           # and here ?
        plt.ylabel('Note length')          # and here ?
        plt.title('Note Length Transition Matrix')
        pp.savefig()

    if 'total_pitch_class_histogram' in list(metrics.keys()):
        print("total_pitch_class_histogram")
        metric = metrics['total_pitch_class_histogram']
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(pitches)))
        ax.set_xticklabels(pitches)
        plt.bar(range(len(metric)),metric,color=(0,0.498, 0.4, 1))
        plt.xlabel('Pitch')
        plt.title('Total Pitch Class Histogram')
        pp.savefig()

    if 'pitch_class_transition_matrix' in list(metrics.keys()):
        print("pitch_class_transition_matrix")
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(pitches)))
        ax.set_yticks(np.arange(len(pitches)))
        ax.set_xticklabels(pitches)
        ax.set_yticklabels(pitches)
        plt.imshow( metrics['pitch_class_transition_matrix'] , interpolation='nearest', cmap=plt.cm.summer)
        plt.xlabel('Pitch')           # and here ?
        plt.ylabel('Pitch')          # and here ?
        plt.title('Pitch Class Transition Matrix')
        pp.savefig()

    if 'bar_pitch_class_histogram' in list(metrics.keys()):
        print("bar_pitch_class_histogram")
        #todo

    pp.close()


absolute_niceplot(song1_eval)























#
#
#
# """statistic analysis: absolute measurement"""
#
# for i in range(0, len(metrics_list)):
#
#     print('\n------------------------------------------------')
#     print((metrics_list[i] + ':'))
#     print('------------------------------------------------')
#     print(' train_set')
#     print(('  mean: ', np.mean(set1_eval[metrics_list[i]], axis=0)))
#     # print(('  std: ', np.std(set1_eval[metrics_list[i]], axis=0)))
#
#     # print('------------------------')
#     print(' generated_test')
#     print(('  mean: ', np.mean(set2_eval[metrics_list[i]], axis=0)))
#     # print(('  std: ', np.std(set2_eval[metrics_list[i]], axis=0)))
#     from sklearn.preprocessing import normalize
#     if(metrics_list[i] is 'note_length_transition_matrix'):
#         fig = plt.figure()
#         ax = fig.add_subplot(2,2,1)
#         # ax.set_aspect('equal')
#         plt.imshow( np.mean(set1_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.bone)
#
#
#         ax = fig.add_subplot(2,2,2)
#         # ax.set_aspect('equal')
#         normalized = normalize(np.mean(set1_eval[metrics_list[i]], axis=0),axis=1, norm='l1')
#         plt.imshow(normalized, interpolation='nearest', cmap=plt.cm.bone)
#
#         a2 = fig.add_subplot(2,2,3)
#         # a2.set_aspect('equal')
#         plt.imshow( np.mean(set2_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.bone)
#
#         ax = fig.add_subplot(2,2,4)
#         # ax.set_aspect('equal')
#         normalized = normalize(np.mean(set2_eval[metrics_list[i]], axis=0),axis=1, norm='l1')
#         plt.imshow(normalized , interpolation='nearest', cmap=plt.cm.bone)
#
#         plt.colorbar()
#         plt.show()
#         #
#         # fig = plt.figure()
#         # ax = fig.add_subplot(1,1,1)
#         # ax.set_aspect('equal')
#         # plt.imshow( np.std(set1_eval[metrics_list[i]], axis=0) , interpolation='nearest', cmap=plt.cm.ocean)
#         # plt.colorbar()
#         # plt.show()

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
