# -*- coding: utf-8 -*-

import midi
import sys
import glob
import numpy as np
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

if(len(sys.argv) is 1) or (len(sys.argv) > 3):
    print "ERROR!\nUsage: " + sys.argv[0] + " <midi-folder> <opt: monochrome>"
setfolder = str(sys.argv[1])
midiset = glob.glob(setfolder + '/*.mid')

colormap = plt.cm.summer
histcolor = color=(0,0.498, 0.4, 1)
if(len(sys.argv) > 2):
    if sys.argv[2] == '-m':
        colormap = plt.cm.bone
        histcolor=(0,0, 0, 1)
    else:
        print("ERROR! bad argument")
        quit()


num_samples = len(midiset)
print "numsamples is " + str(num_samples)

eval = {'total_used_pitch':np.zeros((num_samples,1)),
               # 'bar_used_pitch':np.zeros((num_samples,1)),
               'total_used_note':np.zeros((num_samples,1)),
                #'bar_used_note':np.zeros((num_samples,1)),
               'total_pitch_class_histogram':np.zeros((num_samples,12)),
                #'bar_pitch_class_histogram':np.zeros((num_samples,12)),
               'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),
               'pitch_range':np.zeros((num_samples,1)),
               'avg_pitch_shift':np.zeros((num_samples,1)),
               'avg_IOI':np.zeros((num_samples,1)),
               'note_length_hist':np.zeros((num_samples,12)),
               'note_length_transition_matrix':np.zeros((num_samples,12,12))
            }

metrics_list = list(eval.keys())


for i in range(0,num_samples):
    # try:
    print "song " + midiset[i]
    feature = core.extract_feature(midiset[i])
    for metric_name in metrics_list:
        if (metric_name is 'pitch_class_transition_matrix') or (metric_name is 'note_length_transition_matrix'):
            (eval[metric_name])[i] = getattr(core.metrics(), metric_name)(feature)  #''',normalize = 2'''
        else:
            if (metric_name is 'bar_used_pitch') or (metric_name is 'bar_used_note'):
                (eval[metric_name])[i] = np.mean(getattr(core.metrics(), metric_name)(feature))
            else:
                vect = getattr(core.metrics(), metric_name)(feature)
                (eval[metric_name])[i] = vect

    # except:
    #     print("An exception occurred")

def add_value_labels(ax, labels, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for i in range(0,len(ax.patches)):
        rect = ax.patches[i]
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        val = round(labels[i],2)

        if(val > 0.01):
            # Use Y value as label and format number with one decimal place
            label = "std: " + str(val)
            # label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,
                size=6)                      # Vertically align label differently for
                                            # positive and negative values.




def absolute_niceplot (metrics):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('absmeasurement-' + setfolder.replace('/','') + '.pdf')

    pitches = ["C ","C#","D ","D#","E ","F ","F#","G ","G#","A ","Bb","B "]
    notelengths = ["1","1/2","1/4","1/8","1/16","3/4","3/8","3/16","3/32","1t","1/2t","1/4t"]



    if 'total_used_pitch' in list(metrics.keys()):
        print("Total used pitch")
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Total used pitch: ' + str(np.mean(metrics['total_used_pitch'])) +'\nSTD Total used pitch: ' + str(np.std(metrics['total_used_pitch']))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'bar_used_pitch' in list(metrics.keys()):
        print("bar_used_pitch")
        metric = metrics['bar_used_pitch']
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Average Bar used pitch: ' + str(round(np.mean(metric),3)) + '\nSTD Average Bar used pitch: ' + str(round(np.std(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'total_used_note' in list(metrics.keys()):
        print("total_used_note")
        metric = metrics['total_used_note']
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Total used note: ' + str(round(np.mean(metric),3)) + '\nSTD Total used note: ' + str(round(np.std(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'bar_used_note' in list(metrics.keys()):
        print("bar_used_note")
        metric = metrics['bar_used_note']
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Average Bar used note: ' + str(round(np.mean(metric),3)) + '\nSTD Average Bar used note: ' + str(round(np.std(metric),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'pitch_range' in list(metrics.keys()):
        print("pitch_range")
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Pitch range: ' + str(round(np.mean(metrics['pitch_range']))) + '\nSTD Pitch range: ' + str(round(np.std(metrics['pitch_range'])))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'avg_pitch_shift' in list(metrics.keys()):
        print("avg_pitch_shift")
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Average Pitch Shift: ' + str(round(np.mean(metrics['avg_pitch_shift']),3)) + '\nSTD Average Pitch Shift: ' + str(round(np.std(metrics['avg_pitch_shift']),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'avg_IOI' in list(metrics.keys()):
        print("avg_IOI")
        fig = plt.figure()
        fig.clf()
        txt = 'AVG Average inter-onset-intervals: ' + str(round(np.mean(metrics['avg_IOI']),3)) + '\nSTD Average inter-onset-intervals: ' + str(round(np.std(metrics['avg_IOI']),3))
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=18, ha="center")
        pp.savefig()

    if 'note_length_hist' in list(metrics.keys()):
        print("note_length_hist")
        ormetric = metrics['note_length_hist']
        stds = np.std(ormetric,axis=0)
        metric = np.mean(ormetric,axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(notelengths)))
        ax.set_xticklabels(notelengths)
        plt.bar(range(len(metric)),metric,color=histcolor)
        plt.xlabel('Note Length')
        plt.title('AVG Note Length Histogram')
        add_value_labels(ax,stds)
        pp.savefig()

    if 'note_length_transition_matrix' in list(metrics.keys()):
        print("note_length_transition_matrix")
        ogmetric = metrics['note_length_transition_matrix']
        metric = np.mean(ogmetric,axis=0)
        stds = np.std(ogmetric,axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(notelengths)))
        ax.set_yticks(np.arange(len(notelengths)))
        ax.set_xticklabels(notelengths, fontsize=8)
        ax.set_yticklabels(notelengths, fontsize=8)
        plt.imshow(metric, interpolation='nearest', cmap=colormap)
        plt.xlabel('Note length')           # and here ?
        plt.ylabel('Note length')          # and here ?
        plt.title('AVG Note Length Transition Matrix')
        plt.colorbar()

        # Uncomment to add standard deviation to the plot
        # for i in range(len(metric)):
        #     for j in range(len(metric[0])):
        #         text = ax.text(j, i, stds[i, j],
        #                        ha="center", va="center", color="bk")

        pp.savefig()

    if 'total_pitch_class_histogram' in list(metrics.keys()):
        print("total_pitch_class_histogram")
        ogmetric = metrics['total_pitch_class_histogram']
        metric = np.mean(ogmetric,axis=0)
        stds = np.std(ogmetric,axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(pitches)))
        ax.set_xticklabels(pitches)
        plt.bar(range(len(metric)),metric,color=histcolor)
        plt.xlabel('Pitch')
        plt.title('AVG Total Pitch Class Histogram')
        add_value_labels(ax,stds)
        pp.savefig()

    if 'pitch_class_transition_matrix' in list(metrics.keys()):
        print("pitch_class_transition_matrix")
        ogmetric = metrics['pitch_class_transition_matrix']
        metric = np.mean(ogmetric,axis=0)
        stds = np.std(ogmetric,axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks(np.arange(len(pitches)))
        ax.set_yticks(np.arange(len(pitches)))
        ax.set_xticklabels(pitches)
        ax.set_yticklabels(pitches)
        plt.imshow(metric, interpolation='nearest', cmap=colormap)
        plt.xlabel('Pitch')           # and here ?
        plt.ylabel('Pitch')          # and here ?
        plt.title('AVG Pitch Class Transition Matrix')
        plt.colorbar()

        # Uncomment to add standard deviation to the plot
        # for i in range(len(metric)):
        #     for j in range(len(metric[0])):
        #         if stds[i, j] > 0.0:
        #             text = ax.text(j, i, stds[i, j],
        #                            ha="center", va="center", color=(0,0,0,1))
        pp.savefig()
    #
    # if 'bar_pitch_class_histogram' in list(metrics.keys()):
    #     print("bar_pitch_class_histogram")
        #todo


    pp.close()


absolute_niceplot(eval)
