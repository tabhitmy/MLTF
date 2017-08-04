# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import glob
import math
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import itertools
import pickle
import time
from time import clock

from toolkitJ import cell2dmatlab_jsp
import GVal
from controlPanel_NFDA_J import controlPanel
from controlPanel_NFDA_J import processCodeEncoder
from controlPanel_NFDA_J import processCodeDecoder
from controlPanel_NFDA_J import initializationProcess


from sklearnTrainer import sklearnTrainer
from kerasTrainer import kerasTrainer
from datasetSeparation_NFDA_J import datasetSeparation
from feaSelection_NFDA_J import feaSelection
from dataBalance_NFDA_J import dataBalance
from labelProcessor_NFDA_J import labelProcessor
#
# @profile


def dataConstruction_NFDA(select_index, online_fea_selectindex, subject_sample_length, label_signal, online_fea_signal, noise_label_signal, path, save_flag):
    for subject_num in np.arange(len(select_index)):
        print('Data Constructing: (' + str(subject_num + 1) + '/' + str(len(select_index)) + ') ')
        for sample_num in np.arange(int(subject_sample_length[subject_num][0])):

            label_temp = np.zeros([4, 1])
            # Class label / Confident label
            for label_num in range(2):
                label_temp[label_num] = label_signal[subject_num][label_num][sample_num]

            # Noise label / Blink Num label
            for label_num in range(2, 4):
                label_temp[label_num] = noise_label_signal[subject_num][label_num - 2][sample_num]

            online_fea_temp = np.zeros([len(online_fea_selectindex), 1])
            for fea_num in range(len(online_fea_selectindex)):
                online_fea_temp[fea_num] = online_fea_signal[subject_num][fea_num][sample_num]

            if subject_num == 0 and sample_num == 0:
                label_all = label_temp
                online_fea_all = online_fea_temp
            else:
                label_all = np.concatenate((label_all, label_temp), axis=1)
                online_fea_all = np.concatenate((online_fea_all, online_fea_temp), axis=1)

    if save_flag:
        np.savetxt((path['pdata_path'] + 'label_all.txt'), label_all)
        np.savetxt((path['pdata_path'] + 'online_fea_all.txt'), online_fea_all)
        print('### Data saved! Saving directory is: ' + path['pdata_path'])

    return label_all, online_fea_all

#
# @profile


def labelReading_NFDA(select_index, data_file, path):

    # Initialize several matrix
    label_signal = cell2dmatlab_jsp([len(select_index), 2], 2, [])
    subject_sample_length = np.zeros([len(select_index), 1])
    active_index = []

    GVal.initPARA('label_raw_cache', {'name': 'data'})

    # Iplb LooPLaBel ,for the for loop
    lplb_count1 = 0
    for i in select_index:
        recordname = data_file[i]
        print(recordname)
        # Loading of labels.
        label_file = glob.glob(path['label_path'] + recordname + '*.txt')
        if not label_file:
            lplb_count1 += 1
            continue
        active_index.append(lplb_count1)

        if recordname in GVal.getPARA('label_raw_cache').keys():
            label_raw = GVal.getPARA('label_raw_cache')[recordname]
            # print('^^^' * 100)
        else:
            label_raw = np.loadtxt(label_file[0])
            GVal.getPARA('label_raw_cache')[recordname] = label_raw
            # print('&&&' * 100)
            # GVal.setPARA('label_raw_cache', {recordname: label_raw})

        subject_sample_length[lplb_count1] = len(label_raw)

        # Label
        label_signal[lplb_count1][0] = label_raw[:, 1]
        # Label Confidence
        label_signal[lplb_count1][1] = label_raw[:, 2]

        lplb_count1 += 1

    active_index += select_index[0]
    return active_index, subject_sample_length, label_signal

#
# @profile


def featurePreparation_NFDA(select_index, online_fea_selectindex, noise_label_index, active_index, data_file):
    # Concatenate all the subjects together in one huge matrix.
    # Comprehensive feature preparation.
    # Bluring the concept of subjects. Combine all the labels together.

    # [TODO] Add the Developing Features
    dev_fea_signal = []
    GVal.initPARA('online_fea_raw_cache', {'name': 'data'})
    online_fea_signal = cell2dmatlab_jsp([len(select_index), len(online_fea_selectindex)], 2, [])
    noise_label_signal = cell2dmatlab_jsp([len(select_index), 2], 2, [])
    lplb_count1 = 0
    for i in select_index:
        if i not in active_index:
            lplb_count1 += 1
            continue
        recordname = data_file[i]
        # Loading of the online features (Developed)
        online_fea_file = glob.glob(path['online_fea_path'] + recordname + '*.txt')

        if recordname in GVal.getPARA('online_fea_raw_cache').keys():
            online_fea_raw = GVal.getPARA('online_fea_raw_cache')[recordname]
        else:
            online_fea_raw = np.loadtxt(online_fea_file[0])
            GVal.getPARA('online_fea_raw_cache')[recordname] = online_fea_raw

        j_lplb_count = 0
        for j in noise_label_index:
            noise_label_signal[lplb_count1][j_lplb_count] = online_fea_raw[:, j]
            j_lplb_count += 1

        # Online Developed Features
        j_lplb_count = 0
        for j in online_fea_selectindex:
            online_fea_signal[lplb_count1][j_lplb_count] = online_fea_raw[:, j]
            j_lplb_count += 1

        lplb_count1 += 1
    return online_fea_signal, dev_fea_signal, noise_label_signal


#


def featurePlotting_NFDA(label_all, online_fea_all):
    nega_index_raw = np.nonzero(label_all[0] == 0)[0]
    posi1_index_raw = np.nonzero(label_all[0] == 1)[0]
    posi2_index_raw = np.nonzero(label_all[0] == 2)[0]
    posi3_index_raw = np.nonzero(label_all[0] == 3)[0]

    # [nega, posi1, posi2, posi3]
    colorlist = ['#0d75f8', '#ff9a8a', '#ef4026', '#8f1402']

    # [TODO] The sample selecting and balancing.(Normally two things to do: 1. Downsampling the negative; 2. Merging the positive)
    nega_total_num = len(nega_index_raw)
    posi_total_num = len(posi1_index_raw) + len(posi2_index_raw) + len(posi3_index_raw)
    nega_ds = int(nega_total_num / posi_total_num)
    nega_index = nega_index_raw[1: nega_total_num: nega_ds]
    posi1_index = posi1_index_raw
    posi2_index = posi2_index_raw
    posi3_index = posi3_index_raw

    print(len(nega_index), len(posi1_index), len(posi2_index), len(posi3_index))
    pic_num = 1
    for fea1_num in range(10):
        for fea2_num in range(fea1_num + 1, 10):
            h = plt.figure(num=str(pic_num), figsize=(17, 9.3))
            plt.plot(online_fea_all[fea1_num][nega_index], online_fea_all[fea2_num][nega_index], '.', color=colorlist[0], label='No Fatigue')
            plt.plot(online_fea_all[fea1_num][posi1_index], online_fea_all[fea2_num][posi1_index], '.', color=colorlist[1], label='Fatigue LV7')
            plt.plot(online_fea_all[fea1_num][posi2_index], online_fea_all[fea2_num][posi2_index], '.', color=colorlist[2], label='Fatigue LV8')
            plt.plot(online_fea_all[fea1_num][posi3_index], online_fea_all[fea2_num][posi3_index], '.', color=colorlist[3], label='Fatigure LV9')
            plt.legend(loc='best', prop=zhfont)
            plt.xlabel(online_fea_name[fea1_num], FontProperties=zhfont)
            plt.ylabel(online_fea_name[fea2_num], FontProperties=zhfont)
            plt.title('Figure #' + str(pic_num) + ' With Recording Num: ' + str(select_index))
            plt.show()
            plt.savefig((work_path + '/pic/Figure' + str(pic_num) + '.png'))
            print('Picture' + str(pic_num) + 'Saved!')
            plt.close(h)
            pic_num += 1
    return 0


def mainProcesser(process_code):
    select_index, classifier_num = processCodeDecoder(process_code, 1)

    online_fea_selectindex = GVal.getPARA('online_fea_selectindex_PARA')
    noise_label_index = GVal.getPARA('noise_label_index_PARA')
    ################################################################################
    ############## [ Engine START! ] ###################################################
    ################################################################################

    ####################################
    # [TODO] Raw Data Reading methods
    #    FUNCTION     rawDataReading_NFDA()

    ####################################
    # [TODO] New preprocessing methods
    #　FUNCTION     raw_dev_fea = preProcessing_NFDA()

    # Output raw_dev_fea should then become input of the Function featurePreparation_NFDA()

    if FLAG['data_prepare_flag'] and GVal.getPARA('firstTimeConstruction'):
        active_index, subject_sample_length, label_signal = labelReading_NFDA(select_index, data_file, path)
        online_fea_signal, dev_fea_signal, noise_label_signal = featurePreparation_NFDA(select_index, online_fea_selectindex, noise_label_index, active_index, data_file)
        label_all, online_fea_all = dataConstruction_NFDA(select_index, online_fea_selectindex, subject_sample_length, label_signal, online_fea_signal, noise_label_signal, path, FLAG['save_flag'])
        GVal.setPARA('firstTimeConstruction', 0)
    else:
        print('### Skip the data construction, load from the database')
        print('### The loading data directory is: ' + path['pdata_path'])
        label_all = np.loadtxt((path['pdata_path'] + 'label_all.txt'))
        online_fea_all = np.loadtxt((path['pdata_path'] + 'online_fea_all.txt'))
    X_rawraw = np.transpose(online_fea_all)
    Y_raw = np.transpose(label_all)
    print('######## [Data Preparation Accomplished! ] ########')
    print('### Label  Raw Matrix Size: ' + str(Y_raw.shape))
    print('### Feature  Raw Matrix Size: ' + str(X_rawraw.shape))

    # Until here, all the developed label and features are prepared in a certain format.
    # for one single data sample, get the label by:
    #   snum:  Sample NUM  ||    fnum:   Feature NUM
    #   label_all[0,snum]  : label
    #   label_all[1,snum]  : confidence
    #   online_fea_all[fnum,snum] : difference features.

    # Feature Plotting [N/A]
    if FLAG['plotting_flag']:
        featurePlotting_NFDA(label_all, online_fea_all)
    else:
        print('### No plotting, exit feature plotting...')

    # Prepare the F, Y

    # X_raw[snum, fnum]
    # Y[snum,:]
    if FLAG['label_process_flag']:
        X_raw, Y = labelProcessor(X_rawraw, Y_raw)
    else:
        X_raw = copy.deepcopy(X_rawraw)
        Y = copy.deepcopy(Y_raw)

    ####################################
    # [TODO IN PROGRESS]  Add the feature selection Method
    if FLAG['fea_selection_flag']:
        X = feaSelection(X_raw)
    else:
        X = copy.deepcopy(X_raw)

    ####################################
    # [TODO] Feature Regulation
    # Three types of features:
    #     1 -- Continuous
    #     2 -- Discrete (Categorical)
    #     3 -- Discrete Binary (0 and 1)

    ####################################
    # [TODO DONE!]  Compare with Y. Do the validation .   Try leave-one-out, or K-fold
    X_train, X_validation, y_train, y_validation, N = datasetSeparation(X, Y, GVal.getPARA('split_type_num'), GVal.getPARA('split_type_para'))
    # Sample: X_train[n_fold][fnum,snum]

    classifier_frame = {
        3: sklearnTrainer,
        2: sklearnTrainer,
        1: kerasTrainer
    }

    # Loop working when n fold is working.

    for N_lplb_count in np.arange(N):
        ####################################
        # [TODO IN PROGRESS] Data balancing method!
        # [IMPORTANT!] Data balance should be after the separation and only affect on training set!
        if FLAG['data_balance_flag']:
            X_train[N_lplb_count], y_train[N_lplb_count] = dataBalance(X_train[N_lplb_count], y_train[N_lplb_count])

        res_temp = cell2dmatlab_jsp([1, 4], 2, [])

        # [TODO IN PROGRESS] Classifier Designing
        # Try different classifiers, Get CY
        print(' ')
        print('#' * 35)
        print('######## [Fold (#' + str(N_lplb_count + 1) + '/' + str(N) + ') ] ########')
        print('#' * 35)
        res_temp[0][0] = process_code
        res_temp[0][1:4] = classifier_frame[int(str(classifier_num)[0])](classifier_num, X_train[N_lplb_count], y_train[N_lplb_count], X_validation[N_lplb_count], y_validation[N_lplb_count], path)
        # Sample: res[fold number][infoserial]
        # infoserial :  0- model object   1- Training Score  2- FRAP
        if N_lplb_count == 0:
            res = res_temp
        else:
            res += res_temp
        time.sleep(0.001)
    return res, N

if __name__ == "__main__":
    start = clock()
    path_prefix = initializationProcess()
    path = {
        'data_path': (path_prefix + 'public/EEG_RawData/'),
        'label_path': (path_prefix + 'GaoMY/EXECUTION/NFDA/Label/'),
        'parent_path': (path_prefix + 'GaoMY/EXECUTION/NFDA/code/'),
        'online_fea_path': (path_prefix + 'GaoMY/EXECUTION/NFDA/online_new/')
    }

    path['work_path'] = (path['parent_path'] + 'python_code/')
    path['pdata_path'] = (path['parent_path'] + 'data/')

    GVal.setPARA('path_PARA', path)
    os.chdir(path['work_path'])

    # RawData Reading.
    data_file = sorted(os.listdir(path['data_path']))

    FLAG, process_code_pack = controlPanel()

    # ProcessingCodeLoop
    L_lplb_count = 0
    L = len(process_code_pack)
    for process_code in process_code_pack:
        GVal.setPARA('process_code_PARA', process_code)
        res_singleLOOP, N = mainProcesser(process_code)
        if L_lplb_count == 0:
            res = res_singleLOOP
        else:
            res += res_singleLOOP
        L_lplb_count += 1
        print(' ')
        print('$' * 70)
        print('####### [ Loop (#' + str(L_lplb_count) + '/' + str(L) + ')DONE! ] #######')
        print('$' * 70)
        time.sleep(0.001)

    # [TODO] Result Visualization  (Code referring the picturing.py)
    # [Tempory Code]Now saving the res data into a pickle file.
    #   And read in picturing and plotting. Later this will be moduled
    #   and parameters can be tuned in control panel
    with open('res.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)

    L_total = len(res)
    F_res_store = np.zeros((L_total, 7))
    print(' ')
    print('#' * 15 + ' [SINGLE FOLD OUTPUT, ' + str(L) + ' process packs in total] ' + '#' * 20 + ' [Loop Para Name: ' + GVal.getPARA('loopPARA_name') + ']' + '#' * 38)
    beta = GVal.getPARA('beta_PARA')
    res_lplb_count = 0
    res_c = 1
    for i in range(L_total):
        res_lplb_count += 1
        if (res_lplb_count % N) == 1:
            print('=' * 15 + ' [Process Pack NO.' + str(res_c) + ', ' + str(N) + ' folds in total] ' + '=' * 91)
            res_c += 1
        F_res_store[i, 0] = res[i][3][0]
        F_res_store[i, 1] = res[i][3][1]
        F_res_store[i, 2] = res[i][3][2]
        F_res_store[i, 3] = res[i][3][3]
        F_res_store[i, 4] = res[i][3][4]
        F_res_store[i, 5] = res[i][3][5]
        F_res_store[i, 6] = res[i][3][6]
        print(('###[P: ' + str(round(res[i][3][0], 4)) + ' ' * (6 - len(str(round(res[i][3][0], 4)))) +
               ' || A: ' + str(round(res[i][3][1], 4)) + ' ' * (6 - len(str(round(res[i][3][1], 4)))) +
               ' || R: ' + str(round(res[i][3][2], 4)) + ' ' * (6 - len(str(round(res[i][3][2], 4)))) +
               ' || MA: ' + str(round(res[i][3][3], 4)) + ' ' * (6 - len(str(round(res[i][3][3], 4)))) +
               ' || FA: ' + str(round(res[i][3][4], 4)) + ' ' * (6 - len(str(round(res[i][3][4], 4)))) +
               ' || F1: ' + str(round(res[i][3][5], 4)) + ' ' * (6 - len(str(round(res[i][3][5], 4)))) +
               ' || F' + str(beta) + ': ' + str(round(res[i][3][6], 4)) + ' ' * (6 - len(str(round(res[i][3][6], 4)))) +
               '] | [PCode: ' + str(res[i][0]) + ' ' * (10 - len(str(res[i][0]))) +
               '] | [Classifier: ' + res[i][1] + ' ' * (17 - len(res[i][1])) +
               ']###'
               ))
        if (res_lplb_count % N) == 0:
            print('-' * 15 + ' [MEAN OUTPUT ] ' + '-' * 126)
            print(('###[P: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 0]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 0]) / N, 4)))) +
                   ' || A: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 1]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 1]) / N, 4)))) +
                   ' || R: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 2]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 2]) / N, 4)))) +
                   ' || MA: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 3]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 3]) / N, 4)))) +
                   ' || FA: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 4]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 4]) / N, 4)))) +
                   ' || F1: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 5]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 5]) / N, 4)))) +
                   ' || F' + str(beta) + ': ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 6]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 6]) / N, 4)))) +
                   '] | [PCode: ' + str(res[res_lplb_count - 1][0]) + ' ' * (10 - len(str(res[res_lplb_count - 1][0]))) +
                   '] | [Classifier: ' + res[res_lplb_count - 1][1] + ' ' * (17 - len(res[res_lplb_count - 1][1])) +
                   ']###'
                   ))
            print('=' * 157)

    print('#' * 157)
    print(' ')
    finish = clock()
    print('######## [ Time Consumed: ' + str(round((finish - start), 4)) + 's ] ############')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    ####################################
    # [TODO]  Training the classifier. Calculate the CY. According to the return information

    # X_train, X_test, y_train, y_test = dataSet4TrainTestValidation(X, Y, 5, 0.3)

    ####################################
    # [TODO! DONE! FRAP] ROC method. And other Evaluator

    ####################################
    # [TODO] FINAL EVAULATION. Output.
