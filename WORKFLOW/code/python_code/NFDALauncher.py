# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import glob
import math
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
# zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import itertools
import pickle
import time
from time import clock

import scipy.io as sio

import toolkitJ
from toolkitJ import cell2dmatlab_jsp
from toolkitJ import Logger_J
from toolkitJ import listtranspose_J

import GVal
from controlPanel_NFDA_J import controlPanel
from controlPanelSubFunc_NFDA_J import initializationProcess
from controlPanelSubFunc_NFDA_J import processCodeDecoder


from sklearnTrainer import sklearnTrainer
from kerasTrainer import kerasTrainer
from datasetSeparation_NFDA_J import datasetSeparation
from feaSelection_NFDA_J import feaSelection
from dataBalance_NFDA_J import dataBalance
from labelProcessor_NFDA_J import labelProcessor


from plotting_NFDA_J import featurePlotting
from plotting_NFDA_J import resultPlotting
from plotting_NFDA_J import FRAPPlotting
from plotting_NFDA_J import cFRAPPlotting
from plotting_NFDA_J import pairDistPlotting

from trainerSubFunc_NFDA_J import calculateFRAP

from labelProcessor_NFDA_J import noiseFrameDilation
#
# @profile


def dataConstruction(select_index, online_fea_selectindex, subject_sample_length, label_signal, online_fea_signal, noise_label_signal, path, save_flag):
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

            # Developed Feature
            online_fea_temp = np.zeros([len(online_fea_selectindex), 1])
            for fea_num in range(len(online_fea_selectindex)):
                online_fea_temp[fea_num] = online_fea_signal[subject_num][fea_num][sample_num]

            # Append each temp to the corresponding standard output
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

        with open(path['pdata_path'] + 'label_signal.pickle', 'wb') as outfile:
            pickle.dump(label_signal, outfile)
        with open(path['pdata_path'] + 'noise_label_signal.pickle', 'wb') as outfile:
            pickle.dump(noise_label_signal, outfile)
        with open(path['pdata_path'] + 'online_fea_signal.pickle', 'wb') as outfile:
            pickle.dump(online_fea_signal, outfile)

    return label_all, online_fea_all

#
# @profile


def labelReading(select_index, data_file, path):

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
        # print(path['label_path'] + recordname + '*.txt')
        if not label_file:
            # return if no file in local database
            lplb_count1 += 1
            continue
        active_index.append(lplb_count1)

        # If the
        if recordname in GVal.getPARA('label_raw_cache').keys():
            label_raw = GVal.getPARA('label_raw_cache')[recordname]
        else:
            label_raw = np.loadtxt(label_file[0])
            GVal.getPARA('label_raw_cache')[recordname] = label_raw

        # Save the total frame length of the current recording
        subject_sample_length[lplb_count1] = len(label_raw)

        # Label
        label_signal[lplb_count1][0] = label_raw[:, 1]
        # Label Confidence
        label_signal[lplb_count1][1] = label_raw[:, 2]

        lplb_count1 += 1

    active_index += select_index[0]
    # Save for FINAL EVALUATION
    GVal.setPARA('label_signal_PARA', label_signal)
    return active_index, subject_sample_length, label_signal

#
# @profile


def featurePreparation(select_index, online_fea_selectindex, noise_label_index, active_index, data_file):
    # Concatenate all the subjects together in one huge matrix.
    # Comprehensive feature preparation.
    # Bluring the concept of subjects. Combine all records together.

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

        # Noise label information. (Contain noise and noblink label information)
        j_lplb_count = 0
        for j in noise_label_index:
            noise_label_signal[lplb_count1][j_lplb_count] = online_fea_raw[:, j]
            j_lplb_count += 1

        # Online Developed Features
        j_lplb_count = 0
        for j in online_fea_selectindex:
            online_fea_signal[lplb_count1][j_lplb_count] = online_fea_raw[:, j]
            j_lplb_count += 1
        # exit()
        lplb_count1 += 1

        # Save for FINAL EVALUATION
        # GVal.setPARA('feas_PARA',online_fea_raw[])
        GVal.setPARA('online_fea_signal_PARA', online_fea_signal)
        GVal.setPARA('noise_label_signal_PARA', noise_label_signal)
    return online_fea_signal, dev_fea_signal, noise_label_signal


#


def mainProcesser(process_code):
    # The decoder translate the process_code into different parameter to be set.
    select_index, classifier_num = processCodeDecoder(process_code)
    # Prepare some information
    online_fea_selectindex = GVal.getPARA('online_fea_selectindex_PARA')
    noise_label_index = GVal.getPARA('noise_label_index_PARA')

    ####################################
    # [TODO] Raw Data Reading methods
    #    FUNCTION     rawDataReading()

    ####################################
    # [TODO] New preprocessing methods
    #　FUNCTION     raw_dev_fea = preProcessing()

    # Output raw_dev_fea should then become input of the Function featurePreparation()
    if FLAG['data_prepare_flag'] and GVal.getPARA('firstTimeConstruction'):
        # Load the label information from local database
        active_index, subject_sample_length, label_signal = labelReading(select_index, data_file, path)
        # Load and prepare the feature from local database
        online_fea_signal, dev_fea_signal, noise_label_signal = featurePreparation(select_index, online_fea_selectindex, noise_label_index, active_index, data_file)
        # Construct all the information, make a standard format. [label_all] & [online_fea_all]
        label_all, online_fea_all = dataConstruction(select_index, online_fea_selectindex, subject_sample_length, label_signal, online_fea_signal, noise_label_signal, path, FLAG['save_flag'])
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

    ####################################
    # [TODO DONE!]  Compare with Y. Do the validation .   Try leave-one-out, or K-fold
    X_train_rawraw, X_validation_raw, y_train_raw, y_validation_raw, N = datasetSeparation(X_rawraw, Y_raw, GVal.getPARA('split_type_num'), GVal.getPARA('split_type_para'))
    # Sample: X_train[n_fold][fnum,snum]

    classifier_frame = {
        3: sklearnTrainer,
        2: sklearnTrainer,
        1: kerasTrainer
    }

    # Loop working when n fold is working.
    for N_lplb_count in np.arange(N):
        print(' ')
        print('#' * 32)
        print('######## [Fold (#' + str(N_lplb_count + 1) + '/' + str(N) + ') ] ########')

        # Prepare the F, Y

        if FLAG['label_process_flag']:
            # label processor  targeting on noconfident frame / noise frame / noblink frame 'N3 frame'
            # Extendable function strucutre. For any customized label processing tasks.
            X_train_raw, y_train, X_validation_rawS, y_validation_rawS = labelProcessor(X_train_rawraw[N_lplb_count], y_train_raw[N_lplb_count], X_validation_raw[N_lplb_count], y_validation_raw[N_lplb_count])
        else:
            X_train_raw = copy.deepcopy(X_train_rawraw[N_lplb_count])
            y_train = copy.deepcopy(y_train_raw[N_lplb_count])
            X_validation_rawS = copy.deepcopy(X_validation_raw[N_lplb_count])
            y_validation_rawS = copy.deepcopy(y_validation_raw[N_lplb_count])
        # # # Feature Plotting [N/A]
        # if FLAG['fea_plotting_flag']:
        #     featurePlotting(Y_train, X_train_raw, process_code)
        # else:
        #     print('### No plotting, exit feature plotting...')
        ####################################
        # [TODO IN PROGRESS]  Add the feature selection Method
        if FLAG['fea_selection_flag']:
            # feaSelection is doing the feature selection only on trainning set, return the fea selection result
            X_train, feaS_mdl = feaSelection(X_train_raw)
            # According the fea selection result, preprocess  validation set in corresponding to the  trainning set.
            X_validation = feaS_mdl.fit_transform(X_validation_rawS)
        else:
            X_train = copy.deepcopy(X_train_raw)
            X_validation = copy.deepcopy(X_validation_rawS)

        ####################################
        # [TODO] Feature Regulation
        # Three types of features:
        #     1 -- Continuous
        #     2 -- Discrete (Categorical)
        #     3 -- Discrete Binary (0 and 1)

        y_validation = copy.deepcopy(y_validation_rawS)

        ####################################
        # [TODO IN PROGRESS] Data balancing method!
        # Data balance only affect on training set.
        # The method is solving the unbalance problem.
        # The function is extendable, customized data balancing method can be applied.
        if FLAG['data_balance_flag']:
            X_train, y_train = dataBalance(X_train, y_train)

        res_temp = cell2dmatlab_jsp([1, 5], 2, [])

        # [TODO IN PROGRESS] Classifier Designing

        res_temp[0][0] = process_code
        res_temp[0][1:4] = classifier_frame[int(str(classifier_num)[0])](classifier_num, X_train, y_train, X_validation, y_validation, path)
        # Sample: res[fold number][infoserial]
        # infoserial :  1- model object [1: num, 2:name, 3:mdl obejct]  2- Training Score  3- FRAP
        resCalMethodList = {
            'median': np.median,
            'mean': np.mean
        }
        if N == 1:
            res = res_temp
        else:
            if N_lplb_count == 0:
                res_mid = res_temp
            elif N_lplb_count < N - 1:
                res_mid += res_temp
            elif N_lplb_count == N - 1:
                res_mid += res_temp
                res = res_temp
                for ires in range(len(res[0][3])):
                    temptemp = []
                    for iN in range(N):
                        # if np.logical_and(res_mid[iN][3][ires] != 0.0, res_mid[iN][3][ires] != 1.0):
                        if res_mid[iN][3][ires] not in [0.0, 1.0, -1.0]:
                            temptemp.append(res_mid[iN][3][ires])
                    res[0][3][ires] = resCalMethodList[GVal.getPARA('resCalMethod_PARA')](temptemp)

        if FLAG['pairDristributionPlotting_flag']:
            FRAP = res_temp[0][3]
            pairDistPlotting(FRAP, process_code)
        if FLAG['resultDristributionPlotting_flag']:
            FRAP = res_temp[0][3]
            resultPlotting(FRAP, process_code)

        else:
            print('### No plotting, exit feature plotting...')

        time.sleep(0.001)

    return res, N


if __name__ == "__main__":
    start = clock()
    # Gain the path prefix, this is added to suit up different processing environment.
    path_prefix, username_raw = initializationProcess()
    if username_raw[-4:] == '4BEA':
        username = username_raw[:-5]
        standardMode = 1
    else:
        username = username_raw
        standardMode = 0
    # print(path_prefix + username + '/EXECUTION/NFDA/code/tlog.txt')
    sys.stdout = Logger_J(path_prefix + username + '/EXECUTION/NFDA/code/tlog.txt')

    print('### Log starting!')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # Define the basic path.
    path = {
        'data_path': (path_prefix + 'public/EEG_RawData/'),
        'label_path': (path_prefix + 'public/backend_data/Label/'),
        'online_fea_path': (path_prefix + 'public/backend_data/online_new/'),
        'test_fea_path': (path_prefix + 'public/backend_data/test_fea/'),
        'parent_path': (path_prefix + username + '/EXECUTION/NFDA/code/')
    }

    # Define the several sub path
    # work path, in which all the processing code are placed
    path['work_path'] = (path['parent_path'] + 'python_code/')
    # fig path, to save the result figs
    path['fig_path'] = (path['parent_path'] + 'python_code/fig/')
    # pdata_path, to save several types of processing data, (temporial data for accelerating the procedure)
    path['pdata_path'] = (path['parent_path'] + 'python_code/data/')

    # Global Statement of [path]
    GVal.setPARA('path_PARA', path)

    # Change the current path if necessary.
    os.chdir(path['work_path'])

    # RawData folder list reading.
    data_file_temp = sorted(os.listdir(path['online_fea_path']))
    data_file = data_file_temp
    knnn = 0
    for i in data_file_temp:
        print(str(knnn) + ' || ' + i)
        data_file[knnn] = data_file_temp[knnn][:-4]
        knnn += 1
    # exit()
    # Processing the Control Panel! All the important configuration are pre-defined in control panel.
    # Refer the file controlPanel_NFDA_J.py for more details.
    # [FLAG] is a flag controller, to switch on/off several processing procedure.
    # [process_code_pack] is a package containing all the targeting process loops, each loop has an independent process_code.

    FLAG, process_code_pack = controlPanel(username, standardMode)
    GVal.setPARA('FLAG_PARA', FLAG)
    # ProcessingCodeLoop
    # The core processing loop. Inside pay attention to mainProcesser()
    L_lplb_count = 0
    L = len(process_code_pack)

    for process_code in process_code_pack:
        print(' ')
        print('$' * 150)
        print('$$$$$$$$ [ Loop (#' + str(L_lplb_count + 1) + '/' + str(L) + ')Start! ] ' + '$' * 119)
        print('$' * 150)
        GVal.setPARA('process_code_PARA', process_code)
        res_singleLOOP, N = mainProcesser(process_code)
        if L_lplb_count == 0:
            res = res_singleLOOP
        else:
            res += res_singleLOOP
        L_lplb_count += 1
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        time.sleep(0.001)

    if FLAG['plotting_flag']:
        FRAPPlotting(res)
        cFRAPPlotting(res)
        # [TODO] Result Visualization  (Code referring the picturing.py)
        # [Tempory Code]Now saving the res data into a pickle file.
        #   And read in picturing and plotting. Later this will be moduled
        #   and parameters can be tuned in control panel

        # with open('res.pickle', 'wb') as outfile:
        #     pickle.dump(res, outfile)

    L_total = len(res)
    F_res_store = np.zeros((L_total, 7))
    print(' ')
    # print('#' * 15 + ' [SINGLE FOLD OUTPUT, ' + str(L) + ' process packs in total] ' + '#' * 20 + ' [Loop Para Name: ' + GVal.getPARA('loopPARA_name') + ']' + '#' * 38)
    beta = GVal.getPARA('beta_PARA')
    res_lplb_count = 0
    res_c = 1
    for i in range(L_total):
        res_lplb_count += 1
        # if (res_lplb_count % N) == 1:
        #     print('=' * 15 + ' [Process Pack NO.' + str(res_c) + ', ' + str(N) + ' folds in total] ' + '=' * 105)
        #     res_c += 1
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
               '] | [Classifier: ' + res[i][1][1] + ' ' * (17 - len(res[i][1])) +
               ']###'
               ))
        # if (res_lplb_count % N) == 0:
        #     print('-' * 15 + ' [MEAN OUTPUT ] ' + '-' * 128)
        #     print(('###[P: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 0]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 0]) / N, 4)))) +
        #            ' || A: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 1]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 1]) / N, 4)))) +
        #            ' || R: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 2]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 2]) / N, 4)))) +
        #            ' || MA: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 3]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 3]) / N, 4)))) +
        #            ' || FA: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 4]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 4]) / N, 4)))) +
        #            ' || F1: ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 5]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 5]) / N, 4)))) +
        #            ' || F' + str(beta) + ': ' + str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 6]) / N, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[res_lplb_count - N:res_lplb_count, 6]) / N, 4)))) +
        #            '] | [PCode: ' + str(res[res_lplb_count - 1][0]) + ' ' * (10 - len(str(res[res_lplb_count - 1][0]))) +
        #            '] | [Classifier: ' + res[res_lplb_count - 1][1] + ' ' * (17 - len(res[res_lplb_count - 1][1])) +
        #            ']###'
        #            ))
        #     print('=' * 159)

    print('#' * 159)
    print('-' * 15 + ' [BIG MEAN OUTPUT ] ' + '-' * 124)
    print(('###[P: ' + str(round(sum(F_res_store[:, 0]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 0]) / L_total, 4)))) +
           ' || A: ' + str(round(sum(F_res_store[:, 1]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 1]) / L_total, 4)))) +
           ' || R: ' + str(round(sum(F_res_store[:, 2]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 2]) / L_total, 4)))) +
           ' || MA: ' + str(round(sum(F_res_store[:, 3]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 3]) / L_total, 4)))) +
           ' || FA: ' + str(round(sum(F_res_store[:, 4]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 4]) / L_total, 4)))) +
           ' || F1: ' + str(round(sum(F_res_store[:, 5]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 5]) / L_total, 4)))) +
           ' || F' + str(beta) + ': ' + str(round(sum(F_res_store[:, 6]) / L_total, 4)) + ' ' * (6 - len(str(round(sum(F_res_store[:, 6]) / L_total, 4)))) +
           ']###'
           ))

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

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    indep_testdata_flag = 0
    delete_noise_RAE = 1

    Portion_Overall = []
    FA_Overall = []
    RAT_Overallmean = []
    RAT_Overallmedian = []
    FINALSCORE1 = []
    FINALSCORE2 = []
    FINALSCORE3 = []

    if indep_testdata_flag:
        print(path['test_fea_path'])
        testdata_file_temp = sorted(os.listdir(path['test_fea_path']))
        testdata_file = testdata_file_temp
        knnn = 0
        for i in testdata_file_temp:
            print(str(knnn) + ' || ' + i)
            testdata_file[knnn] = testdata_file_temp[knnn][:-4]
            knnn += 1
        testlist_RAE = np.arange(46)
        online_fea_selectindex = GVal.getPARA('online_fea_selectindex_PARA')
        noise_label_index = GVal.getPARA('noise_label_index_PARA')
        active_index, subject_sample_length, label_signal = labelReading(testlist_RAE, testdata_file, path)
        online_fea_signal, dev_fea_signal, noise_label_signal = featurePreparation(testlist_RAE, online_fea_selectindex, noise_label_index, active_index, testdata_file)
        label_RAE = GVal.getPARA('label_signal_PARA')
        noise_label_RAE = GVal.getPARA('noise_label_signal_PARA')
        online_fea_RAE = GVal.getPARA('online_fea_signal_PARA')
    else:
        testlist_RAE = GVal.getPARA('recording_index_list_PARA')[0]
        testdata_file = data_file
        if FLAG['data_prepare_flag']:
            label_RAE = GVal.getPARA('label_signal_PARA')
            noise_label_RAE = GVal.getPARA('noise_label_signal_PARA')
            online_fea_RAE = GVal.getPARA('online_fea_signal_PARA')
        else:
            with open(path['pdata_path'] + 'label_signal.pickle', 'rb') as outfile:
                label_RAE = pickle.load(outfile)
                # [subject_num][label_num][sample_num]
            with open(path['pdata_path'] + 'noise_label_signal.pickle', 'rb') as outfile:
                noise_label_RAE = pickle.load(outfile)
                #[subject_num][label_num][sample_num]
            with open(path['pdata_path'] + 'online_fea_signal.pickle', 'rb') as outfile:
                online_fea_RAE = pickle.load(outfile)
                #[subject_num][fea_num][sample_num]
    for i in range(L_total):
        clf_RAE = res[i][1][2]
        print('######## [ REAL APPLICATION EVALUATION ] ############')

        # print(len(label_RAE))
        # exit()
        target_group = []
        fullfati_group = []
        midlvl_group = []
        nofati_group = []
        all_group = []
        RAT_temp = []
        FA_RAE_temp = []
        Nnega_RAE_temp = []
        FP_RAE_temp = []
        FPL1_RAE_temp = []
        FPL0_RAE_temp = []
        undetect_Fati_temp = []
        N_fati_temp = []
        N_fati_all = []
        M_fati_all = []
        satisgroup = 0
        posigroup = 0
        negagroup = 0
        unsatisgroup = 0
        negaamount = 0
        posiamount = 0
        N_f1 = 16
        N_f2 = 17
        name_f1 = GVal.getPARA('online_fea_engname_PARA')[N_f1][0]
        name_f2 = GVal.getPARA('online_fea_engname_PARA')[N_f2][0]
        print(len(label_RAE))

        isub_jump = [7, 10, 14, 37, 43, 49, 50, 53]
        # The For loop traverse all the recording . In each single loop, various calculation made.
        for isubject in range(len(label_RAE)):
            # if isubject in isub_jump:
                # continue
            # Apply the newlabel rule on the raw label file. The details of the rule is refered in excel file in NFDA folder.
            a_RAE = label_RAE[isubject][0]
            raw_a_RAE = copy.deepcopy(a_RAE)
            raw_a_RAE_matlab = copy.deepcopy(a_RAE)
            l_RAE = len(a_RAE)
            y_RAE = 5
            lplb_RAE = 0
            no_jump = 1
            while lplb_RAE + 1 < l_RAE:
                # if no_jump:
                lplb_RAE += 1
                if a_RAE[lplb_RAE - 1] <= 1 and a_RAE[lplb_RAE] > 1:
                    for k in range(1, y_RAE):
                        if a_RAE[lplb_RAE + k] < 1:
                            break
                        elif a_RAE[lplb_RAE + k] > 2:
                            a_RAE[lplb_RAE + k] = 3
                            k -= 1
                            break
                        else:
                            a_RAE[lplb_RAE + k] = 2
                    lplb_RAE += k
                    no_jump = 0
                elif a_RAE[lplb_RAE] == 3:
                    for k in range(1, y_RAE):
                        if a_RAE[lplb_RAE + k] < 1:
                            break
                        else:
                            a_RAE[lplb_RAE + k] = max([2, a_RAE[lplb_RAE + k]])
                    lplb_RAE += k
                    no_jump = 0
                else:
                    no_jump = 1

            # Get the current classifier model, and do the prediction on the current recording
            mdl_RAE = clf_RAE
            Z_RAE = mdl_RAE.predict(listtranspose_J(online_fea_RAE[isubject]))
            Z_RAE_matlab = copy.deepcopy(Z_RAE)
            # Convert the fatigue Label from [0,1,2,3] to [0,1] scale
            newlabel_RAE = np.zeros(a_RAE.shape)
            for isample in range(len(label_RAE[isubject][0])):
                if a_RAE[isample] > 1:
                    newlabel_RAE[isample] = 1
            newlabel_RAE_matlab = copy.deepcopy(newlabel_RAE)

            # noise frame dilation.
            noise_serial_temp = noiseFrameDilation(noise_label_RAE[isubject][0])
            if FLAG['noise_frame_dilation'] == 1:
                for ino in range(len(noise_serial_temp)):
                    noise_label_RAE[isubject][0][ino] = noise_serial_temp[ino]
            if delete_noise_RAE:
                # Porcessing on all the noise frames
                target_noise_index = np.nonzero(np.logical_or((noise_label_RAE[isubject][0] > 0), (noise_label_RAE[isubject][1] == 0)))[0]
                raw_a_RAE = np.delete(raw_a_RAE, target_noise_index, axis=0)
                Z_RAE = np.delete(Z_RAE, target_noise_index, axis=0)
                newlabel_RAE = np.delete(newlabel_RAE, target_noise_index, axis=0)

            # Creating the fatigure period data matrix. Each row is corresponding to one fatigure period. Colomn 0 is the start time, while 1 is the end time for this period
            fatiperiod_temp = []
            for isample in range(len(Z_RAE) - 1):
                if newlabel_RAE[isample + 1] == 1 and newlabel_RAE[isample] == 0:
                    fatiperiod_temp.append(isample + 1)
                if newlabel_RAE[isample + 1] == 0 and newlabel_RAE[isample] == 1:
                    fatiperiod_temp.append(isample)
            if newlabel_RAE[-1] == 1 and newlabel_RAE[-2] == 1:
                fatiperiod_temp.append(len(Z_RAE) - 1)
            fatiperiod_temp = np.array(fatiperiod_temp)
            fatiperiod = np.concatenate((fatiperiod_temp[0::2].reshape(-1, 1), fatiperiod_temp[1::2].reshape(-1, 1)), axis=1)

            # Get the number of fatigure period in current recording
            N_fati = len(fatiperiod)

            # Seperate Recordings in different groups.
            if len(np.nonzero(a_RAE == 3)[0]) > 0:
                fullfati_group.append(isubject)
                target_group.append(isubject)
            else:
                if len(np.nonzero(a_RAE == 2)[0]) > 0:
                    midlvl_group.append(isubject)
                    if len(np.nonzero(a_RAE == 2)[0]) > 10:
                        target_group.append(isubject)
                else:
                    nofati_group.append(isubject)
            all_group.append(isubject)
            # Get the current FRAP between the classifier results and the newlabel
            FRAP_RAE = calculateFRAP(Z_RAE, newlabel_RAE, 0)
            # print(FRAP_RAE)
            FA_RAE = np.around(FRAP_RAE[4], decimals=4)
            if FA_RAE >= 0:
                FA_RAE_temp.append(FA_RAE)

            Nnega_RAE = len(np.nonzero(Z_RAE == 1)[0])
            Nnega_RAE_temp.append(Nnega_RAE)

            FP_RAE_flagindex = np.logical_and((Z_RAE == 1), (newlabel_RAE == 0))
            FP_RAE = len(np.nonzero(FP_RAE_flagindex)[0])
            # FP_RAE = len(np.nonzero(np.logical_and((Z_RAE == 1), (raw_a_RAE < 2)))[0])
            FP_RAE_temp.append(FP_RAE)
            np.logical_and
            FPL1_RAE = len(np.nonzero(np.logical_and(FP_RAE_flagindex, (raw_a_RAE == 1)))[0])
            FPL1_RAE_temp.append(FPL1_RAE)
            FPL0_RAE = len(np.nonzero(np.logical_and(FP_RAE_flagindex, (raw_a_RAE == 0)))[0])
            FPL0_RAE_temp.append(FPL0_RAE)
            if Nnega_RAE > 0:
                FAL1_RAE = np.round(FPL1_RAE / Nnega_RAE, decimals=4)
                FAL0_RAE = np.round(FPL0_RAE / Nnega_RAE, decimals=4)
            else:
                FAL1_RAE = -1
                FAL0_RAE = -1
            # print(Nnega_RAE, FP_RAE)
            # When there's no fatigue period, jump the fati period Analysis
            # if N_fati == 0:
            #     continue

            # [Fatigure Period Analysis ]
            M_fati = 0
            #   Detecting the first Classifier Alert in each single Fatigue Period
            for nfati in range(N_fati):
                fati_st = fatiperiod[nfati, 0]
                fati_ed = fatiperiod[nfati, 1]
                flag_fati = 0
                for ifati in range(fati_st, fati_ed + 6):
                    if Z_RAE[ifati] == 1:
                        flag_fati = 1
                        M_fati += 1
                        RAT_temp.append(ifati - fati_st)
                        break
            N_P3 = len(np.nonzero(raw_a_RAE == 3)[0])
            if N_P3 > 0:
                P3 = np.around(len(np.nonzero(np.logical_and((Z_RAE == 1), (raw_a_RAE == 3)))[0]) / N_P3, decimals=4)
            else:
                P3 = -1

            undetect_Fati_temp.append(N_fati - M_fati)
            N_fati_temp.append(N_fati)
            # Show some results.
            if N_fati != 0:
                detect_portion = np.around(M_fati / N_fati * 100, decimals=2)
                if detect_portion < 75 and N_fati - M_fati > 5:
                    satisflag = 0
                else:
                    if FPL0_RAE > 30:
                        satisflag = 0
                    else:
                        if (FAL1_RAE / FA_RAE) < 0.6:
                            satisflag = 0
                        else:
                            satisflag = 1
                if satisflag == 1:
                    posigroup += 1
                    satisgroup += 1
                else:
                    unsatisgroup += 1
                posiamount += 1
            else:
                detect_portion = 0
                if FPL0_RAE < 10:
                    satisflag = 1
                else:
                    if FAL1_RAE > 0.75:
                        satisflag = 1
                    else:
                        satisflag = 0
                if satisflag == 1:
                    negagroup += 1
                    satisgroup += 1
                else:
                    unsatisgroup += 1
                negaamount += 1
            if satisflag == 1:
                satisstring = '[O]'
            else:
                satisstring = '[X]'

            M_fati_all.append(M_fati)
            N_fati_all.append(N_fati)
            if isubject in all_group:
                print('--' * 50)
                print('Group No. ' + str(isubject))
                print('M:[' + str(M_fati) +
                      '] |N:[' + str(N_fati) +
                      '] |Por:' + str(detect_portion) +
                      '% |P3:' + str(P3) + '[' + str(N_P3) + ']' +
                      ' |FA:' + str(FA_RAE) +
                      ' |FAL1:' + str(FAL1_RAE) + '[' + str(FPL1_RAE) + ']' +
                      ' |FAL0:' + str(FAL0_RAE) + '[' + str(FPL0_RAE) + ']' +
                      ' ' * 12 + satisstring)
                titletext = ['M:[' + str(M_fati) +
                             '] |N:[' + str(N_fati) +
                             '] |Por:' + str(detect_portion) +
                             '% |P3:' + str(P3) + '[' + str(N_P3) + ']' +
                             ' |FA:' + str(FA_RAE) +
                             ' |FAL1:' + str(FAL1_RAE) + '[' + str(FPL1_RAE) + ']' +
                             ' |FAL0:' + str(FAL0_RAE) + '[' + str(FPL0_RAE) + ']' +
                             '   ||  EXAMINATION RESULT: ' + satisstring]
            general_output = [M_fati, N_fati, detect_portion, P3, N_P3, FA_RAE, FAL1_RAE, FAL0_RAE]
            general_output_name = ['M', 'N', 'Portion', 'P3', 'NP3', 'FA', 'FAL1', 'FAL0']
            # Saved for matlab programming.
            # [raw_a_RAE] raw label [0,1,2,3]
            # [a_RAE] modified label with new rule [0,1,2,3]
            # [newlabel_RAE] convert into [0,1]  0 from old 0,1; 1 from old 2,3
            # [Z_RAE] classification result [0,1]
            # np.savetxt(path['fig_path'] + 'FatiIdRef/' + testdata_file[testlist_RAE[isubject]] + '_FatiId_REF.txt', Z_RAE)
            # np.savetxt(path['fig_path'] + 'FatiIdRef/' + testdata_file[testlist_RAE[isubject]] + '_FatiId_REF2.txt', raw_a_RAE)
            sio.savemat(path['fig_path'] + '/matinterface/labelinfo' + str(isubject) + '.mat',
                        {'rawlabel': raw_a_RAE_matlab,
                         'a': a_RAE,
                         'nlabel': newlabel_RAE_matlab,
                         'Z': Z_RAE_matlab,
                         'subject': testdata_file[testlist_RAE[isubject]],
                         'fatiperiod': fatiperiod,
                         'fea1': online_fea_RAE[isubject][N_f1],
                         'fea1name': name_f1,
                         'fea2': online_fea_RAE[isubject][N_f2],
                         'fea2name': name_f2,
                         'noiselabel': noise_label_RAE[isubject],
                         'noiseserial': noise_serial_temp,
                         'feas': online_fea_RAE[isubject][18],
                         'generaloutput': general_output,
                         'generaloutputname': general_output_name,
                         'titletext': titletext})
        # Loop end
        print('==' * 5 + '[No. ' + str(i) + 'Loop GENERAL RESULT ]' + '=' * 35)
        detect_portion_all = np.around(sum(M_fati_all) / sum(N_fati_all) * 100, decimals=2)
        meanFA = np.around(np.mean(FA_RAE_temp), decimals=4)
        if sum(Nnega_RAE_temp) > 0:
            overallFA = np.around(sum(FP_RAE_temp) / sum(Nnega_RAE_temp), decimals=4)
            overallFAL1 = np.around(sum(FPL1_RAE_temp) / sum(Nnega_RAE_temp), decimals=4)
            overallFAL0 = np.around(sum(FPL0_RAE_temp) / sum(Nnega_RAE_temp), decimals=4)
        else:
            overallFA = -1
            overallFAL1 = -1
            overallFAL0 = -1
        print('M_ALL: [' + str(sum(M_fati_all)) +
              '] | N_ALL: [' + str(sum(N_fati_all)) +
              '] | Portion: ' + str(detect_portion_all) + '%')

        print('Mean FA: ' + str(meanFA) +
              ' | Overall FA: ' + str(overallFA) +
              ' | Overall FAL1: ' + str(overallFAL1) +
              ' | Overall FAL0: ' + str(overallFAL0))
        RAT_mean = np.around(np.mean(RAT_temp), decimals=3)
        RAT_median = np.around(np.median(RAT_temp), decimals=3)
        print('RAT Mean: ' + str(RAT_mean * 10) + 's | RAT Median: ' + str(RAT_median * 10) + ' | Data Size: [' + str(len(RAT_temp)) + ']')

        h = plt.figure()
        plt.plot(sorted(RAT_temp))
        plt.show()
        plt.savefig((path['fig_path'] + '/RAT/RAT' + str(i) + '.png'))

        # loopX = GVal.getPARA('')

        #   Overall Analysis
        Portion_Overall.append(detect_portion_all)
        FA_Overall.append(meanFA)
        RAT_Overallmean.append(RAT_mean)
        RAT_Overallmedian.append(RAT_median)
        FS1 = (detect_portion_all + 100 * (1 - meanFA)) / 2
        FS2 = 1 / (1 / detect_portion_all + 1 / (100 * (1 - meanFA)))
        FS3 = math.sqrt(detect_portion_all * 100 * (1 - meanFA))
        FINALSCORE1.append(FS1)
        FINALSCORE2.append(FS2)
        FINALSCORE3.append(FS3)
        print('Satisfied Group Number: [ ' + str(satisgroup) + ' / ' + str(satisgroup + unsatisgroup) + ']' + 'Wtih [ Class +: (' + str(posigroup) + '/' + str(posiamount) + ') | Class -: (' + str(negagroup) + '/' + str(negaamount) + ')]')
        print('FINAL SCORE: ' + str(FS1) + ' | ' + str(FS2) + ' | ' + str(FS3))
        Nundetect = 10
        print('-' * 10 + ' Worst top ' + str(Nundetect) + ' Recording ' + '-' * 40)
        undetect_Fati_value = np.sort(undetect_Fati_temp)
        undetect_Fati_index = np.argsort(undetect_Fati_temp)
        for iNu in range(1, Nundetect + 1):
            undetect = undetect_Fati_value[-iNu]
            undetect_index = undetect_Fati_index[-iNu]
            print('Top ' + str(iNu) + ' Worst Recording: [' + testdata_file[testlist_RAE[undetect_index]] + '(No.' + str(undetect_index) + ')] with [' + str(undetect) + ' of ' + str(N_fati_temp[undetect_index]) + '] Undetected Fatigue Period.')
    h = plt.figure()
    plt.plot(Portion_Overall)
    plt.show()
    plt.savefig((path['fig_path'] + '/CLFA/Portion_Overall' + str(i) + '.png'))

    h = plt.figure()
    plt.plot(FA_Overall)
    plt.show()
    plt.savefig((path['fig_path'] + '/CLFA/FA_Overall' + str(i) + '.png'))

    h = plt.figure()
    plt.plot(RAT_Overallmean)
    plt.show()
    plt.savefig((path['fig_path'] + '/CLFA/RAT_Overallmean' + str(i) + '.png'))

    h = plt.figure()
    plt.plot(RAT_Overallmedian)
    plt.show()
    plt.savefig((path['fig_path'] + '/CLFA/RAT_Overallmedian' + str(i) + '.png'))

    h = plt.figure()
    plt.plot(FINALSCORE1)
    plt.plot(FINALSCORE2)
    plt.plot(FINALSCORE3)
    plt.show()
    plt.savefig((path['fig_path'] + '/CLFA/FINALSCORE' + str(i) + '.png'))

    save_classifier = open(path['fig_path'] + '/CLFA/clf' + time.strftime('%Y%m%d%Hh%M', time.localtime(time.time())) + '.pickle', 'wb')
    pickle.dump(mdl_RAE, save_classifier)
    save_classifier.close()
