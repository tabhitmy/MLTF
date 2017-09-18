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
from toolkitJ import Logger_J

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

        lplb_count1 += 1
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
            X_train_raw, y_train = labelProcessor(X_train_rawraw[N_lplb_count], y_train_raw[N_lplb_count])
        else:
            X_train_raw = copy.deepcopy(X_train_rawraw[N_lplb_count])
            y_train = copy.deepcopy(y_train_raw[N_lplb_count])

        # # Feature Plotting [N/A]
        # if FLAG['plotting_flag']:
        #     featurePlotting(Y, X_raw, process_code)
        # else:
        #     print('### No plotting, exit feature plotting...')
        ####################################
        # [TODO IN PROGRESS]  Add the feature selection Method
        if FLAG['fea_selection_flag']:
            # feaSelection is doing the feature selection only on trainning set, return the fea selection result
            X_train, feaS_mdl = feaSelection(X_train_raw)
            # According the fea selection result, preprocess  validation set in corresponding to the  trainning set.
            X_validation = feaS_mdl.fit_transform(X_validation_raw[N_lplb_count])
        else:
            X_train = copy.deepcopy(X_train_raw)
            X_validation = copy.deepcopy(X_validation_raw[N_lplb_count])

        ####################################
        # [TODO] Feature Regulation
        # Three types of features:
        #     1 -- Continuous
        #     2 -- Discrete (Categorical)
        #     3 -- Discrete Binary (0 and 1)

        y_validation = copy.deepcopy(y_validation_raw[N_lplb_count])

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
        # infoserial :  1- model object   2- Training Score  3- FRAP
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

        if FLAG['plotting_flag']:
            FRAP = res_temp[0][3]
            # resultPlotting(FRAP, process_code)
        else:
            print('### No plotting, exit feature plotting...')

        time.sleep(0.001)

    return res, N


if __name__ == "__main__":
    start = clock()

    # Gain the path prefix, this is added to suit up different processing environment.
    path_prefix, username = initializationProcess()

    sys.stdout = Logger_J(path_prefix + 'GaoMY/tlog.txt')

    print('### Log starting!')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # Define the basic path.
    path = {
        'data_path': (path_prefix + 'public/labcompute/BK_OLD_EEG_RawData/'),
        'label_path': (path_prefix + 'public/backend_data/Label/'),
        'online_fea_path': (path_prefix + 'public/backend_data/online_new/'),
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
    data_file = sorted(os.listdir(path['data_path']))
    # Processing the Control Panel! All the important configuration are pre-defined in control panel.
    # Refer the file controlPanel_NFDA_J.py for more details.
    # [FLAG] is a flag controller, to switch on/off several processing procedure.
    # [process_code_pack] is a package containing all the targeting process loops, each loop has an independent process_code.

    FLAG, process_code_pack = controlPanel(username)

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
               '] | [Classifier: ' + res[i][1] + ' ' * (17 - len(res[i][1])) +
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
