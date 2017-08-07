# -*- coding:utf-8 -*-

import numpy as np
import GVal
from toolkitJ import cell2dmatlab_jsp


def initializationProcess():
    localprefix = '/home/'
    serverprefix = '/home/labcompute/'
    GVal.setPARA('prefix_PARA', serverprefix)
    # GVal.setPARA('prefix_PARA', localprefix)

    return GVal.getPARA('prefix_PARA')


def controlPanel():

    ########################################
    ############## [ Control panel ] #############
    ########################################

    #### [ Loop Parameters ] ####################
    GVal.setPARA(
        'recording_index_list_PARA',
        [
            # np.array([21])
            # np.arange(1, 137, 1),
            np.arange(138, 234, 1)

        ])
    ########################################
    # select_index = np.arange(10, 20)

    # Label Reading
    # select_index
    # 0 - 99 Lab Environment (High SNR)
    # 100 - 137  Other Environment (High SNR)
    # 138 - 234 Real Industrial Environment  (Low SNR)

    #### [ Static Parameters ] ####################
    GVal.setPARA(
        'classifier_list_PARA',
        [
            # 21,
            # 24,
            # 25,
            # 30,
            # 23,
            32

        ])
    ########################################
    # 2,3 - SciKitLearn,  1- Keras (Nerual Network),   4- Other

    # classifier_num

    # Sub Classifier cateloge.
    # KT Frame
    # 11- Sequenial NN

    # SKL Frame
    # 21 - SVM Linear(Default for SciKitLearn)
    # 22 - SVM Kernal
    # 221 - SVM RBF Kernal(Default if only 22 is input)
    # 222 - SVM Poly Kernal
    # 223 - SVM sigmoid Kernal
    # 23 - LDA (Linear Discriminant Analysis)
    # 24 - QDA (Quadratic Discriminant Analysis)
    # 25 - Naive Bayes (Default Gaussian)
    # 251 - Naive Bayes Gaussian
    # 252 - Naive Bayes Multinominal
    # 253 - Naive Bayes Bernoulli
    # 26 - Neural Network ( by sklearn frame)
    # 27 - Adaboost(Default for Decision Tree base)
    # 271 - Adaboost with Decision Tree Weak Classifier
    # 28 - Linear Regression Weighted [N/A]
    # 29 - SGD Classifier
    # 30 - Logistic Regression
    # 31 - Decision Tree
    # 32 - Random Forest
    ########################################

    ########################################

    GVal.initLoopPARA('random_seed_PARA', np.arange(20, 22, 1))
    GVal.initLoopPARA('kick_off_no1_PARA', np.arange(8))
    GVal.initLoopPARA('noconfident_frame_process_PARA',  np.arange(6))
    # Flags
    FLAG = {}
    FLAG['data_prepare_flag'] = 0
    FLAG['plotting_flag'] = 0
    FLAG['save_flag'] = 0

    # online_fea_name = [['眨眼时间'], ['眨眼速度'], ['perclos'], ['sem'], ['sem重心频率'], ['标准差'], ['眨眼数'], ['sem合成特征'], ['眨眼合成特征'], ['综合特征']]

    GVal.setPARA('weights_on_PARA', 0)
    GVal.setPARA('beta_PARA', 1.5)
    #########################################
    GVal.setPARA('kick_off_no1_PARA', 7)
    # 0 - Train: [del] || Validation: [del]
    # 1 - Train: [del] || Validation: [0]
    # 2 - Train: [del] || Validation: [1]
    # 3 - Train: [1] || Validation: [del]
    # 4 - Train: [1] || Validation: [0]
    # 5 - Train: [1] || Validation: [1]
    # 6 - Train: [0] || Validation: [del]
    # 7 - Train: [0] || Validation: [0]
    # 8 - Train: [0] || Validation: [1]

    #########################################
    FLAG['label_process_flag'] = 1
    GVal.setPARA('noconfident_frame_process_PARA', 2)
    # 0 - Remain the current label (Do nothing)
    # 1 - Set all not confident frame as label 7 and remain all
    # 2 - Set all not confident frame as label 0 and remain all
    # 3 - Delete not confident frame with label 7
    # 4 - Delete not confident frame with label 0
    # 5 - Delete all the not confident frame
    GVal.setPARA('noise_frame_process_PARA', 1)
    # 0 - Remain all the noise frame (Do nothing)
    # 1 - Simply Delete the noise frame and remain the noblink frame
    # 2 - Delete the noise frame and the noblink frame
    #########################################

    # Weight_list, the key is referring the label num (0 -nofatigue, 1 -7gradefatigue ... )
    # And the corresponding value for each key is the weight for this certain class
    GVal.setPARA(
        'weight_list_PARA',  {
            0: 0.8,
            1: 1,
            2: 2,
            3: 3
        })

    #########################################
    # 1:  caseLOO, nonparameters
    # 2:  caseLPO, parameter P (positive integer)
    # 3:  caseKFold, parameter K fold  (positive integer)
    # 4:  caseStratifiedKFold, parameter K fold  (positive integer)
    # 5:  caseRawsplit, parameter test_proportion  (range [ 0 : 1])
    GVal.setPARA('split_type_num', 5)
    GVal.setPARA('split_type_para', 0.3)
    GVal.setPARA('random_seed_PARA', 2)

    #########################################
    FLAG['fea_selection_flag'] = 0
    # 1 - tSNE(Default with PCA)
    # 2 - normalPCA
    GVal.setPARA('feaSelectionStrategy', 2)
    GVal.setPARA('nComponent_PARA', 10)

    #########################################
    FLAG['data_balance_flag'] = 1
    # 1 - Down sampling the negative class (para is the [0,1] float, indicating the percentage of retain)
    GVal.setPARA('data_balance_process_PARA', 1)
    GVal.setPARA('data_balance_para_PARA', 0.25)

    #########################################
    # Select several features from the online feature file
    # OLD feature Info
    # GVal.setPARA('online_fea_selectindex_PARA', np.array([0, 1, 2, 3, 4, 14, 15, 11, 12, 13]))
    # GVal.setPARA('online_fea_name_PARA', [['眨眼时间'], ['眨眼速度'], ['perclos'], ['sem'], ['sem重心频率'], ['标准差'], ['眨眼数'], ['sem合成特征'], ['眨眼合成特征'], ['综合特征']])

    # NEW feature Info
    GVal.setPARA('online_fea_selectindex_PARA', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 22, 23, 24, 25, 26]))
    # GVal.setPARA('online_fea_selectindex_PARA', np.array([22, 23, 24]))
    GVal.setPARA('feature_index_PARA', np.arange(0, len(GVal.getPARA('online_fea_selectindex_PARA'))))

    # 0 -- 12 维 对应的分别是：眨眼时间、眨眼速度、perclos_first、闭眼时间、完全闭合时间、
    #                                           睁眼时间、闭眼平均速度、睁眼平均速度、闭眼最大速度、睁眼最大速度、
    #                                           perclos_second、perclos_third、perclos_fourth
    #                                           perclos_first=完全闭合时间/闭眼时间
    #                                           perclos_second=完全闭合时间/眨眼时间
    #                                           perclos_third=(open_st-close_st)/(open_en-close_st)
    #                                           perclos_fourth=(close_en-close_st)/(open_en-close_st)
    # 13 -- 15 维 对应的分别是： sem、sem重心频率、快速眼电相对功率(1.2Hz-3Hz/0.3Hz-4.8Hz)
    # 16 维 对应的分别是：状态指数
    # 18 维 对应的分别是：眨眼极值差的均值
    # 19 维  噪声类型的标记，0--无噪声，
    #                                         1--25Hz噪声，
    #                                         2--低频晃动噪声，
    #                                         3--空载(无明显中频脑电)，
    #                                         4--咀嚼噪声,
    #                                         5--信号长度不足8秒，
    #                                         6--空载(标准差过小4) ,
    #                                         7--严重漂移
    # 22 -- 24 维 对应的分别是：sem合成特征、眨眼合成特征、综合特征
    # 25 -- 26 维 对应的分别是：标准差，眨眼数
    ### [ Essential Initialization Parameters ] #################################
    GVal.setPARA('noise_label_index_PARA', np.array([19, 26]))
    GVal.setPARA('firstTimeConstruction', FLAG['data_prepare_flag'])

    return FLAG, processCodeEncoder()


def processCodeEncoder():
    recording_index_list = GVal.getPARA('recording_index_list_PARA')
    classifier_list = GVal.getPARA('classifier_list_PARA')

    loopPARA_cache = GVal.getLoopPARA_cache()[0]

    loopPARA_amount = len(loopPARA_cache)
    GVal.setPARA('loopPARA_amount_PARA', loopPARA_amount)
    loopPARA_totalsize = 1
    loopPARA_bitsize = np.zeros([loopPARA_amount, 1])
    loopPARA_namecache = cell2dmatlab_jsp([loopPARA_amount, 1], 2, [])

    j = 0
    for loopPARA_name in loopPARA_cache.keys():
        loopPARA_totalsize *= loopPARA_cache[loopPARA_name][1]
        loopPARA_bitsize[j] = loopPARA_cache[loopPARA_name][1]
        loopPARA_namecache[j][0] = loopPARA_name
        j += 1

    # Important processCode Para 1.
    GVal.setPARA('loopPARA_namecache_PARA', loopPARA_namecache)
    loopPARA_codepool = cell2dmatlab_jsp([loopPARA_totalsize, loopPARA_amount], 2, [])

    k = cell2dmatlab_jsp([loopPARA_amount], 1, [0])

    for i in range(loopPARA_totalsize):
        for jj in range(loopPARA_amount):
            loopPARA_codepool[i][jj] = loopPARA_cache[loopPARA_namecache[jj][0]][0][k[jj]][0]
            if k[jj] < loopPARA_bitsize[jj] - 1:
                k[jj][0] += 1
            else:
                k[jj][0] = 0

    # Important processCode Para 2.
    GVal.setPARA('loopPARA_codepool_PARA', loopPARA_codepool)
    print('### Loop Para Pool Size: [' + str(loopPARA_totalsize) + ', ' + str(loopPARA_amount) + '] ([Total Loop,Loop Parameter amount])')

    process_code_pack = []
    for recording_index_list_lplb in np.arange(len(recording_index_list)):
        for classifier_list_lplb in np.arange(len(classifier_list)):
            for loopPARA_serialnum in range(loopPARA_totalsize):
                code_temp = int(1e0 * classifier_list[classifier_list_lplb] +
                                1e3 * (recording_index_list_lplb + 1) +
                                1e5 * loopPARA_amount +
                                1e7 * loopPARA_serialnum +
                                1e11 * 1)
                process_code_pack.append(code_temp)

    return process_code_pack


def processCodeDecoder(process_code):
    print('### Input process_code is: [' + str(process_code) + ']')
    process_code_str = str(process_code)
    classifier_num_temp = int(process_code_str[-3:])
    recording_index_list_selectserial = int(process_code_str[-5: -3]) - 1
    recording_index_list = GVal.getPARA('recording_index_list_PARA')

    loopPARA_amount = GVal.getPARA('loopPARA_amount_PARA')
    loopPARA_codepool = GVal.getPARA('loopPARA_codepool_PARA')
    loopPARA_namecache = GVal.getPARA('loopPARA_namecache_PARA')

    loopPARA_serialnum = int(process_code_str[-11:-7])
    for loopPARA_num in range(loopPARA_amount):
        print('### Looping Parameter (' + str(loopPARA_num) + '/' + str(loopPARA_amount) + '): [' + loopPARA_namecache[loopPARA_num][0] + '], with value set: [' + str(loopPARA_codepool[loopPARA_serialnum][loopPARA_num]) + ']')
        GVal.setPARA(loopPARA_namecache[loopPARA_num][0], loopPARA_codepool[loopPARA_serialnum][loopPARA_num])

    select_index_temp = recording_index_list[recording_index_list_selectserial]

    GVal.setPARA('classifier_num_PARA', classifier_num_temp)
    GVal.setPARA('select_index_PARA', select_index_temp)
    print('######## [ Decoder Success! ] ########')

    return select_index_temp, classifier_num_temp
