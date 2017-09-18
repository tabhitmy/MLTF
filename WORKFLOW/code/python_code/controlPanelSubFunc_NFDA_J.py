import os
import numpy as np
import GVal
import copy
from toolkitJ import cell2dmatlab_jsp
from toolkitJ import str2num
# default input value matrix
# For inner-classifier parameter loop. Here contains default loop
# Specific for loop parameter setting in controlPanel()
dVM = {}


# 21 -- Linear SVM
dVM[2100] = ['penalty', ['l1', 'l2'], 'l2']
dVM[2101] = ['loss', ['hinge', 'squared_hinge'], 'squared_hinge']
dVM[2102] = ['dual', [True, False], False]  # dual classification or primal classification
dVM[2103] = ['tol', np.arange(5e-5, 15e-5, 1e-5), 1e-4]  # tolerance for the stopping criteria
dVM[2104] = ['C', np.arange(0.1, 2, 0.1), 1]  # Penalty parameter C of the error term
dVM[2105] = ['multi_class']  # n/a
dVM[2106] = ['fit_intercept']  # n/a
dVM[2107] = ['intercept_scaling']  # n/a
dVM[2108] = ['class_weight']  # n/a
dVM[2109] = ['verbose']
dVM[2110] = ['random_state', np.arange(100, 120, 2), 115]  # n/a
dVM[2111] = ['max_iter']  # n/a


# 23 -- LDA
dVM[2300] = ['solver', ['svd', 'lsqr', 'eigen'], 'svd']
dVM[2301] = ['shrinkage']  # n/a only work with lsqr or eigen in 2300
dVM[2302] = ['priors']  # n/a
dVM[2303] = ['n_components', np.arange(10), 2]  # number of components for dimensionality reduction
dVM[2304] = ['store_covariance']  # n/a for svd only
dVM[2305] = ['tol', np.arange(5e-5, 15e-5, 1e-5), 1e-4]  # tolerance for the stopping criteria


# 25 -- Naive Bayes Guassian  (There's no parameter for NB Gaussian.)

# 27 -- Adaboost
dVM[2700] = ['n_estimators', np.arange(5, 105, 10), 50]
dVM[2701] = ['learning_rate', np.arange(0.1, 2, 0.1), 1]
dVM[2702] = ['algorithm', ['SAMME', 'SAMME.R'], 'SAMME.R']
dVM[2703] = ['random_state', np.arange(100, 120, 2), 115]


# 30 -- Logistic Regression Classifier LRC
dVM[3000] = ['penalty', ['l1', 'l2'], 'l2']
dVM[3001] = ['dual', [True, False], False]
dVM[3002] = ['tol', np.arange(5e-5, 15e-5, 1e-5), 1e-4]  # tolerance for the stopping criteria
dVM[3003] = ['C', np.arange(0.1, 2, 0.1), 1]  # Penalty parameter C of the error term
dVM[3004] = ['fit_intercept']  # n/a
dVM[3005] = ['intercept_scaling']  # n/a
dVM[3006] = ['class_weight']  # n/a
dVM[3007] = ['random_state', np.arange(100, 120, 2), 115]  # n/a
dVM[3008] = ['solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'liblinear']
dVM[3009] = ['max_iter', np.arange(10, 200, 20), 100]
dVM[3010] = ['verbose']
dVM[3011] = ['warm_start']
dVM[3012] = ['n_jobs']

# 31 -- Decision Tree
dVM[3100] = ['criterion', ['gini', 'entropy'], 'gini']
dVM[3101] = ['splitter', ['best', 'random'], 'best']
dVM[3102] = ['max_depth', np.arange(1, 10, 1), 5]
dVM[3103] = ['min_samples_split', np.arange(2, 100, 10), 2]
dVM[3104] = ['min_samples_leaf', np.arange(1, 5, 1), 1]
dVM[3105] = ['min_weight_fraction_leaf', np.arange(0, 0.5, 0.1), 0]  # n/a
dVM[3106] = ['max_features', np.arange(1, 10, 1), 2]  # other possible  'sqrt','log2'
dVM[3107] = ['random_state', np.arange(100, 120, 2), 115]
dVM[3108] = ['max_leaf_nodes']  # n/a
dVM[3109] = ['min_impurity_decrease']  # n/a
dVM[3110] = ['min_impurity_split']  # n/a
dVM[3111] = ['class_weight']  # n/a
dVM[3112] = ['presort']  # n/a


#  32 -- Random Forest
dVM[3200] = ['n_estimators', np.arange(5, 15, 1), 10]
dVM[3201] = ['criterion', ['gini', 'entropy'], 'gini']
dVM[3202] = ['max_features', np.arange(2, 10, 1), 2]  # other possible  'sqrt','log2'
dVM[3203] = ['max_depth', np.arange(2, 10, 1), 5]
dVM[3204] = ['min_samples_split', np.arange(2, 100, 10), 2]
dVM[3205] = ['min_samples_leaf', np.arange(1, 5, 1), 1]
dVM[3206] = ['min_weight_fraction_leaf', np.arange(0, 0.5, 0.1), 0]
dVM[3207] = ['max_leaf_nodes']  # n/a
dVM[3208] = ['min_impurity_decrease']  # n/a
dVM[3209] = ['bootstrap', [False, True], True]  # n/a
dVM[3211] = ['oob_score']  # n/a
dVM[3212] = ['n_jobs']  # n/a
dVM[3213] = ['random_state', np.arange(100, 120, 2), 115]  # n/a
dVM[3214] = ['verbose']  # n/a
dVM[3215] = ['warm_start']  # n/a
dVM[3216] = ['class_weight']  # n/a


#  33 -- bagging
dVM[3300] = ['n_estimators', np.arange(5, 15, 1), 10]
dVM[3301] = ['max_samples', np.arange(0.3, 1.0, 0.1), 1.0]
dVM[3302] = ['max_features', np.arange(0.3, 1.0, 0.1), 1.0]
dVM[3303] = ['bootstrap', [True, False], True]
dVM[3304] = ['boostrap_features']  # n/a
dVM[3305] = ['oob_score']  # n/a
dVM[3306] = ['warm_start']  # n/a
dVM[3307] = ['n_jobs']  # n/a
dVM[3308] = ['random_state', np.arange(100, 120, 2), 115]
dVM[3309] = ['warm_start']  # n/a


# 34 -- voting
dVM[3400] = ['estimators', [21, 23, 25, 30, 31], [30, 21, 23, 25, 30, 31]]  # no loop
dVM[3401] = ['voting', ['hard', 'soft'], 'hard']


# 35 -- gradient boosting
dVM[3500] = ['loss', ['deviance', 'exponential'], 'deviance']
dVM[3501] = ['learning_rate', np.arange(0.1, 2, 0.1), 1]
dVM[3502] = ['n_estimators', np.arange(5, 105, 10), 50]
dVM[3503] = ['max_depth', np.arange(2, 5, 1), 3]
dVM[3504] = ['criterion', ['friedman_mse', 'mse', 'mae'], 'friedman_mse']
dVM[3505] = ['min_samples_split', np.arange(2, 100, 10), 2]
dVM[3506] = ['min_samples_leaf', np.arange(1, 5, 1), 1]
dVM[3507] = ['min_weight_fraction_leaf', np.arange(0, 0.5, 0.1), 0]  # n/a
dVM[3508] = ['subsample', np.arange(0.5, 1, 0.1), 1]
dVM[3509] = ['max_features', np.arange(2, 10, 1), 2]  # N/A  # other possible  'sqrt','log2'
dVM[3510] = ['max_leaf_nodes']  # n/a
dVM[3511] = ['min_impurity_decrease']  # n/a
dVM[3512] = ['init']  # n/a
dVM[3513] = ['verbose']  # n/a
dVM[3514] = ['warm_start']  # n/a
dVM[3515] = ['random_state', np.arange(100, 120, 2), 115]  # n/a
dVM[3516] = ['presort']

GVal.setPARA('dVM_PARA', dVM)


def controlPanel_admin(username):
    # Control Panel for the whole system. It contains all the essential parameter setting before the execution.

    ###########################################
    #### [ Loop Parameters #1] #####################
    # Loop Parameters 1 is the different recording groups.
    # It is tuned by only the serial number below:
    # 0 - 99 Lab Environment (High SNR)
    # 100 - 137  Other Environment (High SNR)
    # 138 - 234 Real Industrial Environment  (Low SNR)

    GVal.setPARA(
        'recording_index_list_PARA',
        [
            # np.array([21])
            np.arange(1, 137, 1),
            # np.arange(138, 234, 1),
            # np.arange(1, 7, 1)
            # np.arange(147, 155, 1)

        ])
    # Temporial Code later will be deprecated.
    if 136 in GVal.getPARA('recording_index_list_PARA')[0]:
        GVal.setPARA('recordname', 'Experiment')
    else:
        GVal.setPARA('recordname', 'Realistic')

    ###########################################

    ###########################################
    #### [ Loop Parameters #2] #####################
    # Loop parameter #2 are the different classifiers.
    # In this system, each classifier (no matter in which frame, SKL or KT) has its own classifier serial number. List below:
    # Choose 1 or more together for loop. and then compare among your choices.

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
    # 33 - Bagging with DT
    GVal.setPARA(
        'classifier_list_PARA',
        [
            # 21,
            # 24,
            # 25,
            # 30,
            # 27
            # 32,
            31
            # 31
            # 23
            # 33,
            # 34,
            # 35
        ])

    ###########################################

    ###########################################
    ### [ Loop Parameters, others ] ##################
    # Add any para you want to loop. You can loop many parameters together.
    # The loop mechanism is boosted by GVal.initLoopPARA, add the exact name of any parameters in this control panel, then set a coresponding array, in which all the values will be import as looping parameters.

    # iter , plain
    GVal.setPARA('codepoolCreator', 'plain')

    # GVal.initLoopPARA('noconfident_frame_process_PARA',  np.arange(6))
    GVal.initLoopPARA('data_balance_para_PARA', np.arange(0.01, 0.99, 0.04))
    # GVal.initLoopPARA('split_type_para', np.arange(0.01,0.99,0.04))


    GVal.initLoopPARA('kick_off_no1_PARA', [0,1,5,7])
    ###########################################
    ### [Loop Parameter, Classifier inner ] #############
    # The loop parameter for the classifier inner parameter must be restricted when only 1 classifier in loop
    # The default Inputvalue matrix [dIM] is set in GVal.py, please refer  the GVal.py for more information
    if len(GVal.getPARA('classifier_list_PARA')) < 2:
        GVal.initLoopPARA(0000, [])
        # GVal.initLoopPARA(3204, np.arange(2, 10, 2))
        # GVal.initLoopPARA(2701, [0.7, 0.8, 0.9, 1, 1.1, 1.2])

        # GVal.initLoopPARA(3303, [])
        # GVal.initLoopPARA(3501, [])
        # GVal.initLoopPARA(3502, [])
        # GVal.initLoopPARA(3503, [])
        # GVal.initLoopPARA(3504, [])
        # GVal.initLoopPARA(3505, [])
        # GVal.initLoopPARA(3506, [])
        # GVal.initLoopPARA(3507, [])


    ###########################################

    ###########################################
    ### [ FLAG settings ] ##########################
    # FLAG is a main output for the controlPanel() function. Those FLAGs are only available in file NFDALauncher.py (including its main section as well as all the subfunctions in this file. )
    # Initialize the FLAG
    FLAG = {}

    # [data_prepare_flag] to control whether read the raw data and create the label and feature.
    # 1 - Read raw data, and create feature and label according to the raw data
    # 0 - Donot read raw data, just load the last time processed data(feature and label), this saved much time when raw data is big and no need to create label and feature again.
    FLAG['data_prepare_flag'] = 0
    # [firstTimeConstruction], when many loops are processing in one execution, the construcion of the label and feature matrix should be done only one time. This is initializing this.
    GVal.setPARA('firstTimeConstruction', FLAG['data_prepare_flag'])

    # [save_flag] to save the created label and feature in this processing. No matter these label and feature and created or loaded
    # 1- save  0 -not save
    FLAG['save_flag'] = 0
    # [plotting_flag] to control whether to plot the result. 1 - plot, 0 -not plot
    # This flag will soon be deprecated since there will be various plotting tasks and more detailed flag will then be applied.
    FLAG['plotting_flag'] = 1
    ###########################################

    ###########################################
    ### [ Static Parameter #1] ######################
    # Static parameter #1 is the weights setting. The weight has its own coressponding label type. Detail below:

    # [weights_on_PARA] is a switcher for using(1) or not using(0) the weights.
    GVal.setPARA('weights_on_PARA', 0)

   # Weight_list, the key is referring the label num (0 -nofatigue, 1 -7gradefatigue ... )
    # And the corresponding value for each key is the weight for this certain class
    GVal.setPARA(
        'weight_list_PARA',  {
            0: 0.8,
            1: 1,
            2: 2,
            3: 3
        })
    ###########################################

    ###########################################
    ### [ Static Parameter #2] ######################
    # Static parameter # 2 is the beta setting. beta is a coefficient for the F-score calculation.
    # Please refer the  subfucntion fScore( )  in trainerSubFunc_NFDA_J.py
    GVal.setPARA('beta_PARA', 1.5)
    ###########################################

    ###########################################
    ### [ Static Parameter #3] ######################
    # Static parameter # 3 is the method choosing on processing the label '1' during the training and validation process
    # [del] means kick off the label '1'
    # [0] means set the label '1' as negative label
    # [1] means set the label '1' as positive label

    # [kick_off_no1_PARA] is the main switcher, the parameter can be set from 0 - 8
    GVal.setPARA('kick_off_no1_PARA', 0)
    GVal.setDVPARA('kick_off_no1_PARA', 0)
    # Recording the processing details for each different methods.
    kick_off_no1_detail = {
        0: '0 - Train: [del] || Validation: [del]',
        1: '1 - Train: [del] || Validation: [0]',
        2: '2 - Train: [del] || Validation: [1]',
        3: '3 - Train: [1] || Validation: [del]',
        4: '4 - Train: [1] || Validation: [0]',
        5: '5 - Train: [1] || Validation: [1]',
        6: '6 - Train: [0] || Validation: [del]',
        7: '7 - Train: [0] || Validation: [0]',
        8: '8 - Train: [0] || Validation: [1]'
    }
    # Save the processing details in the global statement.
    GVal.setPARA('kick_off_no1_detail', kick_off_no1_detail)
    ###########################################

    ###########################################
    ### [ Static Parameter #4] #####################
    # Static Parameter 4 is the method choosing on processing the special labels.(Including nonconfident frame and noise frame, two types)

    # Main Switcher to do the label processing. 1 - do, 0 - not do
    FLAG['label_process_flag'] = 1

    # [noconfident_frame_process_PARA], set as 0-5, details in the comments below.
    GVal.setPARA('noconfident_frame_process_PARA', 5)
    GVal.setDVPARA('noconfident_frame_process_PARA', 5)
    # 0 - Remain the current label (Do nothing)
    # 1 - Set all not confident frame as label 7 and remain all
    # 2 - Set all not confident frame as label 8 and remain all
    # 3 - Delete not confident frame with label 7
    # 4 - Delete not confident frame with label 8
    # 5 - Delete all the not confident frame

    # [noise_frame_process_PARA], set as 0 -2,details in the comments below.
    GVal.setPARA('noise_frame_process_PARA', 2)
    GVal.setDVPARA('noise_frame_process_PARA', 2)
    # 0 - Remain all the noise frame (Do nothing)
    # 1 - Simply Delete the noise frame and remain the noblink frame
    # 2 - Delete the noise frame and the noblink frame

    # The colomn number of the feature matrix
    # 20 - noise frame info
    # 22 - noblink frame info
    GVal.setPARA('noise_label_index_PARA', np.array([20, 22]))
    #########################################

    ##########################################
    ### [ Static Parameter #5] #####################
    # Static parameter #5 is the method choosing on the separation strategy of training and validation sets.

    # [split_type_num], set as 1-5, details in the comments below.0
    GVal.setPARA('split_type_num', 5)
    GVal.setDVPARA('split_type_num', 5)
    # 1:  caseLOO, nonparameters
    # 2:  caseLPO, parameter P (positive integer)
    # 3:  caseKFold, parameter K fold  (positive integer)
    # 4:  caseStratifiedKFold, parameter K fold  (positive integer)
    # 5:  caseRawsplit, parameter test_proportion  (range [ 0 : 1])

    # [split_type_para], for each split type, there is a certain parameter, detail is in the comments above
    GVal.setPARA('split_type_para', 0.25)
    GVal.setDVPARA('split_type_para', 0.25)
    # [random_seed_PARA], a random seed to get different random series.
    # Later this parameter will be improved and applied an overall random seed. Not only the dataset separation.
    GVal.setPARA('random_seed_PARA', np.arange(30,41,1))
    # Calulate the N fold into one res by 'mean ' /  'median'
    GVal.setPARA('resCalMethod_PARA','median')
    ##########################################

    ##########################################
    ### [ Static Parameter #6] #####################
    # Static parameter # 6 is the method choosing on the  feature selection

    # Main Switcher to do the feature selection. 1 - do, 0 - not do
    FLAG['fea_selection_flag'] = 0

    # [feaSelectionStrategy], set as 1-2, details in the comments below.
    GVal.setPARA('feaSelectionStrategy', 2)
    # 1 - tSNE(Default with PCA)
    # 2 - normalPCA

    # [nComponent_PARA] is the retained components number, should be smaller than total number of label
    GVal.setPARA('nComponent_PARA', 10)
    ##########################################
    ##########################################

    ### [ Static Parameter #7] #####################
    # Static parameter # 7 is the method choosing on the data balancing

    # Main Switcher to do the data balance. 1 - do, 0 - not do
    FLAG['data_balance_flag'] = 1

    # [data_balance_process_PARA] is set as 1, details in comments below
    GVal.setPARA('data_balance_process_PARA', 1)
    # 1 - Down sampling the negative class (para is the [0,1] float, indicating the percentage of retain)

    # [data_balance_para_PARA], for each data balance method above, there is a corresponding parameters as input,its range define in comments above
    GVal.setPARA('data_balance_para_PARA', 0.25)
    GVal.setDVPARA('data_balance_para_PARA', 0.25)
    ##########################################

    ##########################################
    ### [ Static Parameter #8] #####################
    # Static parameter # 8 is the feature information management.

    # Select several features from the online feature file
    # [online_fea_selectindex_PARA], defined by an array, it contains the colomn number of each feature in the created feature matrix
    GVal.setPARA('online_fea_selectindex_PARA', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
    # GVal.setPARA('online_fea_selectindex_PARA', np.array([16, 17]))

    # [online_fea_name_PARA] is the name for all those features in the created feature matrix. Breif info is attached in the comments below.
    GVal.setPARA('online_fea_name_PARA', [['眨眼时间'], ['眨眼速度'], ['perclos_first'], ['闭眼时间'], ['完全闭合时间'],
                                          ['睁眼时间'], ['闭眼平均速度'], ['睁眼平均速度'], ['闭眼最大速度'], ['睁眼最大速度'],
                                          ['perclos_second'], ['perclos_third'], ['perclos_fourth'], ['sem'], ['sem重心频率'],
                                          ['快速眼电相对功率'], ['sem合成特征'], ['眨眼合成特征'], ['综合特征'],
                                          ['状态指数'], ['噪声标记'], ['标准差'], ['眨眼数'], ['眨眼幅值'], ['眨眼极值差'],
                                          [' '], [' '],
                                          ])
    # 0 -- 12 维 对应的分别是：眨眼时间、眨眼速度、perclos_first、闭眼时间、完全闭合时间、
    #                                           睁眼时间、闭眼平均速度、睁眼平均速度、闭眼最大速度、睁眼最大速度、
    #                                           perclos_second、perclos_third、perclos_fourth
    #                                           perclos_first=完全闭合时间/闭眼时间
    #                                           perclos_second=完全闭合时间/眨眼时间
    #                                           perclos_third=(open_st-close_st)/(open_en-close_st)
    #                                           perclos_fourth=(close_en-close_st)/(open_en-close_st)
    # 13 -- 15 维 对应的分别是： sem、sem重心频率、快速眼电相对功率(1.2Hz-3Hz/0.3Hz-4.8Hz)
    # 16 -- 18 维 对应的分别是：sem合成特征、眨眼合成特征、综合特征
    # 19 维 对应的分别是：状态指数
    # 20 维  噪声类型的标记，0--无噪声，
    #                                         1--25Hz噪声，
    #                                         2--低频晃动噪声，
    #                                         3--空载(无明显中频脑电)，
    #                                         4--咀嚼噪声,
    #                                         5--信号长度不足8秒，
    #                                         6--空载(标准差过小4) ,
    #                                         7--严重漂移
    # 21 -- 24 维 对应的分别是：标准差，眨眼数  眨眼幅值  眨眼极值差

    # [feature_index_PARA], serial number for the online_fea_selectindex.
    GVal.setPARA('feature_index_PARA', np.arange(0, len(GVal.getPARA('online_fea_selectindex_PARA'))))  # THINK TO RETIRE THIS LINE

    # [screen_fea_list_PARA] is the colomn number for features, which used as the result plotting.
    GVal.setPARA('screen_fea_list_PARA', np.array([16, 17]))
    ##########################################
    return FLAG


def initializationProcess():
    #  Define the prefix, this is used for different computation environment.
    localprefix = '/home/'
    serverprefix = '/home/labcompute/'

    # locipe local ip end
    locipe = os.environ['SSH_CLIENT'][10:12]
    print(locipe)
    if locipe == '21':
        # When start working with run_NFDA.sh  (run in slave service)
        GVal.setPARA('prefix_PARA', serverprefix)
        username = os.environ["USERNAME"]
    elif locipe == '45':
        # When working with command: python3 NFDA_Launcher.py (run in master local)
        GVal.setPARA('prefix_PARA', localprefix)
        username = 'GaoMY'
        os.remove(localprefix + 'GaoMY/tlog.txt')
    return GVal.getPARA('prefix_PARA'), username


def processCodeEncoder():
    # processCodeEncoder, get all the parameters that set in the controlPanel(), and encode everything into unique process_code for each unique process loop.
    process_code_pack = []

    # [ Loop Parameters #1]
    recording_index_list = GVal.getPARA('recording_index_list_PARA')
    # [ Loop Parameters #2]
    classifier_list = GVal.getPARA('classifier_list_PARA')

    # [ Other Loop Parameter]
    loopPARA_cache = GVal.getLoopPARA_cache()
    # Get total amount of loop parameters
    loopPARA_amount = len(loopPARA_cache)
    GVal.setPARA('loopPARA_amount_PARA', loopPARA_amount)

    # When there's no LOOP parameter
    if loopPARA_amount == 0:
        for recording_index_list_lplb in np.arange(len(recording_index_list)):
            for classifier_list_lplb in np.arange(len(classifier_list)):
                code_temp = int(1e0 * classifier_list[classifier_list_lplb] +
                                1e3 * (recording_index_list_lplb + 1) +
                                1e5 * 0 +
                                1e7 * 0 +
                                1e11 * 1)
                process_code_pack.append(code_temp)
        return process_code_pack

    # initialize the totalsize parameter. (n1*n2*n3...)
    if GVal.getPARA('codepoolCreator') == 'iter':
        loopPARA_totalsize = 1
    if GVal.getPARA('codepoolCreator') == 'plain':
        loopPARA_totalsize = 0
    # initialize the bitsize for each parameter ([n1, n2 , n3,...])
    loopPARA_bitsize = cell2dmatlab_jsp([loopPARA_amount], 1, 0)
    # cache to save name for each loop parameter
    loopPARA_namecache = cell2dmatlab_jsp([loopPARA_amount, 1], 2, [])
    #
    dVPARA_cache = cell2dmatlab_jsp([loopPARA_amount], 1, 0)

    # Loop scan the loopPara_cache.
    j = 0
    h = {}

    for loopPARA_name in loopPARA_cache.keys():
        # The inner-classifier parameters are defined with int number.
        if type(loopPARA_name) == int:
            if len(loopPARA_cache[loopPARA_name][0]) < 1:
                loopPARA_cache[loopPARA_name][0] = dVM[loopPARA_name][1]
                loopPARA_cache[loopPARA_name][1] = len(loopPARA_cache[loopPARA_name][0])
            dVPARA_cache[j] = dVM[loopPARA_name][2]
        else:
            dVPARA_cache[j] = GVal.getDVPARA(loopPARA_name)

        # print(type(dVPARA_cache[j]))
        # print(dVPARA_cache[j])
        # print(type(loopPARA_cache[loopPARA_name][0].tolist()))
        # print(loopPARA_cache[loopPARA_name][0])
        # print(type(loopPARA_cache[loopPARA_name][0][0].tolist()))
        # print(loopPARA_cache[loopPARA_name][0][0])

        # When the default value is not in the loop list, added into loop list.
        if dVPARA_cache[j] not in loopPARA_cache[loopPARA_name][0]:
            loopPARA_cache[loopPARA_name][0] = np.append(loopPARA_cache[loopPARA_name][0], dVPARA_cache[j])
            loopPARA_cache[loopPARA_name][1] += 1
        # the totalsize of all loop
        if GVal.getPARA('codepoolCreator') == 'iter':
            loopPARA_totalsize *= loopPARA_cache[loopPARA_name][1]
        if GVal.getPARA('codepoolCreator') == 'plain':
            loopPARA_totalsize += loopPARA_cache[loopPARA_name][1]
        # the size of each loop parameter
        loopPARA_bitsize[j] = int(loopPARA_cache[loopPARA_name][1])
        # the name of each loop parameter
        loopPARA_namecache[j][0] = loopPARA_name
        # the content of each loop parameter( the true value to be set in each loop)
        h[j] = loopPARA_cache[loopPARA_name][0]
        j += 1

    # Important processCode Para 1.Convey into decoder
    GVal.setPARA('loopPARA_namecache_PARA', loopPARA_namecache)

    loopPARA_codepool = cell2dmatlab_jsp([loopPARA_amount], 1, 0)
    loopPARA_temppool = cell2dmatlab_jsp([loopPARA_amount], 1, 0)

    # Create the real code pool (n x 1 array, each line is a process code, n = totalsize)
    codepoolCreatorList = {
        'iter': iterCodePool,
        'plain': plainCodePool,
    }
    loopPARA_codepool = codepoolCreatorList[GVal.getPARA('codepoolCreator')](loopPARA_amount, loopPARA_bitsize, loopPARA_temppool, h, loopPARA_codepool)

    # loopPARA_codepool = iterCodePool(loopPARA_amount, loopPARA_bitsize, loopPARA_temppool, h, loopPARA_codepool)
    loopPARA_codepool = copy.deepcopy(loopPARA_codepool[1:])

    row_dV = loopPARA_codepool.shape[0]
    colomn_dV = loopPARA_codepool.shape[1]
    dVPARA_count = cell2dmatlab_jsp([row_dV, colomn_dV + 1], 2, 0)
    dVPARA_index = cell2dmatlab_jsp([colomn_dV], 1, [])
    dVPARA_value = cell2dmatlab_jsp([colomn_dV], 1, [])
    for line_dV in range(row_dV):
        for col_dV in range(colomn_dV):
            if str2num(loopPARA_codepool[line_dV][col_dV]) == dVPARA_cache[col_dV]:
                dVPARA_count[line_dV][col_dV] = 1
            else:
                dVPARA_count[line_dV][col_dV] = 0
        dVPARA_count[line_dV][colomn_dV] = sum(dVPARA_count[line_dV][0:colomn_dV])

        if dVPARA_count[line_dV][colomn_dV] == colomn_dV - 1:
            # We have all-1 '1'
            dVPARA_index[dVPARA_count[line_dV].index(0)].append(line_dV)
            dVPARA_value[dVPARA_count[line_dV].index(0)].append(str2num(loopPARA_codepool[line_dV][dVPARA_count[line_dV].index(0)]))
        elif dVPARA_count[line_dV][colomn_dV] == colomn_dV:
            # We have all '1'
            for i in range(colomn_dV):
                dVPARA_index[i].append(line_dV)
                dVPARA_value[i].append(str2num(loopPARA_codepool[line_dV][i]))
    # dVPARA_index contain the index for single parameter analysis (mostly picturing)
    GVal.setPARA('dVPARA_index_PARA', dVPARA_index)
    GVal.setPARA('dVPARA_value_PARA', dVPARA_value)

    # Example:
    # loopPARA_codepool[loopPARA_serialnum][loopPARA_num]

    # Important processCode Para 2. Convey into decoder
    GVal.setPARA('loopPARA_codepool_PARA', loopPARA_codepool)
    print('### Loop Para Pool Size: [' + str(loopPARA_totalsize) + ', ' + str(loopPARA_amount) + '] ([Total Loop,Loop Parameter amount])')

    # Code rule: [seal bit set 1](11)[loopParaSerialCode](10-7)[loopParaAmount](6,5)[indexlistSerialNum](4,3)[ClassifierList](2-0)
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
    if loopPARA_amount > 0:
        loopPARA_codepool = GVal.getPARA('loopPARA_codepool_PARA')
        loopPARA_namecache = GVal.getPARA('loopPARA_namecache_PARA')

        loopPARA_serialnum = int(process_code_str[-11:-7])
        for loopPARA_num in range(loopPARA_amount):
            # Set the value in current loop according to the process_code
            GVal.setPARA(loopPARA_namecache[loopPARA_num][0], loopPARA_codepool[loopPARA_serialnum][loopPARA_num])
            if type(loopPARA_namecache[loopPARA_num][0]) == int:
                dVM[loopPARA_namecache[loopPARA_num][0]][2] = str2num(loopPARA_codepool[loopPARA_serialnum][loopPARA_num])
                print('### Looping Parameter (' + str(loopPARA_num + 1) + '/' + str(loopPARA_amount) + '): [' + str(loopPARA_namecache[loopPARA_num][0]) + ' | ' + dVM[loopPARA_namecache[loopPARA_num][0]][0] + '], with value set: [' + str(loopPARA_codepool[loopPARA_serialnum][loopPARA_num]) + ']')
            else:
                print('### Looping Parameter (' + str(loopPARA_num + 1) + '/' + str(loopPARA_amount) + '): [' + str(loopPARA_namecache[loopPARA_num][0]) + '], with value set: [' + str(loopPARA_codepool[loopPARA_serialnum][loopPARA_num]) + ']')

    select_index_temp = recording_index_list[recording_index_list_selectserial]

    GVal.setPARA('classifier_num_PARA', classifier_num_temp)
    GVal.setPARA('select_index_PARA', select_index_temp)
    print('######## [ Decoder Success! ] ########')

    return select_index_temp, classifier_num_temp


def iterCodePool(nlevel, dims, temp, h, y):
    # This function is working to create a matrix contain all the possible combination of a uncertian number of lists(or arrays).
    # Use the iteration.
    nlevel -= 1
    for i in range(dims[-1]):
        temp[nlevel] = h[nlevel][i]
        if nlevel > 0:
            y = iterCodePool(nlevel, dims[:-1], temp, h, y)
        else:
            y = np.vstack((y, temp))
    return y


def plainCodePool(nlevel, dims, temp, h, y):
    LP_nc = GVal.getPARA('loopPARA_namecache_PARA')
    dV = cell2dmatlab_jsp([len(LP_nc)], 1, [])
    for n in range(len(LP_nc)):
        dV[n] = GVal.getDVPARA(LP_nc[n][0])

    for i in range(nlevel):
        for j in range(dims[i]):
            tempdV = copy.deepcopy(dV)
            tempdV[i] = h[i][j]
            y = np.vstack((y, tempdV))

    return y
