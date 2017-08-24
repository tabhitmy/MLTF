import numpy as np
import GVal
import copy
from toolkitJ import cell2dmatlab_jsp


def initializationProcess():
    #  Define the prefix, this is used for different computation environment.
    localprefix = '/home/'
    serverprefix = '/home/labcompute/'
    GVal.setPARA('prefix_PARA', serverprefix)
    # GVal.setPARA('prefix_PARA', localprefix)

    return GVal.getPARA('prefix_PARA')


def controlPanel():
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
    GVal.setPARA(
        'classifier_list_PARA',
        [
            21,
            # 24,
            # 25,
            30,
            # 23,
            32

        ])
    ###########################################

    ###########################################
    ### [ Loop Parameters, others ] ##################
    # Add any para you want to loop. You can loop many parameters together.
    # The loop mechanism is boosted by GVal.initLoopPARA, add the exact name of any parameters in this control panel, then set a coresponding array, in which all the values will be import as looping parameters.

    GVal.initLoopPARA('noconfident_frame_process_PARA',  np.arange(6))
    # GVal.initLoopPARA('random_seed_PARA', np.arange(1, 20, 1))
    # GVal.initLoopPARA('kick_off_no1_PARA', np.arange(3))
    ###########################################

    ###########################################
    ### [ FLAG settings ] ##########################
    # FLAG is a main output for the controlPanel() function. Those FLAGs are only available in file NFDALauncher.py (including its main section as well as all the subfunctions in this file. )
    # Initialize the FLAG
    FLAG = {}

    # [data_prepare_flag] to control whether read the raw data and create the label and feature.
    # 1 - Read raw data, and create feature and label according to the raw data
    # 0 - Donot read raw data, just load the last time processed data(feature and label), this saved much time when raw data is big and no need to create label and feature again.
    FLAG['data_prepare_flag'] = 1
    # [firstTimeConstruction], when many loops are processing in one execution, the construcion of the label and feature matrix should be done only one time. This is initializing this.
    GVal.setPARA('firstTimeConstruction', FLAG['data_prepare_flag'])

    # [save_flag] to save the created label and feature in this processing. No matter these label and feature and created or loaded
    # 1- save  0 -not save
    FLAG['save_flag'] = 1

    # [plotting_flag] to control whether to plot the result. 1 - plot, 0 -not plot
    # This flag will soon be deprecated since there will be various plotting tasks and more detailed flag will then be applied.
    FLAG['plotting_flag'] = 0
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
    GVal.setPARA('noconfident_frame_process_PARA', 1)
    # 0 - Remain the current label (Do nothing)
    # 1 - Set all not confident frame as label 7 and remain all
    # 2 - Set all not confident frame as label 8 and remain all
    # 3 - Delete not confident frame with label 7
    # 4 - Delete not confident frame with label 8
    # 5 - Delete all the not confident frame

    # [noise_frame_process_PARA], set as 0 -2,details in the comments below.
    GVal.setPARA('noise_frame_process_PARA', 1)
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

    # [split_type_num], set as 1-5, details in the comments below.
    GVal.setPARA('split_type_num', 5)
    # 1:  caseLOO, nonparameters
    # 2:  caseLPO, parameter P (positive integer)
    # 3:  caseKFold, parameter K fold  (positive integer)
    # 4:  caseStratifiedKFold, parameter K fold  (positive integer)
    # 5:  caseRawsplit, parameter test_proportion  (range [ 0 : 1])

    # [split_type_para], for each split type, there is a certain parameter, detail is in the comments above
    GVal.setPARA('split_type_para', 0.3)

    # [random_seed_PARA], a random seed to get different random series.
    # Later this parameter will be improved and applied an overall random seed. Not only the dataset separation.
    GVal.setPARA('random_seed_PARA', 23)
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

    return FLAG, processCodeEncoder()


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

    # When there's no parameter
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
    loopPARA_totalsize = 1
    # initialize the bitsize for each parameter ([n1, n2 , n3,...])
    loopPARA_bitsize = cell2dmatlab_jsp([loopPARA_amount], 1, 0)
    # cache to save name for each loop parameter
    loopPARA_namecache = cell2dmatlab_jsp([loopPARA_amount, 1], 2, [])

    # Loop scan the loopPara_cache.
    j = 0
    h = {}
    for loopPARA_name in loopPARA_cache.keys():
        # the totalsize of all loop
        loopPARA_totalsize *= loopPARA_cache[loopPARA_name][1]
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
    loopPARA_codepool = iterCodePool(loopPARA_amount, loopPARA_bitsize, loopPARA_temppool, h, loopPARA_codepool)
    loopPARA_codepool = copy.deepcopy(loopPARA_codepool[1:])
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
            print('### Looping Parameter (' + str(loopPARA_num) + '/' + str(loopPARA_amount) + '): [' + loopPARA_namecache[loopPARA_num][0] + '], with value set: [' + str(loopPARA_codepool[loopPARA_serialnum][loopPARA_num]) + ']')
            # Set the value in current loop according to the process_code
            GVal.setPARA(loopPARA_namecache[loopPARA_num][0], loopPARA_codepool[loopPARA_serialnum][loopPARA_num])

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
