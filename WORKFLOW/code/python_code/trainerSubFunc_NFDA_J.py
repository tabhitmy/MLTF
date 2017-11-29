
import numpy as np
import copy


import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
import matplotlib.pyplot as plt
from toolkitJ import cell2dmatlab_jsp
import GVal

import seaborn as sns

####################################################


####################################################

def fScore(beta, P, R):
    if ((beta**2) * P + R) == 0:
        F = -1
    else:
        F = (1 + beta**2) * (P * R) / ((beta**2) * P + R)
    return F


def dataSetPreparation(feature_index, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw):
    # Get only the features requested.
    X = X_train_raw[:, feature_index]
    # Get only the label itselt. no other information
    y = Y_train_raw[:, 0]

    X_valid = X_valid_raw[:, feature_index]
    y_valid = Y_valid_raw[:, 0]

    index_no = cell2dmatlab_jsp([2, 4], 2, [])
    for i in range(4):
        # Count up each label type 0,1,2,3
        index_no[0][i] = np.nonzero(y == i)[0]
        index_no[1][i] = np.nonzero(y_valid == i)[0]

    print(('### Training Dataset Size: ' + str(X_train_raw.shape)))
    print(('### Fatigue Sample Numbers (TrainingSet): ' +
           '[0]-' + str(len(index_no[0][0])) +
           ' || [1]-' + str(len(index_no[0][1])) +
           ' || [2]-' + str(len(index_no[0][2])) +
           ' || [3]-' + str(len(index_no[0][3]))
           ))

    print(('### Validation Dataset Size: ' + str(X_valid_raw.shape)))
    print(('### Fatigue Sample Numbers (ValidationSet): ' +
           '[0]-' + str(len(index_no[1][0])) +
           ' || [1]-' + str(len(index_no[1][1])) +
           ' || [2]-' + str(len(index_no[1][2])) +
           ' || [3]-' + str(len(index_no[1][3]))
           ))

    return X, y, X_valid, y_valid, index_no


def calculateFRAP(Z, y_val, screen):
    # F - Fscore
    # R - Recall
    # A - Accuracy
    # P - Precision
    TP = len(np.nonzero(np.logical_and((Z == 1), (y_val == 1)))[0])
    TN = len(np.nonzero(np.logical_and((Z == 0), (y_val == 0)))[0])
    FP = len(np.nonzero(np.logical_and((Z == 1), (y_val == 0)))[0])
    FN = len(np.nonzero(np.logical_and((Z == 0), (y_val == 1)))[0])

    if TP + FP == 0:
        P = -1
        FA = -1
    else:
        P = TP / (TP + FP)
        FA = 1 - P

    if (TP + TN + FP + FN) == 0:
        A = -1
    else:
        A = (TP + TN) / (TP + TN + FP + FN)

    if TP + FN == 0:
        R = -1
        MA = -1
    else:
        R = TP / (TP + FN)
        MA = 1 - R
    # P = TP / (TP + FP)
    # A = (TP + TN) / (TP + TN + FP + FN)
    # R = TP / (TP + FN)
    # MA = 1 - R
    # FA = 1 - P
    F1 = fScore(1, P, R)
    beta = GVal.getPARA('beta_PARA')
    F = fScore(beta, P, R)

    FRAP = [P, A, R, MA, FA, F1, F]
    if screen == 1:
        print(('###[ P: ' + str(round(FRAP[0], 4)) +
               ' || A: ' + str(round(FRAP[1], 4)) +
               ' || R: ' + str(round(FRAP[2], 4)) +
               ' || MA: ' + str(round(FRAP[3], 4)) +
               ' || FA: ' + str(round(FRAP[4], 4)) +
               ' || F1: ' + str(round(FRAP[5], 4)) +
               ' || F' + str(beta) + ': ' + str(round(FRAP[6], 4)) + ']###'
               ))
    return FRAP


def dataRegulationSKL(y_tra_in, X_tra_in, y_val_in, X_val_in, index_no):
    # This function is working to deal with several data problems below:
    # 1. Processing with the transitional frame (no1)
    # 2. Processing wth label weights

    kick_off_no1 = GVal.getPARA('kick_off_no1_PARA')
    weights_on = GVal.getPARA('weights_on_PARA')
    weight_list = GVal.getPARA('weight_list_PARA')

    # Label weight issue
    if weights_on:
        weights = copy.deepcopy(y_tra_in)
        for key in weight_list.keys():
            weights[np.nonzero(weights == key)[0]] = weight_list[key]
        print(weights)
    else:
        weights = np.ones(y_tra_in.shape)

    # Transitional frame issue
    kick_off_no1_switcher = {
        0: ko1processor0,
        1: ko1processor1,
        2: ko1processor2,
        3: ko1processor3,
        4: ko1processor4,
        5: ko1processor5,
        6: ko1processor6,
        7: ko1processor7,
        8: ko1processor8
    }

    # Avoiding the overwrite problem when multiply excuting dataRegulationSKL
    y_tra_temp = copy.deepcopy(y_tra_in)
    X_tra_temp = copy.deepcopy(X_tra_in)
    y_val_temp = copy.deepcopy(y_val_in)
    X_val_temp = copy.deepcopy(X_val_in)

    print('###  Label 1 Processing Method: ' + GVal.getPARA('kick_off_no1_detail')[int(GVal.getPARA('kick_off_no1_PARA'))])
    y_tra, X_tra, y_val, X_val, weights = kick_off_no1_switcher[int(GVal.getPARA('kick_off_no1_PARA'))](y_tra_temp, X_tra_temp, y_val_temp, X_val_temp, weights, index_no)

    X_tra_res = copy.deepcopy(X_tra)
    X_val_res = copy.deepcopy(X_val)

    y_tra_res = copy.deepcopy(y_tra)
    y_val_res = copy.deepcopy(y_val)
    GVal.setPARA('y_tra_res_PARA', y_tra_res)
    GVal.setPARA('y_val_res_PARA', y_val_res)
    GVal.setPARA('X_tra_res_PARA', X_tra_res)
    GVal.setPARA('X_val_res_PARA', X_val_res)
    # Change the staged level 0,1,2,3 into a binary situation 0 and 1(1,2,3)
    y_tra[np.nonzero(y_tra > 0)[0]] = 1
    y_val[np.nonzero(y_val > 0)[0]] = 1

    # Checking code, do not delete .
    print('X train')
    print(X_tra.shape)
    print('y train')
    print(y_tra.shape)
    print(max(y_tra), min(y_tra))
    print('X validation')
    print(X_val.shape)
    print('y validation')
    print(y_val.shape)
    print(max(y_val), min(y_val))

    # print(len(np.nonzero(y_tra_res == 0)[0]))
    # print(len(np.nonzero(y_tra_res == 1)[0]))
    # print(len(np.nonzero(y_tra_res == 2)[0]))
    # print(len(np.nonzero(y_tra_res == 3)[0]))

    FLAG_temp = GVal.getPARA('FLAG_PARA')
    pcode4fig = GVal.getPARA('process_code_PARA')
    if FLAG_temp['hist_plotting_flag']:
        ftsize = 16
        hist_fea_list = GVal.getPARA('hist_fea_list_PARA')
        path = GVal.getPARA('path_PARA')
        nbins = 20
        figcode = pcode4fig
        h = plt.figure(num=figcode, figsize=(20, 30))
        inu = 0
        for ireal in hist_fea_list:
            iserial = int(np.where(hist_fea_list == ireal)[0])
            print(GVal.getPARA('online_fea_selectindex_PARA'))
            inum = int(np.where(GVal.getPARA('online_fea_selectindex_PARA') == ireal)[0])
            print('inu: ' + str(inu) + ' || ireal: ' + str(ireal) + '|| iserial: ' + str(iserial) + '|| inum: ' + str(inum))
            x = 0.03
            y = 0.89
            a = -0.02
            le = 0.19
            lfig = 0.17
            k = -0.085
            kfig = 0.065

            posx = [x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le,
                    x, x + le, x + 2 * le, x + 3 * le, x + 4 * le]
            posy = [y, y, y, y, y, y + k, y + k, y + k, y + k, y + k,
                    y + 2 * k + a, y + 2 * k + a, y + 2 * k + a, y + 2 * k + a, y + 2 * k + a,
                    y + 3 * k + a, y + 3 * k + a, y + 3 * k + a, y + 3 * k + a, y + 3 * k + a,
                    y + 4 * k + 2 * a, y + 4 * k + 2 * a, y + 4 * k + 2 * a, y + 4 * k + 2 * a, y + 4 * k + 2 * a,
                    y + 5 * k + 2 * a, y + 5 * k + 2 * a, y + 5 * k + 2 * a, y + 5 * k + 2 * a, y + 5 * k + 2 * a,
                    y + 6 * k + 3 * a, y + 6 * k + 3 * a, y + 6 * k + 3 * a, y + 6 * k + 3 * a, y + 6 * k + 3 * a,
                    y + 7 * k + 3 * a, y + 7 * k + 3 * a, y + 7 * k + 3 * a, y + 7 * k + 3 * a, y + 7 * k + 3 * a,
                    y + 8 * k + 4 * a, y + 8 * k + 4 * a, y + 8 * k + 4 * a, y + 8 * k + 4 * a, y + 8 * k + 4 * a,
                    y + 9 * k + 4 * a, y + 9 * k + 4 * a, y + 9 * k + 4 * a, y + 9 * k + 4 * a, y + 9 * k + 4 * a]
            # print(len(posx))
            # print(len(posy))

            inum1 = int(10 * np.floor((inu) / 5) + (iserial) % 5)
            # print(inu, inum1)
            # print(posx[inum1], posy[inum1])
            plt.axes([posx[inum1], posy[inum1], lfig, kfig])
            sns.distplot(X_tra_res[np.nonzero(y_tra_res > 1)[0], inum], bins=nbins, rug=False, label='posi [23]')
            sns.distplot(X_tra_res[np.nonzero(y_tra_res == 0)[0], inum], bins=nbins, rug=False, label='nega [0]')
            sns.distplot(X_tra_res[np.nonzero(y_tra_res == 1)[0], inum], bins=nbins, rug=False, label='tranz [1]')
            # plt.legend .....
            plt.title('[ Feature No. ' + str(ireal) + ' ] Trainning Set ')
            plt.legend()
            inum2 = int(10 * np.floor((inu) / 5) + (iserial) % 5 + 5)
            # print(inu, inum2)
            plt.axes([posx[inum2], posy[inum2], lfig, kfig])
            sns.distplot(X_val_res[np.nonzero(y_val_res > 1)[0], inum], bins=nbins, rug=False, label='posi[23]')
            sns.distplot(X_val_res[np.nonzero(y_val_res == 0)[0], inum], bins=nbins, rug=False, label='nega[0]')
            sns.distplot(X_val_res[np.nonzero(y_val_res == 1)[0], inum], bins=nbins, rug=False, label='tranz [1]')
            # print(GVal.getPARA('online_fea_name_PARA')[i][0])
            plt.title('[ ' + GVal.getPARA('online_fea_name_PARA')[ireal][0] + ' ] Testing Set ', fontproperties=zhfont, fontsize=ftsize)
            if inu == 0:
                plt.legend()
            inu += 1
        plt.show()
        plt.savefig((path['fig_path'] + 'Histo_Fea_Pic' + str(figcode) + '.png'))
        print('Histo Feature Picture' + str(figcode) + 'Saved!')
    else:
        print('### No hist feature plotting... Skip')
    return y_tra, X_tra, y_val, X_val, weights


def ko1processor0(y_tra, X_tra, y_val, X_val, weights, index_no):

    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)
    print('Delete Amount in Training Set: ' + str(len(index_no[0][1])))

    y_val = np.delete(y_val, index_no[1][1], axis=0)
    X_val = np.delete(X_val, index_no[1][1], axis=0)
    print('Delete Amount in Validation Set: ' + str(len(index_no[1][1])))
    return y_tra, X_tra, y_val, X_val, weights


def ko1processor1(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)
    print('Delete Amount in Training Set: ' + str(len(index_no[0][1])))

    y_val[np.nonzero(y_val == 1)[0]] = 0
    print('Flip Amount in Validation Set: ' + str(len(np.nonzero(y_val == 1)[0])))
    return y_tra, X_tra, y_val, X_val, weights


def ko1processor2(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)
    print('Delete Amount in Training Set: ' + str(len(index_no[0][1])))

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor3(y_tra, X_tra, y_val, X_val, weights, index_no):

    y_val = np.delete(y_val, index_no[1][1], axis=0)
    X_val = np.delete(X_val, index_no[1][1], axis=0)
    print('Delete Amount in Validation Set: ' + str(len(index_no[1][1])))

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor4(y_tra, X_tra, y_val, X_val, weights, index_no):

    y_val[np.nonzero(y_val == 1)[0]] = 0
    print('Flip Amount in Validation Set: ' + str(len(np.nonzero(y_val == 1)[0])))

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor5(y_tra, X_tra, y_val, X_val, weights, index_no):

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor6(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra[np.nonzero(y_tra == 1)[0]] = 0
    print('Flip Amount in Training Set: ' + str(len(np.nonzero(y_tra == 1)[0])))

    y_val = np.delete(y_val, index_no[1][1], axis=0)
    X_val = np.delete(X_val, index_no[1][1], axis=0)
    print('Delete Amount in Validation Set: ' + str(len(index_no[1][1])))

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor7(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra[np.nonzero(y_tra == 1)[0]] = 0
    print('Flip Amount in Training Set: ' + str(len(np.nonzero(y_tra == 1)[0])))

    y_val[np.nonzero(y_val == 1)[0]] = 0
    print('Flip Amount in Validation Set: ' + str(len(np.nonzero(y_val == 1)[0])))

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor8(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra[np.nonzero(y_tra == 1)[0]] = 0
    print('Flip Amount in Training Set: ' + str(len(np.nonzero(y_tra == 1)[0])))

    return y_tra, X_tra, y_val, X_val, weights


def processLearning(mdl, X_tra, y_tra, X_val, y_val):
    print('######## [Predicting ... ] ########')
    # Z is the classification result with the current model (mdl,no matter what kind of classifier)
    Z = mdl.predict(X_val)
    score = mdl.score(X_tra, y_tra)
    FRAP = calculateFRAP(Z, y_val, 1)

    Z_tra = mdl.predict(X_tra)
    GVal.setPARA('Z_tra_res_PARA', Z_tra)
    GVal.setPARA('Z_res_PARA', Z)
    return mdl, score,  FRAP
##
