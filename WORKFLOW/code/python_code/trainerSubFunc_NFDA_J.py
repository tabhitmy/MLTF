
import numpy as np
from toolkitJ import cell2dmatlab_jsp
import GVal


def fScore(beta, P, R):
    F = (1 + beta**2) * (P * R) / ((beta**2) * P + R)
    return F


def dataSetPreparation(feature_index, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw):

    X = X_train_raw[:, feature_index]
    y = Y_train_raw[:, 0]

    X_valid = X_valid_raw[:, feature_index]
    y_valid = Y_valid_raw[:, 0]

    index_no = cell2dmatlab_jsp([2, 4], 2, [])
    for i in range(4):
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


def calculateFRAP(Z, y_val):
    TP = len(np.nonzero(np.logical_and((Z == 1), (y_val == 1)))[0])
    TN = len(np.nonzero(np.logical_and((Z == 0), (y_val == 0)))[0])
    FP = len(np.nonzero(np.logical_and((Z == 1), (y_val == 0)))[0])
    FN = len(np.nonzero(np.logical_and((Z == 0), (y_val == 1)))[0])

    P = TP / (TP + FP)
    A = (TP + TN) / (TP + TN + FP + FN)
    R = TP / (TP + FN)
    MA = 1 - R
    FA = 1 - P
    F1 = fScore(1, P, R)
    beta = GVal.getPARA('beta_PARA')
    F = fScore(beta, P, R)

    FRAP = [P, A, R, MA, FA, F1, F]

    print(('###[ P: ' + str(round(FRAP[0], 4)) +
           ' || A: ' + str(round(FRAP[1], 4)) +
           ' || R: ' + str(round(FRAP[2], 4)) +
           ' || MA: ' + str(round(FRAP[3], 4)) +
           ' || FA: ' + str(round(FRAP[4], 4)) +
           ' || F1: ' + str(round(FRAP[5], 4)) +
           ' || F' + str(beta) + ': ' + str(round(FRAP[6], 4)) + ']###'
           ))
    return FRAP


def dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no):

    kick_off_no1 = GVal.getPARA('kick_off_no1_PARA')
    weights_on = GVal.getPARA('weights_on_PARA')
    weight_list = GVal.getPARA('weight_list_PARA')

    # print(max(weights), min(weights))
    if weights_on:
        weights = copy.deepcopy(y_tra)
        for key in weight_list.keys():
            weights[np.nonzero(weights == key)[0]] = weight_list[key]
    else:
        weights = np.ones(y_tra.shape)

    kick_off_no1_switcher = {
        0: ko1processor0,
        1: ko1processor1,
        2: ko1processor2,
        3: ko1processor3,
        4: ko1processor4
    }

    y_tra, X_tra, y_val, X_val, weights = kick_off_no1_switcher[GVal.getPARA('kick_off_no1_PARA')](y_tra, X_tra, y_val, X_val, weights, index_no)

    # Change the staged level 0,1,2,3 into a binary situation 0 and 1(1,2,3)
    y_tra[np.nonzero(y_tra > 0)[0]] = 1
    y_val[np.nonzero(y_val > 0)[0]] = 1
    return y_tra, X_tra, y_val, X_val, weights


def ko1processor0(y_tra, X_tra, y_val, X_val, weights, index_no):

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor1(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    y_val = np.delete(y_val, index_no[1][1], axis=0)
    X_val = np.delete(X_val, index_no[1][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor2(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)

    y_val[np.nonzero(y_val == 1)[0]] = 0

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor3(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra = np.delete(y_tra, index_no[0][1], axis=0)
    X_tra = np.delete(X_tra, index_no[0][1], axis=0)
    weights = np.delete(weights, index_no[0][1], axis=0)

    return y_tra, X_tra, y_val, X_val, weights


def ko1processor4(y_tra, X_tra, y_val, X_val, weights, index_no):
    y_tra[np.nonzero(y_tra == 1)[0]] = 0
    y_tra[np.nonzero(y_tra > 0)[0]] = 1

    y_val[np.nonzero(y_val == 1)[0]] = 0

    return y_tra, X_tra, y_val, X_val, weights


def processLearning(mdl, X_tra, y_tra, X_val, y_val):

    Z = mdl.predict(X_val)
    score = mdl.score(X_tra, y_tra)
    FRAP = calculateFRAP(Z, y_val)
    return mdl, score,  FRAP
##
