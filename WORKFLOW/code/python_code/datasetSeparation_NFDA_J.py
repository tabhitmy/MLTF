# datasetSeparation_NFDA_J.py
import numpy as np

import sklearn.model_selection as skmdls

from toolkitJ import cell2dmatlab_jsp
import GVal

###################################
# Subfunction ########################
###################################


def train_test_constructor(N, mdl, X, Y):
    X_tr = cell2dmatlab_jsp([N], 1, [])
    y_tr = cell2dmatlab_jsp([N], 1, [])
    X_te = cell2dmatlab_jsp([N], 1, [])
    y_te = cell2dmatlab_jsp([N], 1, [])
    n_fold = 0
    for train,  test in mdl.split(X, Y[:, 0]):
        X_tr[n_fold] = X[train, :]
        X_te[n_fold] = X[test, :]
        y_tr[n_fold] = Y[train, :]
        y_te[n_fold] = Y[test, :]
        n_fold += 1

    return X_tr, X_te, y_tr, y_te


def caseLOO(X, Y, para):
    para = 0
    loo = skmdls.LeaveOneOut()
    N = label_all.shape[1]
    mdl = loo
    X_train, X_test, y_train, y_test = train_test_constructor(N, mdl, X, Y)
    return X_train, X_test, y_train, y_test, N


def caseLPO(X, Y, para):
    P = para
    if P % 1 != 0:
        print('Warning! The input K is not a integer, for continue, do the ceilling on k ')
        P = math.ceil(P)

    lpo = skmdls.LeavePOut(p=P)
    nn = Y.shape[1]
    N = math.factorial(nn) / (math.factorial(nn - P) * math.factorial(P))
    mdl = lpo
    X_train, X_test, y_train, y_test = train_test_constructor(N, mdl, X, Y)
    return X_train, X_test, y_train, y_test, N


def caseKFold(X, Y, para):
    K = para
    if K % 1 != 0:
        print('Warning! The input K is not a integer, for continue, do the ceilling on k ')
        K = math.ceil(K)
    kf = skmdls.KFold(n_splits=K)
    N = K
    mdl = kf
    X_train, X_test, y_train, y_test = train_test_constructor(N, mdl, X, Y)
    return X_train, X_test, y_train, y_test, N


def caseStratifiedKFold(X, Y, para):
    K = para
    if K % 1 != 0:
        print('Warning! The input K is not a integer, for continue, do the ceilling on k ')
        K = math.ceil(K)
    skf = skmdls.StratifiedKFold(n_splits=K)
    N = K
    mdl = skf
    X_train, X_test, y_train, y_test = train_test_constructor(N, mdl, X, Y)

    return X_train, X_test, y_train, y_test, N


def caseRawsplit(X, Y, para):
    test_portion = para
    if test_portion > 1 or test_portion < 0:
        print('Warning! The input test_portion is not in range[0,1], set the default test_portion as 0.25')
        testportion = 0.25
    N = 1

    X_train, X_test, y_train, y_test = skmdls.train_test_split(X, Y, test_size=test_portion, random_state=int(GVal.getPARA('random_seed_PARA')))
    return [X_train], [X_test], [y_train], [y_test], N


def datasetSeparation(X, Y, split_type, para):

    ###################################
    # Main  #############################
    ###################################
    split_type_switcher = {
        1:  caseLOO,
        2:  caseLPO,
        3:  caseKFold,
        4:  caseStratifiedKFold,
        5:  caseRawsplit
    }

    return split_type_switcher[split_type](X, Y, para)
