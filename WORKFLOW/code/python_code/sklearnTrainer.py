# sklearnTrainer
import numpy
import numpy as np
import copy

from toolkitJ import cell2dmatlab_jsp

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')


import sklearn.model_selection as skmdls
import sklearn.ensemble as skemb
import sklearn.tree as sktree
import sklearn.linear_model as sklinmdl
import sklearn.discriminant_analysis as skdisa
import sklearn.svm as sksvm
import sklearn.naive_bayes as sknb

import GVal
from controlPanelSubFunc_NFDA_J import dVM

from trainerSubFunc_NFDA_J import *


###################################
# Classifier Subfunction ################
###################################


def adaboost(X_tra, y_tra, X_val, y_val, index_no, classifier_num):
    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    weakClf_list = {
        27: 'decisionTree',
        271: 'decisionTree'
    }

    clf = skemb.AdaBoostClassifier(sktree.DecisionTreeClassifier(max_depth=2, min_samples_split=30, min_samples_leaf=5),
                                   algorithm='SAMME', n_estimators=50, learning_rate=0.7)

    clf.fit(X_tra, y_tra)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def lda(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skdisa.LinearDiscriminantAnalysis(solver=dVM[2300][0], n_component=dVM[2303][2])

    clf.fit(X_tra, y_tra)
    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def qda(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skdisa.QuadraticDiscriminantAnalysis()

    clf.fit(X_tra, y_tra)
    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def naiveBayes(X_tra, y_tra, X_val, y_val, index_no, classifier_num):
    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)
    clfname_list = {
        25: sknb.GaussianNB,
        251: sknb.GaussianNB,
        252: sknb.MultinomialNB,
        253: sknb.BernoulliNB,
    }

    clf = clfname_list[classifier_num]()
    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def svmKernel(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    kernelname_list = {
        22: 'rbf',
        221: 'rbf',
        222: 'poly',
        223: 'sigmoid',
        224: 'precompute'
    }
    kernelname = kernelname_list[classifier_num]

    clf = sksvm.SVC(C=0.1, kernel=kernelname, degree=3, gamma=0.7)

    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)
##


def svmLinear(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = sksvm.LinearSVC(penalty=dVM[2100][2], loss=dVM[2101][2],
                          dual=dVM[2102][2], tol=dVM[2103][2], C=dVM[2104][2])
    # clf = sksvm.LinearSVC()
    clf.fit(X_tra, y_tra, sample_weight=weights)

    cx = clf.coef_[0]
    clfc = np.around(cx, decimals=2)
    print('### Feature coefficient with L penalty: ' + str(clfc))

    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def linearRegression(X_tra, y_tra, X_val, y_val, index_no, classifier_num):
    # Not Applicable at this moment
    # weights = Y_train_raw[:, 0]
    # weights[np.nonzero(weights == 0)[0]] = 1
    # weights = weights / 7

    # y_tra, y_val, X_val = dataRegulation(y_tra, y_val, X_val, index_no)

    # clf = sklinmdl.LinearRegression()
    # clf.fit(X_tra, y_tra, sample_weight=weights)
    # score = clf.score(X_tra, y_tra, sample_weight=weights)
    # print()
    # Z = clf.predict(X_val)
    # print(Z.shape)
    # TP = np.nonzero(np.logical_and(Z == 1, y_val == 1))[0]
    # print(TP)
    # print(TP.shape)

    # print(max(weights))
    # print(min(weights))

    return clf, score, FRAP


##

def sgdClassifier(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = sklinmdl.SGDClassifier(loss='hinge', penalty='l2', alpha=0.1)

    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)


##
def logiRegression(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = sklinmdl.LogisticRegression(penalty=dVM[3000][2], dual=dVM[3001][2], tol=dVM[3002][2],
                                      C=dVM[3003][2], random_state=dVM[3007][2],
                                      solver=dVM[3008][2], max_iter=dVM[3009][2])

    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)


##
def decisionTree(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = sktree.DecisionTreeClassifier(criterion=dVM[3100][2], splitter=dVM[3101][2],
                                        max_depth=dVM[3102][2], min_samples_split=dVM[3103][2],
                                        min_samples_leaf=dVM[3104][2], max_features=dVM[3106][2],
                                        random_state=dVM[3107][2])

    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)


##
def randomForest(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skemb.RandomForestClassifier(n_estimators=dVM[3200][2],
                                       criterion=dVM[3201][2], max_features=dVM[3202][2],
                                       max_depth=dVM[3203][2], min_samples_split=dVM[3204][2],
                                       min_samples_leaf=dVM[3205][2], min_weight_fraction_leaf=dVM[3206][2],
                                       random_state=dVM[3213][2])
    # GVal.show('dVM_PARA')
    clf.fit(X_tra, y_tra, sample_weight=weights)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)

###################################
# Main  #############################
###################################


def sklearnTrainer(classifier_num, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, path):

    feature_index = GVal.getPARA('feature_index_PARA')
    X, y, X_valid, y_valid, index_no = dataSetPreparation(feature_index, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw)

    classifier_list = {
        21: [svmLinear, 'Linear SVM', []],
        22: [svmKernel, 'Kernel SVM (Default:rbf)'],
        221: [svmKernel, 'Kernel SVM (rbf)'],
        222: [svmKernel, 'Kernel SVM (poly)'],
        223: [svmKernel, 'Kernel SVM (sigmoid)'],
        224: [svmKernel, 'Kernel SVM (precompute)'],
        23: [lda, 'LDA'],
        24: [qda, 'QDA'],
        25: [naiveBayes, 'Naive Bayes (Default: Gaussian)'],
        251: [naiveBayes, 'Naive Bayes (Guassian)'],
        252: [naiveBayes, 'Naive Bayes (Multinominal)'],
        253: [naiveBayes, 'Naive Bayes (Bernoulli)'],
        # 26: neuralNetwork,
        27: [adaboost, 'Adaboost'],
        271: [adaboost, 'Adaboost(WC:DecisionTree)'],
        # 28: [linearRegression, 'Linear Regression'],
        29: [sgdClassifier, 'SGD Classifier'],
        30: [logiRegression, 'Logistic Regression'],
        31: [decisionTree, 'Decision Tree'],
        32: [randomForest, 'Random Forest']
    }

    # classifier serial code: [[model], [training score], [predicting rate]]
    clf_cache = {
        21: cell2dmatlab_jsp([1], 1, []),
        22: cell2dmatlab_jsp([1], 1, []),
        221: cell2dmatlab_jsp([1], 1, []),
        222: cell2dmatlab_jsp([1], 1, []),
        223: cell2dmatlab_jsp([1], 1, []),
        224: cell2dmatlab_jsp([1], 1, []),
        23: cell2dmatlab_jsp([1], 1, []),
        24: cell2dmatlab_jsp([1], 1, []),
        25: cell2dmatlab_jsp([1], 1, []),
        251: cell2dmatlab_jsp([1], 1, []),
        252: cell2dmatlab_jsp([1], 1, []),
        253: cell2dmatlab_jsp([1], 1, []),

        27: cell2dmatlab_jsp([1], 1, []),
        271: cell2dmatlab_jsp([1], 1, []),
        28: cell2dmatlab_jsp([1], 1, []),
        29: cell2dmatlab_jsp([1], 1, []),
        30: cell2dmatlab_jsp([1], 1, []),
        31: cell2dmatlab_jsp([1], 1, []),
        32: cell2dmatlab_jsp([1], 1, [])
    }

    print('### With model: [' + classifier_list[classifier_num][1] + ']')

    # Loading model to do the classification
    clf, score, FRAP = classifier_list[int(str(classifier_num)[0:2])][0](X, y, X_valid, y_valid, index_no, classifier_num)
    clf_cache[classifier_num] = clf
    # return clf,score,FRAP
    return classifier_list[classifier_num][1], score, FRAP
