# sklearnTrainer
import numpy
import numpy as np
import copy

from toolkitJ import cell2dmatlab_jsp

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')

import pprint

from sklearn.externals.six import StringIO
# import pydot

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
                                   algorithm=dVM[2702][2], n_estimators=dVM[2700][2], learning_rate=dVM[2701][2], random_state=dVM[2703][2])

    clf.fit(X_tra, y_tra)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def lda(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skdisa.LinearDiscriminantAnalysis(solver=dVM[2300][2], n_components=dVM[2303][2])

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
    path = GVal.getPARA('path_PARA')
    with open(path['fig_path'] + 'dtclf.dot', 'w') as f:
        f = sktree.export_graphviz(clf, out_file=f, class_names=['0', '1'])

    # sktree.export_graphviz(clf, out_file=path['fig_path'] + 'tree.dot')
    # exit()
    return processLearning(clf, X_tra, y_tra, X_val, y_val)

##


def randomForest(X_tra, y_tra, X_val, y_val, index_no, classifier_num):
    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)
    # http://blog.csdn.net/xuxiatian/article/details/54410086
    clf = skemb.RandomForestClassifier(n_estimators=dVM[3200][2],
                                       criterion=dVM[3201][2], max_features=dVM[3202][2],
                                       max_depth=dVM[3203][2], min_samples_split=dVM[3204][2],
                                       min_samples_leaf=dVM[3205][2], min_weight_fraction_leaf=dVM[3206][2],
                                       random_state=dVM[3213][2])
    # GVal.show('dVM_PARA')
    clf.fit(X_tra, y_tra, sample_weight=weights)
    # print(clf.get_params())
    # print(clf)
    path = GVal.getPARA('path_PARA')
    i_tree = 0
    for tree_in_forest in clf.estimators_:
        with open(path['fig_path'] + '/RF/tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = sktree.export_graphviz(tree_in_forest, out_file=my_file, class_names=['0', '1'])
        i_tree = i_tree + 1

    return processLearning(clf, X_tra, y_tra, X_val, y_val)


def bagging(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skemb.BaggingClassifier(base_estimator=sktree.DecisionTreeClassifier(max_depth=2, min_samples_split=30, min_samples_leaf=5),
                                  n_estimators=dVM[3300][2], max_samples=dVM[3301][2], max_features=dVM[3302][2],
                                  bootstrap=dVM[3303][2],  random_state=dVM[3308][2])
    clf.fit(X_tra, y_tra, sample_weight=weights)
    return processLearning(clf, X_tra, y_tra, X_val, y_val)


def voting(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    #
    classifier_list = GVal.getPARA('classifier_list_PARA')
    # dVM[3400] = ['estimators', [21, 23, 25, 30, 31], [21, 23, 25, 30, 31]]
    estims = []
    for i in range(len(dVM[3400][2])):
        clf_temp = (classifier_list[dVM[3400][2][i]][1], classifier_list[int(str(dVM[3400][2][i])[0:2])][0](X_tra, y_tra, X_val, y_val, index_no, dVM[3400][2][i])[0])
        estims.append(clf_temp)

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)
    clf = skemb.VotingClassifier(estimators=estims, voting=dVM[3401][2])
    clf.fit(X_tra, y_tra)

    return processLearning(clf, X_tra, y_tra, X_val, y_val)


def gradboost(X_tra, y_tra, X_val, y_val, index_no, classifier_num):

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)

    clf = skemb.GradientBoostingClassifier(loss=dVM[3500][2], learning_rate=dVM[3501][2],
                                           n_estimators=dVM[3502][2], max_depth=dVM[3503][2], criterion=dVM[3504][2],
                                           min_samples_split=dVM[3505][2], min_samples_leaf=dVM[3506][2],
                                           subsample=dVM[3508][2], random_state=dVM[3515][2])
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
        32: [randomForest, 'Random Forest'],
        33: [bagging, 'bagging with DT'],
        34: [voting, 'Voter'],
        35: [gradboost, 'Gradient Tree Boosting']
    }
    GVal.setPARA('classifier_list_cache', classifier_list)
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
        32: cell2dmatlab_jsp([1], 1, []),
        33: cell2dmatlab_jsp([1], 1, []),
        34: cell2dmatlab_jsp([1], 1, [])
    }

    print('### With model: [' + classifier_list[classifier_num][1] + ']')

    # Loading model to do the classification
    clf, score, FRAP = classifier_list[int(str(classifier_num)[0:2])][0](X, y, X_valid, y_valid, index_no, classifier_num)
    clf_cache[classifier_num] = clf
    # return clf,score,FRAP
    clf_info = cell2dmatlab_jsp([3], 1, [])
    clf_info[0] = classifier_num
    clf_info[1] = classifier_list[classifier_num][1]
    clf_info[2] = clf
    return clf_info, score, FRAP
