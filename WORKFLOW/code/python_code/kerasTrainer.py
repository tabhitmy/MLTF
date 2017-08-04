# kerasTrainer

import numpy as np
import GVal


from toolkitJ import cell2dmatlab_jsp

from trainerSubFunc_NFDA_J import *
###################################
# Classifier Subfunction ################
###################################


def F1score(y_true, y_pred):
    not_y_pred = np.logical_not(y_pred)
    y_int1 = y_true * y_pred
    y_int0 = np.logical_not(y_true) * not_y_pred
    TP = np.sum(y_pred * y_int1)
    FP = np.sum(y_pred) - TP
    TN = np.sum(not_y_pred * y_int0)
    FN = np.sum(not_y_pred) - TN

    P = np.float(TP) / (TP + FP)
    R = np.float(TP) / (TP + FN)
    beta = 0.5
    return (1 + beta**2) * (P * R) / ((beta**2) * P + R)


def sequentialNN(X_tra, y_tra, X_val, y_val, index_no, classifier_num):
    import keras.models as krmodels
    import keras.layers as krlayers
    import keras.optimizers as kroptimizers
    import keras.utils.np_utils as krnp_utils

    y_tra, X_tra, y_val, X_val, weights = dataRegulationSKL(y_tra, X_tra, y_val, X_val, index_no)
    print(X_tra.shape)
    y_tra = y_tra.reshape(-1, 1)
    y_tra = krnp_utils.to_categorical(y_tra)
    print(y_tra.shape)
    # Construct model
    clf = krmodels.Sequential()
    clf.add(krlayers.Dense(64, input_dim=10))
    clf.add(krlayers.Activation("relu"))
    clf.add(krlayers.Dense(2))
    clf.add(krlayers.Activation("softmax"))
    sgd = kroptimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    clf.fit(X_tra, y_tra, epochs=20, batch_size=16)
    Z_raw = clf.predict(X_val)

    Z = np.zeros(y_val.shape)
    print(Z_raw[:, 1].shape)
    print(np.nonzero(Z_raw[:, 1] > Z_raw[:, 0])[0])
    Z[np.nonzero(Z_raw[:, 1] > Z_raw[:, 0])[0]] = 1
    print(sum(Z))
    print(Z.shape)
    FRAP = calculateFRAP(Z, y_val)
    score = clf.evaluate(X_tra, y_tra, batch_size=16)
    return clf, score, FRAP
    # return processLearning(clf, X_tra, y_tra, X_val, y_val)

###################################
# Main  #############################
###################################


def kerasTrainer(classifier_num, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, path):

    feature_index = GVal.getPARA('feature_index_PARA')
    X, y, X_valid, y_valid, index_no = dataSetPreparation(feature_index, X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw)

    classifier_list = {
        11: [sequentialNN, 'Sequential Frame NN'],
        12: [sequentialNN, 'Sequential Frame NN']
    }

    clf_cache = {
        11: cell2dmatlab_jsp([1], 1, []),
        12: cell2dmatlab_jsp([1], 1, []),
    }

    print('### With model: [' + classifier_list[classifier_num][1] + ']')
    print('######## [Predicting ... ] ########')

    # Loading model to do the classification
    clf, score, FRAP = classifier_list[int(str(classifier_num)[0:2])][0](X, y, X_valid, y_valid, index_no, classifier_num)
    clf_cache[classifier_num] = clf
    # return clf,score,FRAP
    return classifier_list[classifier_num][1], score, FRAP
