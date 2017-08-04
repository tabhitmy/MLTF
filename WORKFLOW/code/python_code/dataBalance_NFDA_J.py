# dataBalance_NFDA_J.py
import numpy as np
import sklearn.manifold as skmanifold
import sklearn.decomposition as skdecomp
import GVal
import random

##


def downSamplingNega(X, Y, para):

    target_raw_index = np.nonzero(Y[:, 0] == 0)[0]
    # random.sample(sequence, n_sample)
    # set n_sample == sequence, This makes no repeat index.
    index_seed_raw = random.sample(list(target_raw_index), len(target_raw_index))
    target_index = index_seed_raw[0: int((1 - para) * len(index_seed_raw))]

    X0 = np.delete(X, target_index, axis=0)
    Y0 = np.delete(Y, target_index, axis=0)

    return X0, Y0


def dataBalance(X, Y):
    data_balance_process = GVal.getPARA('data_balance_process_PARA')
    data_balance_para = GVal.getPARA('data_balance_para_PARA')

    data_balance_frame = {
        1: downSamplingNega,
    }

    XX, YY = data_balance_frame[data_balance_process](X, Y, data_balance_para)

    return XX, YY
