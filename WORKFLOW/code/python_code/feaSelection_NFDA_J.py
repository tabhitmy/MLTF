# feaSelection_J.py

import numpy as np
import sklearn.manifold as skmanifold
import sklearn.decomposition as skdecomp
import GVal

##


def tSNE(X, para):
    tsne = skmanifold.TSNE(n_components=para, init='pca', random_state=GVal.getPARA('random_seed_PARA'))
    X_tsne = tsne.fit_transform(X)
    return X_tsne, tsne

##


def normalPCA(X, para):
    pca = skdecomp.PCA(n_components=para, svd_solver='arpack')
    X_PCA = pca.fit_transform(X)
    return X_PCA, pca


def feaSelection(X):
    feaSelectionStrategy = GVal.getPARA('feaSelectionStrategy')
    nComponent = GVal.getPARA('nComponent_PARA')
    feaSelectionStrategies = {
        1: tSNE,
        2: normalPCA
    }

    X_fss, feaS_mdl = feaSelectionStrategies[feaSelectionStrategy](X, nComponent)
    GVal.setPARA('feature_index_PARA', np.arange(0, GVal.getPARA('nComponent_PARA')))
    return X_fss, feaS_mdl
