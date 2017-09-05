# from __future__ import print_function

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC

# print(__doc__)
import numpy as np
from operator import itemgetter


def zipSort(mas, slv):
    # sort the slv with the mas ascending .
    goodsortflag = 1
    for i in range(len(mas) - 2):
        if (mas[i + 2] - mas[i + 1]) * (mas[i + 1] - mas[i]) < 0:
            goodsortflag = 0
            break

    if goodsortflag == 1:
        return mas, slv
    else:
        k = 0
        for slvline in slv:
            zipdict = {}
            for i in range(len(mas)):
                zipdict[mas[i]] = slvline[i]
            zipdict_res = sorted(zipdict.items())

            xdata = np.zeros([1, len(zipdict_res)])
            ydata_temp = np.zeros([1, len(zipdict_res)])
            for j in range(len(zipdict_res)):
                xdata[0][j] = zipdict_res[j][0]
                ydata_temp[0][j] = zipdict_res[j][1]

            if k == 0:
                ydata = ydata_temp
            else:
                ydata = np.concatenate((ydata, ydata_temp))
            k += 1
        xdata = xdata.tolist()
        ydata = ydata.tolist()
        return xdata, ydata


x = [0.1, 1.3, 0.5, 1.2]
y = [[12, 6, 3, 7], [0.01, 0.02, 0.03, 0.04]]


xdata, ydata = zipSort(x, y)
