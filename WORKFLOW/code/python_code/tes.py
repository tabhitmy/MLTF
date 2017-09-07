# from __future__ import print_function

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC

# print(__doc__)
import numpy
import numpy as np


def str2num(x):
    if type(x) == str or type(x) == numpy.str_:
        try:
            yr = float(x)

            if len(str(yr)) == len(x):
                print('# x is a float')
                return yr
            else:
                print('# x is a integer')
                return int(x)
        except ValueError:
            print('# x is not a number. It is a string.')
            return x
    else:
        print('# pass')
        return x


x = 12
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))

x = '12.00'
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))


x = '12.124'
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))


x = 12.00
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))


x = '12.450086000'
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))


x = 'cxv'
print('-' * 100)
print(type(str2num(x)))
print(str2num(x))
