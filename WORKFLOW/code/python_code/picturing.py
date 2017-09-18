# picturing
# ABANDONDED.
import sys
import os
import numpy as np
import glob
import math
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

from toolkitJ import cell2dmatlab_jsp
with open('res.pickle', 'rb') as f:
    res = pickle.load(f)

print(res)

print(len(res))
#
L = len(res)

ft_size = 24


xlbl = cell2dmatlab_jsp([L], 1, [])
y = np.zeros((6, L))
for i in range(L):
    xlbl[i] = res[i][1]
    for j in range(6):
        y[j][i] = res[i][3][j]

xlbl = ['LSVM', 'LDA', 'QDA', 'NB', 'ADAB', 'LRC', 'DT', 'RF']
ylbl = ['P(Precision)', 'A(Accuracy)', 'R(Recall)', 'MA(Missing Alert)', 'FA(False Alert)', 'F1(F1 score)']
x = np.arange(1, 9)
h = plt.figure(num=str(j), figsize=(17, 9.3))
ax = plt.gca()
port = 0.1
ytick = np.arange(0, 1, 0.2)
colorlist = ['blue', 'green', 'yellow', 'yellowgreen', 'purple', 'red']
for j in range(6):
    # plt.subplot(3, 2, j + 1)
    delt = port * j + 0.01 * j
    plt.bar(x - 0.3 + delt, y[j], width=port, facecolor=colorlist[j], label=ylbl[j])

    plt.legend(mode="expand", loc=2, fontsize=ft_size)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbl, fontproperties=zhfont, fontsize=ft_size)
    ax.set_yticklabels(ytick, fontsize=ft_size)
    # plt.xlabel('Classifiers')
    plt.ylabel('scores', fontsize=ft_size)
    # plt.title(ylbl[j])
    plt.ylim((0, 1))
    plt.show()
    plt.savefig('/home/GaoMY/EXECUTION/NFDA/code/python_backup/pic/e.png')


h2 = plt.figure(num=str(j), figsize=(17, 9.3))


for j in range(6):
    plt.subplot(3, 2, j + 1)
    ax = plt.gca()
    plt.bar(x, y[j], label=ylbl[j])
    plt.legend(loc='best')
    ax.set_xticks(x)
    if j > 3:
        ax.set_xticklabels(xlbl, fontproperties=zhfont, fontsize=ft_size)
    else:
        ax.set_xticklabels([], fontproperties=zhfont, fontsize=ft_size)

    ax.set_yticklabels(ytick, fontsize=ft_size)
    # plt.xlabel('Classifiers')
    plt.ylabel('scores', fontsize=ft_size)
    plt.title(ylbl[j], fontsize=ft_size)
    plt.ylim((0, 1))
    plt.show()
    plt.savefig('/home/GaoMY/EXECUTION/NFDA/code/python_backup/pic/SPR.png')
