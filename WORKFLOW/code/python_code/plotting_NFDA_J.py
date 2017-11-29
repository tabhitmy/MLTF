import numpy
import numpy as np


import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
mpl.get_cachedir()
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import scipy.io as sio

import pandas as pd
import seaborn as sns
import GVal
from toolkitJ import cell2dmatlab_jsp
import copy


def featurePlotting(label_all, online_fea_all, processCode):

    nega_index_raw = np.nonzero(label_all[0] == 0)[0]
    posi1_index_raw = np.nonzero(label_all[0] == 1)[0]
    posi2_index_raw = np.nonzero(label_all[0] == 2)[0]
    posi3_index_raw = np.nonzero(label_all[0] == 3)[0]

    # [nega, posi1, posi2, posi3]
    colorlist1 = ['#000000', '#000000', '#0d75f8', '#e50000']

    nega_index = nega_index_raw
    posi1_index = posi1_index_raw
    posi2_index = posi2_index_raw
    posi3_index = posi3_index_raw

    screen_fea_list = GVal.getPARA('screen_fea_list_PARA')
    online_fea_name = GVal.getPARA('online_fea_name_PARA')
    print(len(nega_index), len(posi1_index), len(posi2_index), len(posi3_index))
    pic_num = 1
    for fea1_n in range(len(screen_fea_list)):
        fea1_num = screen_fea_list[fea1_n]
        for fea2_n in range(fea1_n + 1, len(screen_fea_list)):
            fea2_num = screen_fea_list[fea2_n]
            figcode = int(1e13 * pic_num + processCode)
            h = plt.figure(num=figcode, figsize=(17, 9.3))
            plt.subplot(131)
            plt.plot(online_fea_all[fea1_n][nega_index], online_fea_all[fea2_n][nega_index], 'o', color=colorlist1[0], label='No Fatigue')
            plt.plot(online_fea_all[fea1_n][posi1_index], online_fea_all[fea2_n][posi1_index], '.', color=colorlist1[1], label='Fatigue LV7')
            plt.plot(online_fea_all[fea1_n][posi2_index], online_fea_all[fea2_n][posi2_index], 'o', color=colorlist1[2], label='Fatigue LV8')
            plt.plot(online_fea_all[fea1_n][posi3_index], online_fea_all[fea2_n][posi3_index], 'o', color=colorlist1[3], label='Fatigure LV9')
            plt.legend(loc='best', prop=zhfont)
            plt.xlabel(online_fea_name[fea1_num], FontProperties=zhfont)
            plt.ylabel(online_fea_name[fea2_num], FontProperties=zhfont)
            plt.title('Figure #' + str(figcode))
            plt.show()
            plt.close(h)
            # plt.savefig((work_path + '/pic/Figure' + str(pic_num) + '.png'))
            # print('Picture' + str(pic_num) + 'Saved!')
            # plt.close(h)
            pic_num += 1
    return 0


def resultPlotting(FRAP, processCode):
    ftsize = 20
    path = GVal.getPARA('path_PARA')
    X_tra = GVal.getPARA('X_tra_res_PARA')
    X_val = GVal.getPARA('X_val_res_PARA')

    y_tra = GVal.getPARA('y_tra_res_PARA')
    y_val = GVal.getPARA('y_val_res_PARA')

    Z_tra = GVal.getPARA('Z_tra_res_PARA')
    Z = GVal.getPARA('Z_res_PARA')
    p0_index_tra = np.nonzero(y_tra == 0)[0]
    p1_index_tra = np.nonzero(y_tra == 1)[0]
    p2_index_tra = np.nonzero(y_tra == 2)[0]
    p3_index_tra = np.nonzero(y_tra == 3)[0]
    p23_index_tra = np.nonzero(y_tra > 1)[0]

    p0_index_val = np.nonzero(y_val == 0)[0]
    p1_index_val = np.nonzero(y_val == 1)[0]
    p2_index_val = np.nonzero(y_val == 2)[0]
    p3_index_val = np.nonzero(y_val == 3)[0]
    p23_index_val = np.nonzero(y_val > 1)[0]

    nega_index_tra = np.nonzero(Z_tra == 0)[0]
    posi_index_tra = np.nonzero(Z_tra == 1)[0]

    nega_index_val = np.nonzero(Z == 0)[0]
    posi_index_val = np.nonzero(Z == 1)[0]

    # colorlist2 = ['#929591', '#929591', '#fd3c06', '#fd3c06']
    colorlist2 = ['#0c06f7', '#029386', '#5edc1f', '#fd3c06']
    colorlist3 = ['#000000', '#ffffff']
    markerlist1 = ['^', 'D', '*', 'h', '.']

    screen_fea_list = GVal.getPARA('screen_fea_list_PARA')
    online_fea_name = GVal.getPARA('online_fea_name_PARA')
    online_fea_selectindex = GVal.getPARA('online_fea_selectindex_PARA')
    pic_num = 1
    for fea1_n in range(len(screen_fea_list)):
        fea1_num = screen_fea_list[fea1_n]
        fea1 = np.nonzero(online_fea_selectindex == fea1_num)[0][0]
        for fea2_n in range(fea1_n + 1, len(screen_fea_list)):
            fea2_num = screen_fea_list[fea2_n]
            fea2 = np.nonzero(online_fea_selectindex == fea2_num)[0][0]
            figcode = int(1e13 * pic_num + processCode)
            h = plt.figure(num=figcode, figsize=(20, 9.3))

            # Trainning set

            plt.subplot(121)
            plt.scatter(X_tra[p0_index_tra, fea1], X_tra[p0_index_tra, fea2], marker=markerlist1[0], color=colorlist2[0], label='label 0', linewidths=2)
            if int(GVal.getPARA('kick_off_no1_PARA')) not in [0, 1, 2]:
                plt.scatter(X_tra[p1_index_tra, fea1], X_tra[p1_index_tra, fea2], marker=markerlist1[1], color=colorlist2[1], label='label 1', linewidths=2)
            plt.scatter(X_tra[p2_index_tra, fea1], X_tra[p2_index_tra, fea2], marker=markerlist1[2], color=colorlist2[2], label='label 2', linewidths=2)
            plt.scatter(X_tra[p3_index_tra, fea1], X_tra[p3_index_tra, fea2], marker=markerlist1[3], color=colorlist2[3], label='label 3', linewidths=2)

            # plt.scatter(X_tra[nega_index_tra, fea1], X_tra[nega_index_tra, fea2], marker=markerlist1[4], color=colorlist3[0], label='Nega', linewidths=0.1)
            plt.scatter(X_tra[posi_index_tra, fea1], X_tra[posi_index_tra, fea2], marker=markerlist1[4], color=colorlist3[1], label='Posi', linewidths=0.1)
            plt.legend(loc='best', prop=zhfont)
            plt.xlabel(online_fea_name[fea1_num], FontProperties=zhfont, Fontsize=ftsize)
            plt.ylabel(online_fea_name[fea2_num], FontProperties=zhfont, Fontsize=ftsize)
            plt.title('Training Set \n' +
                      '[0]-' + str(len(p0_index_tra)) + ' || [1]-' + str(len(p1_index_tra)) + ' || [2]-' + str(len(p2_index_tra)) + ' || [3]-' + str(len(p3_index_tra)) + '\n' +
                      GVal.getPARA('kick_off_no1_detail')[int(GVal.getPARA('kick_off_no1_PARA'))]
                      )

            # Validation set
            plt.subplot(122)
            plt.scatter(X_val[p0_index_val, fea1], X_val[p0_index_val, fea2], marker=markerlist1[0], color=colorlist2[0], label='label 0', linewidths=2)
            if int(GVal.getPARA('kick_off_no1_PARA')) not in [0, 3, 6]:
                plt.scatter(X_val[p1_index_val, fea1], X_val[p1_index_val, fea2], marker=markerlist1[1], color=colorlist2[1], label='label 1', linewidths=2)
            plt.scatter(X_val[p2_index_val, fea1], X_val[p2_index_val, fea2], marker=markerlist1[2], color=colorlist2[2], label='label 2', linewidths=2)
            plt.scatter(X_val[p3_index_val, fea1], X_val[p3_index_val, fea2], marker=markerlist1[3], color=colorlist2[3], label='label 3', linewidths=2)

            # plt.scatter(X_val[nega_index_val, fea1], X_val[nega_index_val, fea2], marker=markerlist1[4], color=colorlist3[0], label='Nega', linewidths=0.1)
            plt.scatter(X_val[posi_index_val, fea1], X_val[posi_index_val, fea2], marker=markerlist1[4], color=colorlist3[1], label='Posi', linewidths=0.1)
            plt.legend(loc='best', prop=zhfont)
            plt.xlabel(online_fea_name[fea1_num], FontProperties=zhfont)
            plt.ylabel(online_fea_name[fea2_num], FontProperties=zhfont)
            plt.title('Validation Set \n' +
                      '[0]-' + str(len(p0_index_val)) + ' || [1]-' + str(len(p1_index_val)) + ' || [2]-' + str(len(p2_index_val)) + ' || [3]-' + str(len(p3_index_val)) + '\n' +
                      '[ P: ' + str(round(FRAP[0], 4)) + ' || A: ' + str(round(FRAP[1], 4)) + ' || R: ' + str(round(FRAP[2], 4)) + ' || MA: ' + str(round(FRAP[3], 4)) + ' || FA: ' + str(round(FRAP[4], 4)) + ' || F1: ' + str(round(FRAP[5], 4)) + ' || F' + str(GVal.getPARA('beta_PARA')) + ': ' + str(round(FRAP[6], 4))
                      )

    #         plt.subplot(223)
    #         plt.scatter(X_val[p0_index_val, fea1], X_val[p0_index_val, fea2], marker=markerlist1[0], color=colorlist2[0], label='label 0', linewidths=2)
    #         plt.subplot(224)

            plt.show()
            plt.savefig((path['fig_path'] + 'distriRes/Figure' + str(figcode) + '.png'))
            print('Picture' + str(figcode) + 'Saved!')
            plt.close(h)
            pic_num += 1
            # exit()
    return 0


def pairDistPlotting(FRAP, processCode):
    # print(sns.axes_style())
    path = GVal.getPARA('path_PARA')
    X_tra = GVal.getPARA('X_tra_res_PARA')
    X_val = GVal.getPARA('X_val_res_PARA')

    y_tra = GVal.getPARA('y_tra_res_PARA')
    y_val = GVal.getPARA('y_val_res_PARA')

    Z_tra = GVal.getPARA('Z_tra_res_PARA')
    Z = GVal.getPARA('Z_res_PARA')

    # p1_index_tra = np.nonzero(y_tra == 1)[0]
    # y_tra = np.delete(y_tra, p1_index_tra, axis=0)
    # X_tra = np.delete(X_tra, p1_index_tra, axis=0)
    p2_index_tra = np.nonzero(y_tra == 2)[0]
    y_tra[p2_index_tra] = 3

    screen_fea_list = GVal.getPARA('screen_fea_list_PARA')
    online_fea_name = GVal.getPARA('online_fea_engname_PARA')
    online_fea_selectindex = GVal.getPARA('online_fea_selectindex_PARA')
    lplb_count = 0
    columndata = cell2dmatlab_jsp([1, len(screen_fea_list) + 1], 2, [])

    for fea in screen_fea_list:
        fea_serial = np.nonzero(online_fea_selectindex == fea)[0][0]
        if lplb_count == 0:
            pairdata = X_tra[:, fea_serial].reshape(-1, 1)
        else:
            pairdata = np.concatenate((pairdata, X_tra[:, fea_serial].reshape(-1, 1)), axis=1)
        columndata[0][lplb_count] = online_fea_name[fea_serial][0]
        lplb_count += 1

    matcoef = np.corrcoef(pairdata.transpose())

    sio.savemat(path['fig_path'] + 'matcoef.mat', {'matcoef': matcoef})
    # print(matcoef)
    for line in matcoef:
        print(line)

    # GVal.setPARA('')
    pairdata = np.concatenate((pairdata, y_tra.reshape(-1, 1)), axis=1)
    columndata[0][lplb_count] = 'Label'
    pairdf = pd.DataFrame(pairdata, index=np.arange(len(y_tra)), columns=columndata)

    figcode = int(1e13 * 9 + processCode)
    h = plt.figure(num=figcode, figsize=(20, 9.3))
    plt.subplot(211)
    plt.title('Training set')
    sns.set(font=zhfont.get_name())
    g = sns.PairGrid(pairdf, vars=columndata[0][:-1], hue='Label', size=3)
    g.map_diag(plt.hist, edgecolor="w", bins=25)
    g.map_offdiag(plt.scatter, edgecolor="w", s=20)
    plt.show()
    plt.savefig((path['fig_path'] + 'FeaturePairPloting' + str(figcode) + '.png'))
    print('Picture' + str(figcode) + 'Saved!')
    plt.close(h)

    return 0


def cFRAPPlotting(res):
    L_clf = len(GVal.getPARA('classifier_list_PARA'))
    if L_clf < 2:
        print('### Warning! No enough classifiers for cFRAP Plotting, skip it.')
        return 0
    ft_size = 18
    path = GVal.getPARA('path_PARA')
    beta = GVal.getPARA('beta_PARA')
    y = np.zeros((5, L_clf))
    xlbl = cell2dmatlab_jsp([L_clf], 1, [])
    for i in range(L_clf):
        xlbl[i] = res[i][1]
        for j in range(3):
            y[j][i] = res[i][3][j]
        for jj in range(5, 7):
            y[jj - 2][i] = res[i][3][jj]

    ylbl = ['[P] Precision', '[A] Accuracy', '[R] Recall',  '[F1] score', ['[F' + str(beta) + '] score']]
    colorlist = ['#0203e2', '#6ecb3c', '#fd3c06', '#000000', '#000000']
    h = plt.figure(num=1, figsize=(17, 9.3))
    ax = plt.gca()
    port = 0.1
    ytick = np.arange(0, 1, 0.1)
    x = np.arange(1, L_clf + 1, 1)

    for j in range(5):
        delt = port * j + 0.01 * j
        plt.bar(x - 0.3 + delt, y[j], width=port, facecolor=colorlist[j], label=ylbl[j])

    plt.legend(loc=2, fontsize=ft_size)
    ax.set_xticks(x)
    ax.set_yticks(ytick)
    ax.set_xticklabels(xlbl, fontproperties=zhfont, fontsize=ft_size)
    ax.set_yticklabels(ytick, fontsize=ft_size)
    plt.ylabel('scores', fontsize=ft_size)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.show()
    plt.savefig((path['fig_path'] + 'cFRAP.png'))
    print('Picture for ' + str(L_clf) + ' Various Classifers Saved!')


def FRAPPlotting(res):
    if GVal.getPARA('loopPARA_amount_PARA') < 1:
        print('### Warning! No enough loop parameters for FRAP Plotting, skip it.')
        return 0

    path = GVal.getPARA('path_PARA')
    beta = GVal.getPARA('beta_PARA')
    dVM = GVal.getPARA('dVM_PARA')
    dVPARA_index = GVal.getPARA('dVPARA_index_PARA')
    dVPARA_value = GVal.getPARA('dVPARA_value_PARA')
    loopPARA_namecache = GVal.getPARA('loopPARA_namecache_PARA')
    L_pic = len(dVPARA_index)
    # FontSize
    ftsize = 20
    # color
    colorlist = ['#0203e2', '#6ecb3c', '#fd3c06', '#000000', '#929591']
    # marker
    markerlist = ['^', '.', 'v', '*', 'h']
    # linestyle
    linelist = ['-', '-', '-', ':', '-.']

    params = {
        'axes.labelsize': '15',
        'xtick.labelsize': '22',
        'ytick.labelsize': '22',
        'lines.linewidth': 1,
        'legend.fontsize': '15',
        # 'figure.figsize'   : '12, 9'    # set figure size
    }
    # pylab.rcParams.update(params)
    for i in range(L_pic):
        h = plt.figure(num=L_pic, figsize=(20, 9.3))

        xdata_rawraw = dVPARA_value[i]

        if type(xdata_rawraw[0]) == str or type(xdata_rawraw[0]) == numpy.str_:
            xdata_raw = np.arange(len(xdata_rawraw))
            xdata_rawraw_index = np.arange(len(xdata_rawraw))
        else:
            xdata_raw = copy.deepcopy(xdata_rawraw)
            xdata_rawraw_index = xdata_raw

        ydata_raw = cell2dmatlab_jsp([7], 1, [])

        for dV_index in dVPARA_index[i]:
            for k in range(7):
                ydata_raw[k] += [res[dV_index][3][k]]

        xdata, ydata = zipSort(xdata_raw, ydata_raw)

        plt.plot(xdata, ydata[0], marker=markerlist[0], color=colorlist[0], linestyle=linelist[0], label='[P] Precision')
        plt.plot(xdata, ydata[1], marker=markerlist[1], color=colorlist[1], linestyle=linelist[1], label='[A] Accuracy')
        plt.plot(xdata, ydata[2], marker=markerlist[2], color=colorlist[2], linestyle=linelist[2], label='[R] Recall')
        plt.plot(xdata, ydata[5], marker=markerlist[3], color=colorlist[3], linestyle=linelist[3], label='[F1] score')
        plt.plot(xdata, ydata[6], marker=markerlist[4], color=colorlist[4], linestyle=linelist[4], label=['[F' + str(beta) + '] score'])
        legend = plt.legend(loc='best', prop={'size': 18})
        # legend.get_title().set_fontsize(fontsize=50)
        # h.legend(Fontsize=ftsize)
        if type(loopPARA_namecache[i][0]) == int:
            xlabeltext = 'Classifier: [ ' + res[0][1][1] + ' ] | Parameter: [ ' + dVM[loopPARA_namecache[i][0]][0] + ' ]'
        else:
            xlabeltext = 'Classifier: [ ' + res[0][1][1] + ' ] | General Parameter: [ ' + loopPARA_namecache[i][0] + ' ]'
        plt.xticks(xdata_rawraw_index, xdata_rawraw, rotation=0)
        plt.xlabel(xlabeltext, Fontsize=ftsize)
        plt.ylabel('FRAP Value', Fontsize=ftsize)
        plt.title('Recording Envorinment: ' + GVal.getPARA('recordname'), Fontsize=ftsize)
        plt.ylim(0, 1.05)
        plt.grid()
        plt.show()
        plt.savefig((path['fig_path'] + 'FRAP_' + str(loopPARA_namecache[i][0]) + '_' + str(GVal.getPARA('process_code_PARA')) + '.png'))
        print('Picture' + str(i) + 'Saved!')
        plt.close(h)

    return 0


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
        xdata = xdata[0].tolist()
        ydata = ydata.tolist()
        return xdata, ydata
