import os
import numpy as np
import GVal
import copy
from toolkitJ import cell2dmatlab_jsp
from controlPanelSubFunc_NFDA_J import *


def controlPanel(username, stMode):
    # stMode is use for feature Development. When it set to 1. Keep all the parameters .
    FLAG = controlPanel_admin(username)
    if username != 'GaoMY' or stMode == 1:
        # 规范化参数，请勿修改
        GVal.initLoopPARA('cleanup', [0])

        # 所有参数的参考输入关注使用手册，或者controlPanelSubFunc_NFDA_J同参数定义处，有相关详细注释
        # 参数一，数据读取序列
        GVal.setPARA('recording_index_list_PARA', [np.arange(123, 177, 1)])
        # 参数二，分类器
        GVal.setPARA('classifier_list_PARA', [23])

        # 参数三，分类器循环参数
        if len(GVal.getPARA('classifier_list_PARA')) < 2:
            GVal.initLoopPARA(0000, [])

        # 参数四， FLAG
        # 数据读入处理
        FLAG['data_prepare_flag'] = 1
        # [firstTimeConstruction], when many loops are processing in one execution, the construcion of the label and feature matrix should be done only one time. This is initializing this.
        GVal.setPARA('firstTimeConstruction', FLAG['data_prepare_flag'])

        # 中间数据存储
        # [save_flag] to save the created label and feature in this processing. No matter these label and feature and created or loaded
        # 1- save  0 -not save
        FLAG['save_flag'] = 1

        # 结果绘图
        # [plotting_flag] to control whether to plot the result. 1 - plot, 0 -not plot
        # This flag will soon be deprecated since there will be various plotting tasks and more detailed flag will then be applied.
        FLAG['plotting_flag'] = 1

        # 参数五，输入特征
        # Select several features from the online feature file
        # [online_fea_selectindex_PARA], defined by an array, it contains the colomn number of each feature in the created feature matrix
        GVal.setPARA('online_fea_selectindex_PARA', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]))
        # GVal.setPARA('online_fea_selectindex_PARA', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23]))
        # GVal.setPARA('online_fea_selectindex_PARA', np.array([11, 12, 16, 17]))
        #GVal.setPARA('online_fea_selectindex_PARA', np.array([16, 17]))

        # [online_fea_name_PARA] is the name for all those features in the created feature matrix. Breif info is attached in the comments below.
        GVal.setPARA('online_fea_name_PARA', [['眨眼时间'], ['眨眼速度'], ['perclos_first'], ['闭眼时间'], ['完全闭合时间'],
                                              ['睁眼时间'], ['闭眼平均速度'], ['睁眼平均速度'], ['闭眼最大速度'], ['睁眼最大速度'],
                                              ['perclos_second'], ['perclos_third'], ['perclos_forth'],
                                              ['D2_clos'], ['D2_open'], ['D1_clos'], ['D1_open'],
                                              ['sem'], ['sem重心频率'], ['快速眼电相对功率'], ['sem合成特征'], ['眨眼合成特征'], ['综合特征'],
                                              ['状态指数'], ['噪声标记'], ['标准差'], ['眨眼数'], ['眨眼幅值'], ['眨眼极值差'],
                                              ['D2_clos'], ['D2_open'], ['D1_clos'], ['D1_open']
                                              ])

        # ['D1'], ['D2'], ['Add3'], ['Add4'], ['Add5'],
        # [feature_index_PARA], serial number for the online_fea_selectindex.
        GVal.setPARA('feature_index_PARA', np.arange(0, len(GVal.getPARA('online_fea_selectindex_PARA'))))  # THINK TO RETIRE THIS LINE
        GVal.setPARA('screen_fea_list_PARA', np.array([15, 16, 17]))
    return FLAG, processCodeEncoder()
