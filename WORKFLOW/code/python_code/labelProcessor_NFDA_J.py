# labelProcessor_NFDA_J.py
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/cjkuni-ukai/ukai.ttc")  # 图片显示中文字体
mpl.use('Agg')
import matplotlib.pyplot as plt


import GVal


def processor0(X, Y, process_text):
    # Remain the current label (Do nothing)
    return X, Y


def processor1(X, Y, process_text):

    # Get target index
    target_raw_index = np.nonzero(np.logical_and((Y[:, 0] == 2), (Y[:, 1] == 1)))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    # Check the empty index list.
    if len(target_raw_index) > 0:
        # Do the process
        Y[target_raw_index, 0] = 1

    XX = X
    YY = Y

    return XX, YY


def processor2(X, Y, process_text):

    # Get target index
    target_raw_index = np.nonzero(np.logical_and((Y[:, 0] == 1), (Y[:, 1] == 1)))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    # Check the empty index list.
    if len(target_raw_index) > 0:
        # Do the process
        Y[target_raw_index, 0] = 2

    XX = X
    YY = Y

    return XX, YY


def processor3(X, Y, process_text):

    # Get target index
    target_raw_index = np.nonzero(np.logical_and((Y[:, 0] == 1), (Y[:, 1] == 1)))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    if len(target_raw_index) > 0:
        # Do the process
        XX = np.delete(X, target_raw_index, axis=0)
        YY = np.delete(Y, target_raw_index, axis=0)
    else:
        XX = X
        YY = Y
    return XX, YY


def processor4(X, Y, process_text):

    # Get target index
    target_raw_index = np.nonzero(np.logical_and((Y[:, 0] == 2), (Y[:, 1] == 1)))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    if len(target_raw_index) > 0:
        # Do the process
        XX = np.delete(X, target_raw_index, axis=0)
        YY = np.delete(Y, target_raw_index, axis=0)
    else:
        XX = X
        YY = Y
    return XX, YY


def processor5(X, Y, process_text):
    # Delete all the not confident frame
    # Get target index
    target_raw_index = np.nonzero(np.logical_or(np.logical_and((Y[:, 0] == 1), (Y[:, 1] == 1)),  np.logical_and((Y[:, 0] == 2), (Y[:, 1] == 1))))[0]

    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    if len(target_raw_index) > 0:
        # Do the process
        XX = np.delete(X, target_raw_index, axis=0)
        YY = np.delete(Y, target_raw_index, axis=0)
    else:
        XX = X
        YY = Y
    return XX, YY


def processor6(X, Y, process_text):
    # Simply Delete the noise frame and remain the noblink frame
    # Get target index
    target_raw_index = np.nonzero((Y[:, 2] > 0))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    if len(target_raw_index) > 0:
        # Do the process
        XX = np.delete(X, target_raw_index, axis=0)
        YY = np.delete(Y, target_raw_index, axis=0)
    else:
        XX = X
        YY = Y
    return XX, YY


def processor7(X, Y, process_text):
    # Get target index
    target_raw_index = np.nonzero(np.logical_or((Y[:, 2] > 0), (Y[:, 3] == 0)))[0]
    print('Abnormal Frame Amount: ' + str(len(target_raw_index)) + ' || Process method: ' + process_text)
    if len(target_raw_index) > 0:
        # Do the process
        XX = np.delete(X, target_raw_index, axis=0)
        YY = np.delete(Y, target_raw_index, axis=0)
    else:
        XX = X
        YY = Y
    return XX, YY

#####################################
#### [ MAIN ] ##########################
#####################################


def labelProcessor(X0, Y0):
    # labelProcessor is working on processing the noconfident frame problem
    # If a confidential information is attached with the label. This function can be applied to process.
    # There several process methods below. Please add any customized processor in any existed or new switcher and write the funtion above in this file

    path = GVal.getPARA('path_PARA')
    process_code = GVal.getPARA('process_code_PARA')

    # h = plt.figure(num=1, figsize=(17, 9.3))
    # plt.subplot(211)
    # plt.plot(Y0[:, 2], label='Noise Label ')
    # plt.plot(Y0[:, 1], '*', label='Confident')
    # plt.plot(Y0[:, 0], label='Class Label')
    # plt.legend()

    noconfident_frame_process_switcher = {
        0: [processor0, 'Do nothing'],
        1: [processor1, 'Change label 8 to 7'],
        2: [processor2, 'Change label 7 to 8'],
        3: [processor3, 'Delete label 7'],
        4: [processor4, 'Delete label 8'],
        5: [processor5, 'Deleta label 7&8']
    }
    noconfident_frame_process = GVal.getPARA('noconfident_frame_process_PARA')
    print('######## [ Abnormal Frame Processing ] ######## ')
    print('### Current Data Size: [X]--' + str(X0.shape) + ' [Y]--' + str(Y0.shape))
    X1, Y1 = noconfident_frame_process_switcher[noconfident_frame_process][0](X0, Y0, noconfident_frame_process_switcher[noconfident_frame_process][1])

    noise_frame_process_switcher = {
        0: [processor0, 'Do nothing'],
        1: [processor6, 'Delete noise frame'],
        2: [processor7, 'Delete noise frame and noblink frame'],
    }

    noise_frame_process = GVal.getPARA('noise_frame_process_PARA')

    print('### Current Data Size: [X]--' + str(X1.shape) + ' [Y]--' + str(Y1.shape))
    X_out, Y_out = noise_frame_process_switcher[noise_frame_process][0](X1, Y1, noise_frame_process_switcher[noise_frame_process][1])
    print('### Current Data Size: [X]--' + str(X_out.shape) + ' [Y]--' + str(Y_out.shape))

    # plt.subplot(212)
    # plt.plot(Y_out[:, 2], label='Noise Label ')
    # plt.plot(Y_out[:, 1], '*', label='Confident')
    # plt.plot(Y_out[:, 0], label='Class Label')
    # plt.legend()
    # plt.show()
    # plt.savefig((path['work_path'] + 'AbnormalFrameProcess_' + str(process_code) + '.png'))
    # print('### The label processing result comprison figure is saved in workpath!')
    return X_out, Y_out
