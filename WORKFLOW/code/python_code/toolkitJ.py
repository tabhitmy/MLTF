import numpy
import numpy as np
import copy

# Update Information:
# July 6th, 2017, Gao
# Fix a bug in cell2dmatlab_jsp


###################################################
############# [cell2dmatlab_jsp] #######################
###################################################


def cell2dmatlab_make(dims, n, init_value):
    # Subfunction for cell2dmatlab_jsp
    if n == 1:
        return [init_value for i in range(dims[n - 1])]
    else:
        # Iteration to create the list structure with demanded dimension
        return [cell2dmatlab_make(dims, n - 1, init_value) for i in range(dims[n - 1])]


def cell2dmatlab_deepcopy(n, x, y, dims):
    # Subfunction for cell2dmatlab_jsp
    # Create a same size list. Do the deepcopy.
    global ndig
    if n == 1:
        for j in range(dims[n - 1]):
            x[j] = copy.deepcopy(y[j])
    else:
        ndig -= 1
        for i in range(dims[ndig]):
            cell2dmatlab_deepcopy(n - 1, x[i], y[i], dims)
    return x


def cell2dmatlab_jsp(dims, n, init_value):
    # To create the cell data structure. Same like the cell function in Matlab
    # x = cell2dmatlab_jsp(dims, n, init_value)
    # Inputs:
    #       dims, the length on each dimension of the cell. its size is value n
    #       n ,  the dimension of the demanding cell
    #       init_value, the initial value put in the list cell.  can be '[ ]'
    #
    # Author: Jsp GAO,  # Date:  June 21th,2017
    if n == 2:
        dims[0], dims[1] = dims[1], dims[0]
    global ndig
    x = cell2dmatlab_make(dims, n, [])
    y = cell2dmatlab_make(dims, n, init_value)

    ndig = n
    x = cell2dmatlab_deepcopy(n, x, y, dims)

    return x


###################################################
############# [str2num] ############################
###################################################

def str2num(x):
    if type(x) == str or type(x) == numpy.str_:
        try:
            yr = float(x)
            if len(str(yr)) == len(x):
                return yr
            else:
                return int(x)
        except ValueError:
            return x
    else:
        return x
###################################################
############# [firfilter_jsp] ############################
###################################################


def firfilter_jsp(x, h):
    # FIR filter function. Only for real number fir.
    # The output array has the same size with the input [x]
    # Applicable for reuse
    # Author: Jsp GAO  #Date: June 5th,2017

    hlen = len(h)
    xlen = len(x)
    y = np.zeros((xlen + hlen - 1, 1))
    z = np.zeros((xlen, 1))

    if hlen > xlen:
        print('ERROR! Filter is longer than input signal')

    for i in range(hlen):
        y[i] = sum(x[:i] * h[hlen - i:hlen])

    for i in range(hlen, xlen):
        y[i] = sum(x[i - hlen:i] * h)

    for i in range(xlen + 1, xlen + hlen - 1):
        y[i] = sum(x[i - hlen:xlen] * h[:hlen - i + xlen])

    for i in range(xlen):
        z[i] = y[i + int(hlen / 2) + 1]

    return z

###################################################
############# [mean_jsp] #############################
###################################################


def mean_jsp(x):
    # Calculate mean of a array
    # Applicable for reuse
    # Author: Jsp GAO  # Date: June 6th,2017
    y = 0
    for i in x:
        y = y + i

    y = y / len(x)
    return y


###################################################
############# [removeDir] ############################
###################################################

def removeDir(dirPath):
    # Remove the input filepath, even it is not empty
    # Applicable for reuse
    # Author: Jsp GAO  # Date: June 21st, 2017
    if not os.path.isdir(dirPath):
        print('Error! The input of removeDir is not a path! ')
        return
    # Get the file list in the path
    files = os.listdir(dirPath)
    for file in files:
        filePath = os.path.join(dirPath, file)
        if os.path.isfile(filePath):
            # do remove when it is a file
            os.remove(filePath)
        elif os.path.isdir(filePath):
            # Use iteration if it is still a path.
            removeDir(filePath)
    os.rmdir(dirPath)
