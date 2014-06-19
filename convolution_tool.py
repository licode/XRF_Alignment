"""
basic one-D or two-D convolution function
"""
import numpy as np

def conv1D_gauss(inputdata, stdv):
    """
    1D convolution with gaussian function.
    """
    inputdata = np.array(inputdata)
    lenv = len(inputdata)
    
    # xrange is from (-1,1)
    xv = np.linspace(-1.0, 1.0, lenv)

    gfun = 1./np.sqrt(2*np.pi)/stdv*np.exp(-xv**2/2/stdv**2)
    
    outv = np.convolve(inputdata, gfun, 'same')
    
    return outv


def conv2D_1D_gauss(inputdata, stdv):
    """
    2D convolution based on conv1D_gauss function.
    convolve each column of the 2D array with a gauss function
    """
    inputdata = np.array(inputdata)
    
    input_s = inputdata.shape
    
    lenv = input_s[0]
    
    outdata = np.zeros(input_s)
    
    # xrange is from (-1,1)
    xv = np.linspace(-1.0, 1.0, lenv)

    gfun = 1./np.sqrt(2*np.pi)/stdv*np.exp(-xv**2/2/stdv**2)
    
    for i in range(input_s[1]):
        outdata[:,i] = np.convolve(inputdata[:,i], gfun, 'same')
    
    return outdata


