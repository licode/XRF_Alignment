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
    xv = np.arange(lenv)
    halfv = (lenv-1)*1.0/2  
    
    # xrange is from (-1,1)
    xv = (xv - halfv)/halfv

    gfun = 1./np.sqrt(2*np.pi)/stdv*np.exp(-xv**2/2/stdv**2)
    
    outv = np.convolve(inputdata, gfun, 'same')
    
    return outv


