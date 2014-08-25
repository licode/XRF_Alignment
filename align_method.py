import numpy as np
from scipy.optimize import curve_fit


def alignment_on_edge(data, padv=14, maxv=True):
    """
    Do vertical alignment based on max value or min value.

    Parameters
    ----------
    data : 2D array
        input data for alignment
    padv : int
        add padding value on vertical direction
    maxv : bool
        choose max value or min value

    Returns:
    --------
    list :
        how much is shift for each horizontal position
    array 2D :
        aligned data

    """
    d_shape = data.shape

    data0 = np.zeros([d_shape[0]+padv*2, d_shape[1]])
    data0[padv:-padv, :] = data

    mlist = []

    # get shift value
    for i in range(d_shape[1]):
        linedata = data[:, i]
        if maxv is True:
            pos = np.where(linedata == np.max(linedata))
        else:
            pos = np.where(linedata == np.min(linedata))
        mlist.append(pos[0])

    # do correction
    for i in range(datas[1]):
        diff = mlist[i] - mlist[0]
        #print padv+diff, -padv+diff
        #print mlist[i], diff
        data0[padv:-padv, i] = data0[padv+diff:-padv+diff, i]

    return mlist, data0[padv:-padv, :]


    def calculate_alignment_derivative(self, padv = 10, maxv=True):
        """
        do alignment based on max or min value of derivative
        """
        data = self.data
        datas = data.shape 
            
        data0 = np.zeros([datas[0]+padv*2, datas[1]])
        data0[padv:-padv, :] = data
        
        mlist = []
        
        # get shift value
        for i in range(datas[1]):
            grad = data[1:, i] - data[0:-1, i]
            if maxv == True:
                pos = np.where(grad == np.max(grad))
            else:
                pos = np.where(grad == np.min(grad))
            mlist.append(pos[0])
      
        datan = np.zeros([datas[0]+padv*2, datas[1]])
        
        # do correction
        for i in range(datas[1]):
            diff = mlist[i]-mlist[0]
            print padv+diff, -padv+diff
            datan[padv:-padv,i] = data0[padv+diff:-padv+diff, i]
        
        datan = datan[padv:-padv, :]
        
        data = datan
        
        return mlist, data



    def calculate_alignment_corr(self, padv = 2):
        """
        alignment based on cross correlation
        bin the data to high dim for sub-pixel alignment
        """
        
        data = self.data
        datas = data.shape    
            
        #data0 = np.zeros([datas[0]+padv*2,datas[1]])
        #data0[padv:-padv,:] = data
        data0 = data
        
        mlist = []
        cor_all = []
        
        # get shift value
        for i in range(datas[1]):
            cor_list = []
            data1 = data0[padv:-padv,0]
            # calculate cross correlation
            for j in range(-padv,padv):
                data2 = data0[padv+j:-padv+j,i]
                
                cor = get_correlation(data1, data2)
                cor_list.append(cor)
                
            cor_list = np.array(cor_list)
            pos = np.where(cor_list==np.max(cor_list))
            mlist.append(pos[0])
            cor_all.append(cor_list)
            
        #datan = np.zeros([datas[0]+padv*2,datas[1]])
        datan = np.zeros([datas[0],datas[1]])
        
        # do correction
        for i in range(datas[1]):
            diff = mlist[i]-mlist[0]
            print diff
            datan[padv:-padv,i] = data0[padv+diff:-padv+diff, i]
        
        datan = datan[padv:-padv,:]
        
        return mlist-mlist[0], datan, cor_all


    def calculate_alignment_corr_bin(self, padv = 2, bin_n = 1):
        """
        alignment based on cross correlation
        bin the data to high dim for sub-pixel alignment
        """
        
        data = self.data
        datas = data.shape
        
        # bin the data
        databin = np.zeros([datas[0]*bin_n,datas[1]])
        databin_s = databin.shape
        
        for i in range(datas[1]):
            data_temp = rebin_data_expand(data[:,i], m=bin_n)
            # linear interpolation
            #data_temp = np.histogram(data[:,i],bins=databin_s[0])
            databin[:,i] = data_temp#[1][0:-1]
    
            
        data = databin
        datas = databin_s
    
            
        #data0 = np.zeros([datas[0]+padv*2,datas[1]])
        #data0[padv:-padv,:] = data
        data0 = data
        
        mlist = []
        
        # get shift value
        for i in range(datas[1]):
            cor_list = []
            data1 = data0[padv:-padv,0]
            # calculate cross correlation
            for j in range(-padv,padv):
                data2 = data0[padv+j:-padv+j,i]
                
                cor = get_correlation(data1, data2)
                cor_list.append(cor)
                
            cor_list = np.array(cor_list)
            pos = np.where(cor_list==np.max(cor_list))
            mlist.append(pos[0])
        
        #datan = np.zeros([datas[0]+padv*2,datas[1]])
        datan = np.zeros([datas[0],datas[1]])
        
        # do correction
        for i in range(datas[1]):
            diff = mlist[i]-mlist[0]
            print diff
            datan[padv:-padv,i] = data0[padv+diff:-padv+diff, i]
        
        datan = datan[padv:-padv,:]
        
        
        data = datan
        
        
        # rebin it back
        datanobin = np.zeros([datas[0]/bin_n,datas[1]])
        datanobin_s = datanobin.shape
        
        for i in range(datanobin_s[1]):
            data_temp = rebin_data_shrink(data[:,i], m=bin_n)
            #data_temp = np.histogram(data[:,i],bins=datanobin_s[0])
            datanobin[:,i] = data_temp#[1][0:-1]
        
        return mlist-mlist[0], datanobin


    def do_vertical_alignment(self, center_list, data_all):
        """
        adjust center position of y according to center_list
        """

        s = data_all.shape
        
        padv = np.max(np.abs(center_list))
        
        data_aligny = np.zeros([s[0], s[1], s[2]])
        
        for i in range(s[0]):
            d = data_all[i,:,:]
            d = np.reshape(d,[s[1],s[2]])
            
            cenv = center_list[i,0]
            data_temp = np.zeros([s[1]+2*padv, s[2]])
            data_temp[padv:-padv,:] = d
            data_aligny[i,:,:] = data_temp[padv+cenv:-padv+cenv,:]
            
            
        return data_aligny    
    


def get_correlation(data1, data2):
    """
    calculate cross correlation between two data sets
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    #print data1.shape, data2.shape
    return np.dot(data1,data2)/np.sqrt(np.dot(data1,data1))/np.sqrt(np.dot(data2,data2))


def get_correlation2D(data1, data2):
    """
    calculate cross correlation for 2D data sets
    """

    num1_l = 0
    num1_h = 5
    num2_l = 5
    num2_h = 20
    dim = data1.shape

    data1_temp = data1[num1_l:dim[0]-num1_h, num2_l:dim[1]-num2_h]

    sum_i = 0
    i0 = 0
    j0 = 0
    for i in range(0, num1_l+num1_h):
        for j in range(0, num2_l+num2_h):
            data2_temp = data2[i:dim[0]-num1_l-num1_h+i, j:dim[1]-num2_l-num2_h+j]
            sum_v = np.sum(data2_temp*data1_temp)/np.sum(data1_temp)/np.sum(data2_temp)
            if sum_v >= sum_i:
                i0 = i
                j0 = j
            sum_i = sum_v
    print i0, j0

    data2_f = data1[i0:dim[0]-num1_l-num1_h+i0, j0:dim[1]-num2_l-num2_h+j0]

    return data2_f

    
def rebin_data_expand(data, m=2):
    """
    rebin 1D data to large size
    """
    data = np.array(data)
    dlen = len(data)
    datanew = np.zeros([dlen*m])
    for i in range(len(data)):
        #datanew[m*i:m*i+(m-1)] = data[i]
        for j in range(m):
            datanew[i*m+j] = data[i]
        #datanew[i*m+m-1] = data[i]
    return datanew


def rebin_data_shrink(data, m=2):
    '''
    shrink 1D data to small size
    '''
    data = np.array(data)
    dlen = len(data)
    datanew = np.zeros([dlen/m])
    for i in range(len(datanew)):
        #datanew[m*i:m*i+(m-1)] = data[i]
        for j in range(m):
            datanew[i] = datanew[i]+data[i*m+j]
        #datanew[i*m+m-1] = data[i]
    datanew = datanew/m
    
    return datanew



def sin_fun(x,a,b,c):
    return a*np.sin(1.0*x+b)+c


def quad_fun(x, a, b, c):
    """
    for remove wrap
    """
    return a+b*x+c*x*x


def rm_wrap(data):
    """
    remove linear and quadratic term from projected data
    """
    
    data = np.array(data)    
    datas = data.shape
        
    for i in range(datas[1]):
        y = data[:,i]
        x = np.arange(len(y))
        popt, pcov = curve_fit(quad_fun, x, y)
        a,b,c = popt
        yfit = a+b*x+c*x*x
        
        #plt.plot(x, y, x, yfit)
        #plt.show()
        #plt.close()
            
        data[0:len(y),i] = data[0:len(y),i]-yfit

      
    return data

    
    
    
    
    


