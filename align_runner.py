"""
top control of alignment, do interpolation and align in 3D data
"""
import numpy as np

from interpolation_tool import interpolation_3D_by_2Dslice


class AlignRunner(object):
    """
    interpolation, do alignment based on given algorithm, then interpolate back
    """
    def __init__(self, inputdata):
        self.inputdata = np.array(inputdata)
        return
    
    
    def get_interpolation(self, interpv, interph):
        """
        do interpolation in either direction
        """
        inputdata = self.inputdata
        shape_v = inputdata.shape
        newdata = interpolation_3D_by_2Dslice(inputdata,
                                              vsize=shape_v[1]*interpv, 
                                              hsize=shape_v[2]*interph, opt='both')
        
        return newdata
        
        
    def adjust_data(self, listv):
        
        pass


    def run(self, listv, interp=1, cutv=0, opt='vertical'):
        """
        work on either vertical or horizontal
        most of the cases we will focus on vertical direction
        """
        if opt == 'vertical':
            newdata = self.get_interpolation(interp, 1)
        else:
            newdata = self.get_interpolation(1, interp)
            
        shape_n = newdata.shape
        
        lenlist = len(listv)
        if lenlist != shape_n[0]:
            print "please make sure the dimension of listv matches the data size."
            return
        
        
        adj_new = adjust_data(newdata, listv, cutv, opt=opt)
        
        # interpolate back
        shape_adj = adj_new.shape
        
        if opt == 'vertical':
            v_new = int(shape_adj[1]/interp)
            h_new = shape_adj[2]
        else:
            v_new = shape_adj[1]
            h_new = int(shape_adj[2]/interp)
            
        final_data = interpolation_3D_by_2Dslice(adj_new,
                                                 vsize=v_new, 
                                                 hsize=h_new, opt='both')
        return final_data
        
        


def adjust_data(data3D, listv, cutv, opt='vertical'):
    """
    adjust either vertical or horizontal position on each 2D slice
    opt = 'vertical' or 'horizontal'
    """
    data3D = np.array(data3D)
    datas = data3D.shape

    if opt == 'vertical':
        data3D_0 = data3D[:,cutv:-cutv,:]

        for i in range(datas[0]):
            data3D_0[i,:,:] = data3D[i,cutv+listv[i]:-cutv+listv[i],:]

    else:
        data3D_0 = data3D[:,:, cutv:-cutv]

        for i in range(datas[0]):
            data3D_0[i,:,:] = data3D[i,:,cutv+listv[i]:-cutv+listv[i]]

    return data3D_0