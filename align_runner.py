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
    data3D = np.asarray(data3D)
    datas = data3D.shape

    if opt == 'vertical':
        for i in range(datas[0]):
            data3D[i, cutv:-cutv, :] = data3D[i, cutv+listv[i]:-cutv+listv[i], :]
        return data3D[:, cutv:-cutv, :]
    else:
        for i in range(datas[0]):
            data3D[i, :, cutv:-cutv] = data3D[i, :, cutv+listv[i]:-cutv+listv[i]]
        return data3D[:, :, cutv:-cutv]


def change_2Darray_with_center(data, cen_v, cen_h, 
                               v1=10, v2=40, h1=40, h2=40):
    """
    adjust 2D array according to center position
    """

    data = np.array(data)
    datas = data.shape

    data_n = data[cen_v-v1:cen_v+v2, cen_h-h1:cen_h+h2]

    return data_n


def change_3Darray_with_center(data, cenlist_v, cenlist_h, 
                               v1=10, v2=40, h1=40, h2=40):

    data = np.array(data)
    datas = data.shape
    
    datanew = []
    
    for i in range(datas[0]):
        cen_v = cenlist_v[i]
        cen_h = cenlist_h[i]
        
        data_temp  = change_2Darray_with_center(data[i,:,:], cen_v, cen_h, 
                                                v1=v1, v2=v2, h1=h1, h2=h2)
    
        datanew.append(data_temp)
        
    return np.array(datanew)






