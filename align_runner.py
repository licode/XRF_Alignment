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
        self.inputdata = np.asarray(inputdata)
        return
    
    def get_interpolation(self, interpv, interph):
        """
        do interpolation in either direction
        """
        inputdata = self.inputdata
        shape_v = inputdata.shape
        return interpolation_3D_by_2Dslice(inputdata,
                                           vsize=shape_v[1]*interpv,
                                           hsize=shape_v[2]*interph, opt='both')

    def adjust_data(self, listv):
        pass

    def run(self, listv, cutv=0,
            interp=None, vertical=True):
        """
        work on either vertical or horizontal
        most of the cases we will focus on vertical direction
        """

        if interp:
            if vertical is True:
                newdata = self.get_interpolation(interp, 1)
            else:
                newdata = self.get_interpolation(1, interp)

            adj_new = adjust_data(newdata, listv, cutv, vertical=vertical)

            # interpolate back
            shape_adj = adj_new.shape

            if vertical is True:
                v_new = int(shape_adj[1]/interp)
                h_new = shape_adj[2]
            else:
                v_new = shape_adj[1]
                h_new = int(shape_adj[2]/interp)

            return interpolation_3D_by_2Dslice(adj_new,
                                               vsize=v_new,
                                               hsize=h_new, opt='both')
        else:
            return adjust_data(self.inputdata, listv, cutv, vertical=vertical)


def adjust_data(data3D, listv, cutv, vertical=True):
    """
    Adjust either vertical or horizontal position on each 2D slice.

    Parameters
    ----------
    data3D : 3D array
        input data
    listv : list
        position difference to be adjusted
    cutv : float
        cut the array first in order to adjust postion
    vertical : bool
        do vertical or horizontal alignment

    Returns
    -------
    array :
        after adjustment, the size is cutted by cutv in given dimension
    """
    data3D = np.asarray(data3D)

    if vertical is True:
        for i in range(data3D.shape[0]):
            data3D[i, cutv:-cutv, :] = data3D[i, cutv+listv[i]:-cutv+listv[i], :]
        return data3D[:, cutv:-cutv, :]
    else:
        for i in range(data3D.shape[0]):
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
