"""
calculate shift position based different alignment methods
"""
import numpy as np

from interpolation_tool import interpolation2D_from_1D
from align_method import AlignmentMethod


class AlignCalculatorCoarse(object):
    
    def __init__(self, inputdata):
        """
        inputdata is a 3D dataset, alignment is applied for each 2D.
        """
        self.inputdata = np.array(inputdata)
        return
    
    
    def from_edge(self, opt='vertical'):
        """
        find edge feature
        """
        inputdata = self.inputdata
        
        if opt=='vertical':
            data_proj = np.sum(inputdata, axis=2)
        else:
            data_proj = np.sum(inputdata, axis=1)
        
        data_proj = np.transpose(data_proj)    
        shape_p = data_proj.shape

        pos = []
        
        for i in range(shape_p[1]):
            der_v = data_proj[1:,i]-data_proj[:-1,i]
            maxv = np.where(der_v==np.max(der_v))
            pos.append(maxv[0])    
        
        return np.array(pos)
    
    
    def from_center(self, opt='vertical'):
        """
        find mass center
        """
        inputdata = self.inputdata
        
        if opt=='vertical':
            data_proj = np.sum(inputdata, axis=2)
        else:
            data_proj = np.sum(inputdata, axis=1)
        
        data_proj = np.transpose(data_proj)    
        shape_p = data_proj.shape
        
        pos = []
        
        for i in range(shape_p[1]): 
            y = np.arange(shape_p[0])
            data_y = data_proj[:,i]
            cen_y = np.sum(data_y*y)/(np.sum(data_y)) 
            pos.append(int(cen_y))
            
        return np.array(pos)
    
        

class AlignCalculatorFine(object):
    """
    interpolation is applied. 
    """
    
    def __init__(self, data):
        self.data = np.array(data)
        return
    
    def get_projection(self, opt='vertical'):
        data = self.data

        if opt=='vertical':
            data_proj = np.sum(data, axis=2)
        else:
            data_proj = np.sum(data, axis=1)
        
        data_proj = np.transpose(data_proj)  
        
        return data_proj
        
    
    def get_interpolation(self, proj, interp=1, method='cubic'):
        """
        only works on vertical direction, 
        this requires we transfer data into correct format
        """
        
        shapev = proj.shape
                    
        proj_n = interpolation2D_from_1D(proj, 
                                         vsize=shapev[0]*interp, 
                                         hsize=shapev[1], method=method, opt='both') 
            
        return proj_n
        
        
    def get_position(self, proj, cutv):
        """
        output position shift calculated from alignment
        """
        
        #proj = self.get_projection(opt=opt)
        #proj_n = self.get_interpolation(proj, interp=interp)
        
        AM = AlignmentMethod(proj)
        listv, data_n, corr_list = AM.calculate_alignment_corr(padv=cutv)
        
        return listv, data_n, corr_list
        
        
        
        
        
    
    
        
    
    
    
    
    