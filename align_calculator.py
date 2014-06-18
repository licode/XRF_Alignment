"""
calculate shift position based different alignment methods
"""
import numpy as np


class AlignCalculator(object):
    
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
    
        
        
        
    
    
    
    
    