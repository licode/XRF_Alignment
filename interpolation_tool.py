from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import numpy as np

        
def interpolation1D(x,y,xnew, method='linear'):
    """
    1D interpolation using scipy interp1d
    
    methods: nearest, linear, cubic
    
    """
    f = interp1d(x, y, kind = method)
    outv = f(xnew)
    return outv
    
    
def interpolation2D_from_1D(inputdata, vsize=100, hsize=100,
                            method='linear', opt='vertical'):
    """
    2D interpolation using scipy interp1d based on interpolation1D
    
    methods: nearest, linear, cubic
    
    opt: vertical horizontal or both, interpolate data in different dims
    
    """
    
    inputdata = np.array(inputdata)
    myshape = inputdata.shape
    
    if opt=='vertical':
        outdata = np.zeros([vsize, myshape[1]])
        for i in range(myshape[1]):
            x = np.linspace(0, myshape[0], myshape[0])
            y = inputdata[:,i]    
            xnew = np.linspace(0, myshape[0], vsize)
            ynew = interpolation1D(x,y,xnew,method=method)
            outdata[:,i] = ynew
    
    elif opt=='horizontal':
        outdata = np.zeros([myshape[0], hsize])
        for i in range(myshape[0]):
            x = np.linspace(0,myshape[1], myshape[1])
            y = inputdata[i,:]    
            xnew = np.linspace(0, myshape[1], hsize)
            ynew = interpolation1D(x,y,xnew,method=method)
            outdata[i,:] = ynew
    
    elif opt == 'both':
        outdata = np.zeros([vsize, hsize])
        
        outdata1 = np.zeros([vsize, myshape[1]])
        
        for i in range(myshape[1]):
            x = np.linspace(0, myshape[0], myshape[0])
            y = inputdata[:,i]    
            xnew = np.linspace(0, myshape[0], vsize)
            ynew = interpolation1D(x,y,xnew,method=method)
            outdata1[:,i] = ynew
        
        for i in range(vsize):
            x = np.linspace(0, myshape[1], myshape[1])
            y = outdata1[i,:]    
            xnew = np.linspace(0, myshape[1], hsize)
            ynew = interpolation1D(x,y,xnew,method=method)
            outdata[i,:] = ynew
            
    else:
        print "please select opt for interpolation"
        return None
    
    outdata = np.array(outdata)
    
    return outdata


def interpolation2D(inputdata,
                    vsize=200, hsize=400, 
                    method='cubic'):
    """
    use different 2D interpolation methods from scipy
    
    visize and hsize are the output dim after interpolation
    
    methods: nearest, linear, cubic
    
    some bugs exist. This part does't work well.
    
    """
    inputdata = np.array(inputdata)
    datas = inputdata.shape

    grid_x, grid_y = np.mgrid[0:datas[0]:np.complex(0,vsize), 0:datas[1]:np.complex(0,hsize)]

    val = np.zeros([datas[0]*datas[1]])
    pos = np.zeros([datas[0]*datas[1], 2])

    for i in range(datas[0]):
        for j in range(datas[1]):
            pos[i*datas[1]+j,0] = i
            pos[i*datas[1]+j,1] = j
            val[i*datas[1]+j] = inputdata[i,j]

    grid_z = griddata(pos, val, (grid_x, grid_y), method=method)

    return grid_z




        # Spline interpolation
        #x,y = np.mgrid[0:20:20j, 0:20:20j]
        #print x.shape, y.shape
        #tck = interpolate.bisplrep(x, y, new_data, s=0)
        #xnew, ynew = np.mgrid[0:20:40j, 0:20:40j]
        #znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
        

def interpolation_3D_by_2Dslice(inputdata, vsize=100, hsize=100,
                                method='linear', opt='vertical'):
    """
    interpolate each 2D slices from 3D data
    slice = data3D[i,:,:]
    """
    
    inputdata = np.array(inputdata)
    input_s = inputdata.shape
    
    new_data = []
    for i in range(input_s[0]):
        temp = interpolation2D_from_1D(inputdata[i,:,:], vsize=vsize, hsize=hsize,
                                       method=method, opt=opt)
        new_data.append(temp)
    
    new_data = np.array(new_data)
    
    return new_data



        