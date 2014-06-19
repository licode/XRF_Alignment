"""
calculate phase by given phase gradient
"""
import numpy as np

def reconstruct_phase(gx, gy):
    """
    reconstruct the final phase image using gx and gy
    gx: differential phase gradient in x
    gy: differential phase gradient in y
    """
    
    gx = np.array(gx)
    gy = np.array(gy)
    
    shapev = gy.shape
    print "shape is", shapev
    row = shapev[0]
    column = shapev[1]
    totalnum = row*column
    
    dx = 0.2  # in um
    dy = 0.2
    
    w = 1 # Weighting parameter
    tx = np.fft.fftshift(np.fft.fft2(gx))
    ty = np.fft.fftshift(np.fft.fft2(gy))
    
    
    c = np.arange(totalnum, dtype=complex).reshape(row, column)
    
    
    for i in range(row):
        for j in range(column):
            kappax = 2 * np.pi * (j+1-(np.floor(column/2.0)+1)) / (column*dx)
            kappay = 2 * np.pi * (i+1-(np.floor(row/2.0)+1)) / (row*dy)
            if kappax == 0 and kappay == 0:
                c[i, j] = 0
            else:
                cTemp = -1j * (kappax*tx[i][j]+w*kappay*ty[i][j]) / (kappax**2 + w*kappay**2)
                c[i, j] = cTemp
    
    padv1 = row*2
    padv2 = column*2
    c_new = np.zeros([row+2*padv1,column+2*padv2], dtype=complex)
    
    #c_new[padv1:-padv1, padv2:-padv2] = c 
    c_new = c        
    c_new = np.fft.ifftshift(c_new)
    phi = np.fft.ifft2(c_new)
    phi = phi.real
    #imsave(namephi, self.phi)
    
    return phi




    
    
    