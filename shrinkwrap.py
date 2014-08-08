from numpy import *

def convol2d(a,b,center=1):
    if len(a)!=len(b):
	   print 'Sorry but arrays must be same size (for now)!'
    if center==1:
	   array=fft.fftn(fft.fftshift(a))*fft.fftn(fft.fftshift(b))
	   array=fft.fftshift(fft.ifftn(array))
    else:
	   array=fft.fftn(a)*fft.fftn(b)
	   array=fft.ifftn(array)	
    return array

def dist(nx):
	a=arange(nx)
	a=where(a<float(nx)/2.,a,abs(a-float(nx)))**2
	array=zeros((nx,nx))
	for i in range(int(nx)/2+1):
		y=sqrt(a+i**2)
		array[:,i]=y
		if i!=0:
			array[:,nx-i]=y
	
	return fft.fftshift(array)
    
def shrinkwrap(image,sigma=3,threshold=0.2):

    ga=abs(image)
    n=len(ga)
    smt=round(3*sigma)
    rr=dist(smt*2+1)

    rr=exp(-(rr/sigma)*(rr/sigma)/2)
    nr=len(rr)
    gf=zeros((n,n))
    gf[(n/2-(nr+1)/2):(n/2+(nr+1)/2-1),(n/2-(nr+1)/2):(n/2+(nr+1)/2-1)]=rr
    gf=gf/sum(gf)
    gb=abs(convol2d(ga,gf,center=1))
    
    support=where(gb>threshold*gb.max(),ones((n,n)),zeros((n,n)))
    return support
