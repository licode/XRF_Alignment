import numpy as np


def convol2d(a, b, center=True):
    """
    Convolution between 2Darray a and b
    a : 2d array
        First array.
    b : 2d array
        Second array, the size should be the same as a.
    center : bool
        Shift the zero-frequency component
        to the center of the spectrum.
    """

    if center is True:
        array = np.fft.fftn(np.fft.fftshift(a)) * np.fft.fftn(np.fft.fftshift(b))
        array = np.fft.fftshift(np.fft.ifftn(array))
    else:
        array = np.fft.fftn(a) * np.fft.fftn(b)
        array = np.fft.ifftn(array)
    return array


def dist(nx):
    a = np.arange(nx)
    a = np.where(a < float(nx) / 2, a, abs(a - float(nx))) ** 2
    array = np.zeros((nx, nx))

    for i in range(int(nx) / 2 + 1):
        y = np.sqrt(a + i ** 2)
        array[:, i] = y
        if i != 0:
            array[:, nx - i] = y

    return np.fft.fftshift(array)



def shrinkwrap_old(image, sigma=3, threshold=0.2):
    ga = np.abs(image)
    n = len(ga)
    smt = round(3 * sigma)
    rr = dist(smt * 2 + 1)

    rr = np.exp(-(rr / sigma) ** 2 / 2)
    nr = len(rr)
    gf = np.zeros((n, n))
    gf[(n / 2 - (nr + 1) / 2):(n / 2 + (nr + 1) / 2 - 1), (n / 2 - (nr + 1) / 2):(n / 2 + (nr + 1) / 2 - 1)] = rr
    gf = gf / np.sum(gf)
    gb = np.abs(convol2d(ga, gf, center=1))

    support = np.where(gb > threshold * gb.max(), np.ones((n, n)), np.zeros((n, n)))
    return support


def gaussian2D(rangev, data_len,
               stdv, stdh, rho=0, cenv=0, cenh=0):
    """
    Calculate 2D gaussian distribution.

    Parameters
    ----------
    listv : float
        range for gaussian function,
        (-rangev, rangev) for both directions
    cenv : float
        center in vertical direction
    cenh : float
        center in horizontal direction
    stdv : float
        standard deviation in vertical direction
    stdh : float
        standard deviation in horizontal direction
    rho : float
        correlation between listv and listh

    Returns
    -------
    array :
        gaussian distribution
    """

    listh, listv = np.ogrid[-rangev: rangev: data_len*np.complex(1),
                            -rangev: rangev: data_len*np.complex(1)]
    print listh[1]-listh[0]
    gauss_fun = np.exp(-1 / (2 * (1 - rho**2)) *
                       ((listv - cenv)**2 / stdv**2 +
                        (listh - cenh)**2) / stdh**2 -
                       2 * rho * (listv - cenv) * (listh - cenh) / stdv / stdh)

    return gauss_fun * 1 / (2 * np.pi * stdv * stdh * np.sqrt(1 - rho**2))


def shrinkwrap_gaussian(data, sigma=3, threshold=0.2, axis_range=10):
    """
    Shrinkwrap algorithm using gaussian function to convolve with.

    Parameters
    ----------
    data : array
        input image
    sigma : float
        standard deviation of gaussian fucntion
    threshold : float
        cut value used in shrinkwrap algorithm
    axis_range : float
        range of gaussian function, (-axis_range, axis_range)

    Returns
    -------
    array :
        image convolved with gaussian after threshold cutting
    """
    data = np.abs(data)
    n = len(data)

    gau = gaussian2D(axis_range, n, sigma, sigma)

    newdata = np.abs(convol2d(data, gau))

    #support = np.where(gb > threshold * gb.max(), np.ones((n, n)), np.zeros((n, n)))
    return newdata



