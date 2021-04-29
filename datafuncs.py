'''
findfreq(data, l, n, m)
lorentzian(omega, fwhm, omegaList)
locatefreq(freq, freqloc)
loadnorms(l, n, year)
loadHMIdata_concat(l, dat, nyears)
loadHMIdata_avg(l, day, num)
separatefreq(phi)
'''

# {{{ library imports
import numpy as np
import sys
from math import sqrt, pi
from numpy.polynomial.legendre import legval
import scipy.ndimage as sciim
from astropy.io import fits
# }}} imports


# {{{ constants
twopiemin6 = 2*pi*1e-6
# }}} constants


# {{{ def findfreq(data, l, n, m):
def findfreq(data, l, n, m):
    '''
    Find the eigenfrequency for a given (l, n, m)
    using the splitting coefficients

    Inputs: (data, l, n, m)
        data - array (hmi.6328.36)
        l - harmonic degree
        n - radial order
        m - azimuthal order

    Outputs: (nu_{nlm}, fwhm_{nl}, amp_{nl})
        nu_{nlm}    - eigenfrequency in microHz
        fwhm_{nl} - FWHM of the mode in microHz
        amp_{nl}    - Mode amplitude (A_{nl})
    '''
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) *
                             (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l:03d}, n = {n:03d}")
        modeindex = 0
    (nu, amp, fwhm) = data[modeindex, 2:5]
    if m == 0:
        return nu, fwhm, amp
    else:
        splits = np.append([0.0], data[modeindex, 12:48])
        totsplit = legval(1.0*m/L, splits)*L*0.001
        return nu + totsplit, fwhm, amp
# }}} findfreq(data, l, n, m)


# {{{ def find_acoefs(data, l, n):
def find_acoefs(data, l, n):
    '''
    Find the a-coeffs for a given (l, n)

    Inputs: (data, l, n, m)
        data - array (hmi.6328.36)
        l - harmonic degree
        n - radial order
        m - azimuthal order

    Outputs: (splits)
    
    '''
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) *
                             (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l:03d}, n = {n:03d}")
        modeindex = 0
    (nu, amp, fwhm) = data[modeindex, 2:5]
    splits = np.append([0.0], data[modeindex, 12:48])
    return splits
# }}} find_acoefs(data, l, n)


# {{{ def findfreq_vecm(data, l, n, m):
def findfreq_vecm(data, l, n, m):
    '''
    Find the eigenfrequency for a given (l, n, m)
    using the splitting coefficients

    Inputs: (data, l, n, m)
        data - array (hmi.6328.36)
        l - harmonic degree
        n - radial order
        m - azimuthal order

    Outputs: (nu_{nlm}, fwhm_{nl}, amp_{nl})
        nu_{nlm}    - eigenfrequency in microHz
        fwhm_{nl} - FWHM of the mode in microHz
        amp_{nl}    - Mode amplitude (A_{nl})
    '''
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) *
                             (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l:03d}, n = {n:03d}")
        modeindex = 0
    (nu, amp, fwhm) = data[modeindex, 2:5]
    amp = amp*np.ones(m.shape)
    mask0 = m == 0
    maskl = abs(m) >= l
    splits = np.append([0.0], data[modeindex, 12:48])
    totsplit = legval(1.0*m/L, splits)*L*0.001
    totsplit[mask0] = 0
    amp[maskl] = 0
    return nu + totsplit, fwhm, amp
# }}} findfreq_vecm(data, l, n, m)


# {{{ def lorentzian(omega, fwhm, omegaList):
def lorentzian(omega, fwhm, omegaList):
    '''
    Returns the Lorentzian profile for a given frequency,
    and damping, for a given range of frequencies.

    Inputs:
        omega       -   peak of the lorentzian in microHz
        fwhm        - damping rate in microHz
        omegaList - array containing range of frequencies in microHz

    Outputs:
        lorentzian profile - in Hz^{-2}
    '''
    return np.sqrt(fwhm/2/((-omegaList + omega)**2 + fwhm**2/4)/twopiemin6**2)
# }}} lorentzian(omega, fwhm, omegaList)


# {{{ def lorentzian_vec(omega, fwhm, omegaList):
def lorentzian_vec(omega, fwhm, omegaList):
    '''
    Returns the Lorentzian profile for a given frequency,
    and damping, for a given range of frequencies.

    Inputs:
        omega       -   peak of the lorentzian in microHz
        fwhm        - damping rate in microHz
        omegaList - array containing range of frequencies in microHz

    Outputs:
        lorentzian profile - in Hz^{-2}
    '''
    omlen = len(omega.shape)
    if omlen==1:
        omega = omega[:, np.newaxis]
        omegaList = omegaList[np.newaxis, :]
    elif omlen==2:
        omega = omega[:, :, np.newaxis]
        omegaList = omegaList[np.newaxis, np.newaxis, :]
    return 1.0/((omega - 1j*fwhm/2)**2 - omegaList**2)/twopiemin6**2
# }}} lorentzian_vec(omega, fwhm, omegaList):


def locatefreq(freq, freqloc):
    '''Returns the ind, where freq[ind] >= freqloc and freq[ind-1]=<freqloc
    
    Inputs:
        freq        - array
        freqloc - value to be found

    Outpus:
        ind         - index of the array, which corresponds to freqloc
    '''
    try:
        ind = 1 + np.where((freq[:-1]<=freqloc) * (freq[1:]>=freqloc))[0][0]
    except IndexError:
        print(" locatefreq: index not found %10.5e, minfreq = %10.5e, maxfreq = %10.5e" %(freqloc, freq.min(), freq.max()))
        ind = 0
    return ind

def loadnorms(l, n, year):
    '''
    Returns the norms for a given l, n, year

    Inputs:
        l - spherical harmonic degree
        n - radial order
        year - year number of HMI data
    
    Outputs:
        norm - value of the norm C_{nl}
    '''
    path = '/scratch/g.samarth/HMItemp/norms/'
    prefix = 'HMI_'
    suffix = '_year_0' + str(year)
    fname = path + prefix + str(l).zfill(3) + suffix
    
    try:
        norms = np.loadtxt(fname)
    except FileNotFoundError:
        print("Norm file NOT FOUND : "+fname)
        return None;

    if norms.size==2:
        if norms[1]==n:
            return norms[0]
        else:
            print("Norm NOT FOUND for l = %2s, n = %1s, year = %1s" %(l, n, year))
            return None;
    else:
        try:
            ind = np.where(norms[:, 1]==n)[0][0]
        except:
            print("Norm NOT FOUND for l = %2s, n = %1s, year = %1s" %(l, n, year))
            return None;
        return norms[ind, 0]

def loadHMIdata_concat(l, day=6328, nyears=1):
    '''
    Loads time series for a given number of years, 
    by concatenating the required number of 72-day datasets.
    Returns the IFFT of the concatenated time series.

    Inputs:
        l- (int)    spherical harmonic degree
        day - (int) starting day of time series
        nyears - (int) total number of years

    Outputs:
        phi_{l} - (np.ndarray, ndim = 2)[m, omega], frequency series for different m
    '''
    try:
        assert nyears>0
    except AssertionError:
        print("AssertionError: num has to be a positive number")
        sys.exit()
    
    daynum = int(nyears*365/72)
    if daynum<1:
        daynum=1
    for i in range(daynum):
        daynew = day + i*72;
        print("Reading "+'/scratch/seismogroup/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')
        temp = fits.open('/scratch/seismogroup/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')[1].data
        if i==0:
            tempnew = temp
        else:
            tempnew = np.concatenate((tempnew, temp), axis=1)
    return np.fft.ifft(tempnew[:, 0::2] - 1j*tempnew[:, 1::2], axis=1)

def loadHMIdata_avg(l, day=6328, num=1):
    '''
    Loads 72-day time series and averages over a given number.
    Returns the IFFT of the averaged time series.

    Inputs:
    ------
        l - int
            spherical harmonic degree
        day - int
            starting day of time series - default = 6328
        num - int
            total number of years - default = 1

    Returns:
    -------
        phi_{l} - np.ndarray(ndim=2)[m, omega]
            frequency series for different m
    '''
    try:
        assert num>0
    except AssertionError:
        print("AssertionError: While calling " +
              "loadHMIdata_avg(l, day, num) --> num has to be a positive number")
        sys.exit()
    
    for i in range(num):
        daynew = day + i*72;
        datadir = "/scratch/seismogroup/data/HMI/data"
        fname = f"{datadir}/HMI_{l:03d}_{daynew:04d}.fits"
        print(f"Reading {fname}")
        tempopen = fits.open(fname)
        temp = tempopen[1].data
        tempopen.close()
        if i==0:
            tempnew = temp
        else:
            tempnew += temp
    tempnew /= num
    return np.fft.ifft(tempnew[:, 0::2] - 1j*tempnew[:, 1::2], axis=1)

def separatefreq(phi):
    freqshape = int(phi.shape[1]/2)
    phiplus = phi[:, 1:freqshape].copy()
    phiminus = phi[:, freqshape+1:].copy()
    phiminus = phiminus[:, ::-1]
    return phiplus, phiminus

def finda1(data, l, m):
    splits = np.array([0.0])
    L = sqrt(l*(l+1))
    for i in range(data.shape[0]):
        if int(data[i,0])==l:
            for j in range(i, i+30):
                if int(data[j,1]==0):
                    splits = np.append(splits, data[j, 12:48])
                    totsplit = legval(1.0*m/L, splits)*L
                    return totsplit#-31.7*m

def derotate(phi, l, daynum, pm):
    data = np.loadtxt('/home/samarth/leakage/hmi.6328.36')
    phinew = np.zeros(phi.shape, dtype=complex)
    shiftvalmin = 10000
    shiftvalmax = 0
    for i in range(l+1):
        a1 = finda1(data, l, i)
        const = pm*a1*1e-9*72*24*3600*daynum
        phinew[i, :] = sciim.interpolation.shift(phi[i, :].real, const, mode='wrap', order=1) + \
        1j*sciim.interpolation.shift(phi[i, :].imag, const, mode='wrap', order=1)
    return phinew

def hfactor(l, n, m, lprime, nprime, mprime,\
omegaList, data, normfile, numlw, totleak):
    '''H_{nln'l'}(\omega) as defined in Hanasoge (2018)
    Inputs: n, l, m, n', l', m', omegaList
    Output: H_{ll'nn'mm'}
    '''
    '''
    # Loading the old norms C_{nl}
    cn = norms(l, n, normfile)
    cn1 = norms(lprime, nprime, normfile)
    '''
    # new norms
    writedir = '/scratch/samarth/crossSpectra/'
    cn = np.loadtxt(writedir + "norm_"+str(l).zfill(3)+"_"+str(n).zfill(2))
    cn1 = np.loadtxt(writedir + "norm_"+str(lprime).zfill(3)+"_"+str(nprime).zfill(2))

    omegasize = omegaList.shape[0]
    hval = np.zeros(omegasize, dtype=complex)
    
    omegaorig1, fwhmorig1, amporig1 = findfreq(data, l, n, m)
    omegaorig2, fwhmorig2, amporig2 = findfreq(data, lprime, nprime, mprime)
    lim1m = locatefreq(omegaList, omegaorig1 - fwhmorig1)
    lim1p = locatefreq(omegaList, omegaorig1 + fwhmorig1)
    lim2m = locatefreq(omegaList, omegaorig2 - fwhmorig2)
    lim2p = locatefreq(omegaList, omegaorig2 + fwhmorig2)
    r1 = lorentzian(omegaorig1, fwhmorig1, omegaList)
    r2 = lorentzian(omegaorig2, fwhmorig2, omegaList)

    omeganl, fwhm, amp = findfreq(data, l, n, 0)
    omeganl1, fwhm1, amp1 = findfreq(data, lprime, nprime, 0)

    gamnl = fwhm * twopiemin6
    gamnl1 = fwhm1 * twopiemin6
    omeganl = omeganl * twopiemin6
    omeganl1 = omeganl1 * twopiemin6
    omegaList = omegaList * twopiemin6
    
    n1 = cn * amp**2 * omeganl**2 * gamnl
    n2 = cn1 * amp1**2 * omeganl1**2 * gamnl1
    print(f" N1 = {n1}, N2 = {n2}")

    mask = np.zeros(omegasize, dtype=bool)
    mask[lim1m:lim1p] = True
    mask[lim2m:lim2p] = True
    '''
    print(f"Number of freqs = {mask.sum()}")
    print(f"omegamin = {omegaList[mask].min()}, omegamax = {omegaList[mask].max()}")
    print(f"lorentzianmax = {r1[mask].max()}")
    '''
    hval[mask] = -2.0*omegaList[mask]*(n2*abs(r2[mask])**2*r1[mask].conjugate() + n1*abs(r1[mask])**2*r2[mask])
    print(f"hvalsumabs = {abs(hval).sum()}, {np.sum(abs(hval))}")
    print(f"hvalsum= {(hval).sum()}, {np.sum((hval))}")
    return hval[mask]
