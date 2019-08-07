from __future__ import print_function
import numpy as np
import sys
from math import sqrt, pi
from numpy.polynomial.legendre import legval
from astropy.io import fits

twopiemin6 = 2*pi*1e-6

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
		nu_{nlm} - eigenfrequency in microHz
		fwhm_{nl} - FWHM of the mode in microHz
		amp_{nl} - Mode amplitude (A_{nl})
	'''
	L = sqrt(l*(l+1))
	try:
		modeindex = np.where((data[:, 0]==l) * (data[:,1]==n))[0][0]
	except:
		print( "MODE NOT FOUND : l = %3s, n = %2s" %( l, n ) )
		return None, None, None
	(nu, amp, fwhm) = data[modeindex, 2:5]
	splits = np.append([0.0], data[modeindex, 12:48])
	totsplit = legval(1.0*m/L, splits)*L*0.001
	return nu + totsplit, fwhm, amp

def lorentzian(omega, fwhm, omegaList):
	'''
	Returns the Lorentzian profile for a given frequency,
	and damping, for a given range of frequencies.

	Inputs:
		omega	-	peak of the lorentzian in microHz
		fwhm	- damping rate in microHz
		omegaList - array containing range of frequencies in microHz

	Outputs:
		lorentzian profile - in Hz^{-2}
	'''
	if omega<=omegaList.max() and omega>=omegaList.min():
		return 1.0/((omega - 1j*fwhm/2)**2 - omegaList**2)/twopiemin6**2
	else:
		print("Peak frequency %10.5f outside range (%10.5f, %10.5f)" %(omega, omegaList.min(), omegaList.max()))
		return 0*omegaList

def locatefreq(freq, freqloc):
	'''
	Returns the ind, where freq[ind] >= freqloc and freq[ind-1]=<freqloc
	Inputs:
		freq - array
		freqloc - value to be found
	Outpus:
		ind - index of the array, which corresponds to freqloc
	'''
	try:
		ind = 1 + np.where((freq[:-1]<=freqloc) * (freq[1:]>=freqloc))[0][0]
	except:
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
	path = '/scratch/samarth/HMItemp/norms/'
	prefix = 'HMI_'
	suffix = '_year_0' + str(year)
	fname = path + prefix + str(l).zfill(3) + suffix
	
	try:
		norms = np.loadtxt(fname)
	except:
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
		l			- (int)	spherical harmonic degree
		day		- (int) starting day of time series
		nyears- (int) total number of years
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
	for i in xrange(daynum):
		daynew = day + i*72;
		print("Reading "+'/scratch/jishnu/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')
		temp = fits.open('/scratch/jishnu/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')[1].data
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
		l			- (int)	spherical harmonic degree
		day		- (int) starting day of time series - default = 6328
		num 	- (int) total number of years - default = 1
	Outputs:
		phi_{l} - (np.ndarray, ndim = 2)[m, omega], frequency series for different m
	'''
	try:
		assert num>0
	except AssertionError:
		print("AssertionError: While calling loadHMIdata_avg(l, day, num) --> num has to be a positive number")
		sys.exit()
	
	for i in xrange(num):
		daynew = day + i*72;
		print("Reading "+'/scratch/jishnu/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')
		tempopen = fits.open('/scratch/jishnu/data/HMI/data/HMI_'+str(l).zfill(3)+'_'+str(daynew).zfill(4)+'.fits')
		temp = tempopen[1].data
		tempopen.close()
		if i==0:
			tempnew = temp
		else:
			tempnew += temp
	tempnew /= num
	return np.fft.ifft(tempnew[:, 0::2] - 1j*tempnew[:, 1::2], axis=1)
