import numpy as np
from numpy.polynomial.legendre import legval
from math import sqrt, pi

def findfreq(data, l, n, m):
	'''
	Find the eigenfrequency for a given (l, n, m)
	using the splitting coefficients
	
	Inputs: (data, l, n, m)
		data - array (hmi.6328.36)
		l - harmonic degree
		n - radial order
		m - azimuthal order

	Outputs: (nu_{nlm}, fwhm, amp)
		nu_{nlm} - eigenfrequency in micro Hz
		fwhm - FWHM of the mode in micro Hz
		amp - Mode amplitude (A_{nl})
	'''
	L = sqrt(l*(l+1))
	for i in xrange(data.shape[0]):
		if int(data[i,0])==l:
			for j in xrange(i, i+35):
				if int(data[j,1])==n:
					nu = data[j,2]
					amp = data[j,3]
					fwhm = data[j,4]
					splits = np.append([0.0], data[j, 12:48])
					totsplit = legval(1.0*m/L, splits)*L*0.001 #1e-3 factor because split is in nHz
					return nu + totsplit, fwhm, amp
	return None; # if mode is not found
