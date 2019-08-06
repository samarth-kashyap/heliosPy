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
		if n==0:
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
