'''
Module with functions related to 
input and output operations

Functions present:
	writefitsfile
'''
from astropy.io import fits
import os

def writefitsfile(a, fname):
	'''
	Writes a fits file with a given filename
	WARNING: If a file exists with the same filename, it will be deleted.
	
	Inputs:
		a - Real array (If array is imaginary, then pass real and imaginary part separately)
		fname	- filename including the full path
	Output:
		0
	'''
	print("Writing "+fname)
	hdu = fits.PrimaryHDU()
	try:
		os.system("rm "+fname) # removes already existing fits file
	except:
		pass
	hdu.data = a
	hdu.writeto(fname)
	return 0;

def readfits(fname):
	''' 
	Reads fits file from the given path
	Inputs:
		fname	- filename including the full path
	Outputs:
		a - array loaded form fits file
	'''
	temp = fits.open(fname)
	a = temp[0].data
	temp.close()
	return a
