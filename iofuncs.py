"""Module with functions related to input and output operations

Functions present:
writefitsfile
readfitsfile
writepickle
"""
from astropy.io import fits
import pickle as pkl
import os

def writefitsfile(a, fname):
    """Writes a fits file with a given filename
    WARNING: If a file exists with the same filename, it will be deleted.

    Parameters:
    -----------
    a - np.ndarray(dtype=float)
        array that needs to be stored in the FITS file
    fname - string
        filename containing full path

    Returns:
    --------
    None
    """
    print("Writing "+fname)
    hdu = fits.PrimaryHDU()
    try:
        os.system("rm "+fname) # removes already existing fits file
    except OSError:
        pass
    hdu.data = a
    hdu.writeto(fname)
    return None

def readfits(fname):
    """ Reads fits file from the given path
    Parameters:
    -----------
    fname - string
        name of the file including full path

    Returns:
    --------
    a - np.ndarray(dtype=float)

    """
    temp = fits.open(fname)
    a = temp[0].data
    temp.close()
    return a

def writepickle(a, fname):
    """Write to a pickle file

    Parameters:
    -----------
    a - any python data
        any python data that needs to be pickled
    fname - string
        filename including the full path

    Returns:
    --------
    None

    """
    with open(fname, "wb") as f:
        pkl.dump(a, f)
    print(f" Writing to {fname}.pkl")
    return None 
