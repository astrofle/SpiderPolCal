"""
"""

import numpy as np

from astropy import units as u
from astropy import constants as ac


def compute_spectral_axis(table, chstart=1, chstop=-1, apply_doppler=True, verbose=False):
    """
    """

    # Copied from GBT gridder: 
    # https://github.com/nrao/gbtgridder/blob/master/src/get_data.py
    
    shape = table['DATA'].shape
    if len(shape) == 1:
        ax = 0
    elif len(shape) == 2:
        ax = 1
    
    if chstop == -1:
        chstop = shape[ax] + 1
        
    xaxis = np.zeros(shape)

    try:
        cunit1 = u.Unit(table["CUNIT1"][0])
    except (ValueError, KeyError) as error:
        if verbose:
            print(error)
            print("Assuming Hz.")
        cunit1 = u.Hz


    crv1 = table['CRVAL1']*cunit1
    cd1 = table['CDELT1']*cunit1
    crp1 = table['CRPIX1']
    vframe = table['VFRAME']*u.m/u.s

    # Observatory redshift.
    beta = vframe/ac.c
    
    # Doppler correction.
    doppler = np.ones_like(beta)
    if apply_doppler:
        doppler = np.sqrt((1.0 + beta)/(1.0 - beta))
    
    # Full spectral axis in doppler tracked frame from first row.
    # FITS counts from 1, this index refers to the original axis, before channel selection.
    indx = np.arange(chstop - chstart) + chstart
    if ax == 1:
        indx = np.tile(indx, (shape[0],1))
        xaxis[:,:] = (crv1[:,np.newaxis] + cd1[:,np.newaxis]*(indx - crp1[:,np.newaxis]))*doppler[:,np.newaxis]
    else:
        xaxis = (crv1 + cd1*(indx - crp1))*doppler
        
    return xaxis*cunit1
