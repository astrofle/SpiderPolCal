
import time

from astropy.io import fits


# Comments for the FITS header.
COMMENTS = {'dG': 'Differential gain between feed polarizations',
            'psi': 'Phase difference between noise diode and sky',
            'alpha': 'Voltage ratio of polarization ellipse (alpha)',
            'epsilon': 'Cross coupling between polarizations (epsilon)',
            'phi': 'Cross coupling phase angle',
            'q_src': 'Fractional Stokes Q for the source',
            'u_src': 'Fractional Stokes U for the source',
            'v_src': 'Fractional Stokes V for the source',
            'freq': 'Mean frequency (Hz)'}

# Mapping between header keywords and fit parameters.
HEADER_KWDS = {"dG": "DG", "psi": "PSI", "alpha": "ALPH", "epsilon": "EPS", "phi": "PHI", 
               "q_src": "Q", "u_src": "U", "v_src": "V", "freq": "FREQ"}


def mm_fit_to_fits(output, mm, mm_pars, fit_pars, freq, date_obs,
                   overwrite=False):
    """
    """

    hdu = fits.PrimaryHDU(mm)
    head = hdu.header

    # Add fit parameters to the header.
    for k,v in mm_pars.items():
        kh = HEADER_KWDS[k]
        head[kh] = (v, COMMENTS[k])
        head[f"{kh}_FIT"] = fit_pars[k]["fit"]
    head[HEADER_KWDS["freq"]] = freq.to("Hz").value

    head["DATE"] = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                    "Created by spiderpolcal")
    head["DATE-OBS"] = (date_obs, "Observed time of first SDFITS row")

    hdu.writeto(output, overwrite=overwrite)
