"""
"""

import numpy as np


def sky_rot(pa):
    """
    """

    m = np.matrix(
        [[1., 0., 0., 0.],
         [0., np.cos(2*pa), np.sin(2*pa), 0.],
         [0., -np.sin(2*pa), np.cos(2*pa), 0.],
         [0., 0., 0., 1.]
         ])

    return m


def mueller(dG, psi, alpha, epsilon, phi):
    """
    Mueller matrix as defined by Heiles (2002).
    """

    m11 = 1.
    m12 = -2.*epsilon*np.sin(phi)*np.sin(2.*alpha) + dG/2.*np.cos(2.*alpha)
    m13 = 2.*epsilon*np.cos(phi)
    m14 = 2*epsilon*np.sin(phi)*np.cos(2.*alpha) + dG/2.*np.sin(2.*alpha)
    m21 = dG/2.
    m22 = np.cos(2.*alpha)
    m23 = 0.
    m24 = np.sin(2.*alpha)
    m31 = 2.*epsilon*np.cos(phi+psi)
    m32 = np.sin(2.*alpha)*np.sin(psi)
    m33 = np.cos(psi)
    m34 = -np.cos(2.*alpha)*np.sin(psi)
    m41 = 2*epsilon*np.sin(phi+psi)
    m42 = -np.sin(2.*alpha)*np.cos(psi)
    m43 = np.sin(psi)
    m44 = -np.cos(2.*alpha)*np.cos(psi)

    m = np.matrix([[m11, m12, m13, m14],
                   [m21, m22, m23, m24],
                   [m31, m32, m33, m34],
                   [m41, m42, m43, m44]
                  ]
            )

    return m


def unpack_par(key, pci, coefs, params):
    """
    """

    if params[key]['fit']:
        par = coefs[pci]
        pci += 1
    else:
        par = params[key]['value']

    return pci, par


def cost_mueller(coefs, pa, y, params, sigma=1.):
    """
    Cost function for the non-linear fit to a Mueller matrix.
    """

    # Unpack fitting parameters.
    pci = 0
    pci, dG = unpack_par('dG', pci, coefs, params)
    pci, psi = unpack_par('psi', pci, coefs, params)
    pci, alpha = unpack_par('alpha', pci, coefs, params)
    pci, epsilon = unpack_par('epsilon', pci, coefs, params)
    pci, phi = unpack_par('phi', pci, coefs, params)
    pci, q_src = unpack_par('q_src', pci, coefs, params)
    pci, u_src = unpack_par('u_src', pci, coefs, params)
    pci, v_src = unpack_par('v_src', pci, coefs, params)

    q_src_rot = q_src * np.cos(2.*pa) + u_src * np.sin(2.*pa)
    u_src_rot = -q_src * np.sin(2.*pa) + u_src * np.cos(2.*pa)
    v_src_rot = v_src

    # First equation; I.
    i_obs = 1 + q_src_rot * ( -2.*epsilon*np.sin(phi)*np.sin(2.*alpha) + dG/2.*np.cos(2.*alpha) ) + u_src_rot * 2.*epsilon*np.cos(phi) +\
            v_src_rot * ( 2.*epsilon*np.sin(phi)*np.cos(2.*alpha) + dG/2.*np.sin(2.*alpha) )
    # Q.
    q_obs = dG/2. + q_src_rot * np.cos(2.*alpha) + v_src_rot * np.sin(2.*alpha)
    # U.
    u_obs = 2.*epsilon*np.cos(phi+psi) + q_src_rot * np.sin(2.*alpha)*np.sin(psi) + u_src_rot * np.cos(psi) -\
            v_src_rot * np.cos(2.*alpha)*np.sin(psi)
    # V.
    v_obs = 2.*epsilon*np.sin(phi+psi) - q_src_rot * np.sin(2.*alpha)*np.cos(psi) + u_src_rot * np.sin(psi) +\
            v_src_rot * np.cos(2.*alpha)*np.cos(psi)

    s_obs = np.hstack((i_obs, q_obs, u_obs, v_obs))

    return (y - s_obs)/sigma


def q_mod(pa, dG, psi, alpha, epsilon, phi, q_src_rot, u_src_rot, v_src_rot):
    """
    """

    return dG/2. + q_src_rot * np.cos(2.*alpha) + v_src_rot * np.sin(2.*alpha)


def u_mod(pa, dG, psi, alpha, epsilon, phi, q_src_rot, u_src_rot, v_src_rot):
    """
    """

    return 2.*epsilon*np.cos(phi+psi) + q_src_rot * np.sin(2.*alpha)*np.sin(psi) + u_src_rot * np.cos(psi) -\
            v_src_rot * np.cos(2.*alpha)*np.sin(psi)


def v_mod(pa, dG, psi, alpha, epsilon, phi, q_src_rot, u_src_rot, v_src_rot):
    """
    """

    return 2.*epsilon*np.sin(phi+psi) - q_src_rot * np.sin(2.*alpha)*np.cos(psi) + u_src_rot * np.sin(psi) -\
            v_src_rot * np.cos(2.*alpha)*np.cos(psi)


def parse_fit(params, fit):
    """
    """

    params_order = ['dG', 'psi', 'alpha', 'epsilon', 'phi', 
                    'q_src', 'u_src', 'v_src']

    out = dict.fromkeys(params)

    j = 0
    for i,k in enumerate(params_order):
        if params[k]['fit']:
            out[k] = fit[j]
            j += 1
        else:
            out[k] = params[k]['value']

    return out
