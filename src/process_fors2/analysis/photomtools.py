#!/usr/bin/env python3
"""
Module to prepare files for emission lines identification with GELATO and Star Population Synthesis with DSPS or pCIGALE.

Created on Mon Mar 11 09:56:42 2024

@author: joseph
"""
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from sedpy import observate

from process_fors2.fetchData import DEFAULTS_DICT


def loadDataInH5(specid, h5file=DEFAULTS_DICT["FORS2 HDF5"]):
    """
    Returns chosen spectra and associated data from the cross-matched catalog file.

    Parameters
    ----------
    specID : int or str
        Spectrum number in FORS2 dataset.
    h5file : str or path, optional
        Path to the HDF5 file containing the data to extract. The default is the FORS2 data file.

    Returns
    -------
    dict
        All datasets (as numpy arrays) and attributes associated to the specID in a single dictionary.

    """
    with h5py.File(h5file, "r") as catin:
        # Get available keys
        taglist = np.array(list(catin.keys()))

        # Get the key that matches the specified IDs
        tagbools = [f"{specid}" in tag for tag in catin]
        tagsel = taglist[tagbools]
        assert len(tagsel) <= 1, "Multiple entries for the given ID, the first one is selected."
        try:
            tag = tagsel[0]
        except IndexError:
            print("No entry for the given ID, please review your data")
            sys.exit(1)

        # Get the corresponding data group
        group = catin.get(tag)

        # Gather all spectral data in a dictionary
        datadict = {key: np.array(group.get(key)) for key in group}

        # Add attributes in the dictionary
        for key in group.attrs:
            datadict.update({key: group.attrs.get(key)})

    return datadict


def scalingToBand(wl, fl, mab, mab_err, mask=None, band="sdss_r0"):
    """
    Returns the scaling factor of the spectra to match photometry in a set of bands.

    Parameters
    ----------
    wl : array
        Wavelengths in Angstrom.
    fl : array
        Spectral flux CGS units $erg . cm^{-2} . s^{-1} . Ang^{-1}$.
    mAB : array or float
        Observed photometry in AB magnitudes.
    mAB_err : array or float
        Observed photometry errors in AB magnitudes.
    mask : array, optional
        Where the spectral flux is masked. 0 or False = valid flux. The default is None.
    band : list or array or str
        Photometric bands in Which to compute the magnitudes with `sedpy`.
        Must be given as a band name, or list of band names, available in `sedpy`.
        The number of bands must match the length of mAB.
        The default is `'sdss_r0'`.

    Returns
    -------
    float
        Scaling factor by which to multiply the spectral flux to match photometry in the given bands.

    """
    if isinstance(band, str):
        band = [band]
    filts = observate.load_filters(band)

    # Manage mask
    if mask is None:
        mask = np.zeros_like(fl)
    nomask = np.where(mask > 0, False, True)
    sel = np.logical_and(nomask, fl > 0.0)

    if isinstance(mab, float | int):
        mab = np.array([mab])

    if isinstance(mab_err, float | int):
        mab_err = np.array([mab_err])

    a0 = np.power(10, -0.4 * (mab[0] - filts[0].ab_mag(wl[sel], fl[sel])))
    fltofit = fl * a0

    def fun_to_minimize(a):
        diffs = np.power((observate.getSED(wl[sel], a * fltofit[sel], filterlist=filts) - mab) / mab_err, 2.0)
        return np.sum(diffs)

    res = minimize_scalar(fun_to_minimize, bounds=(0.0, 1.0e30))
    return a0 * res.x


def estimateErrors(wl, fl, mask=None, nsigma=1):
    """
    Returns an estimation of noise in FORS2 measurements using gaussian processes.

    Parameters
    ----------
    wl : array
        Wavelengths in Angstrom.
    fl : array
        Spectral flux CGS units $erg . cm^{-2} . s^{-1} . Ang^{-1}$.
    mask : array, optional
        Where the spectral flux is masked. 0 or False = valid flux. The default is None.
    nsigma : float
        Number of std deviations to consider in the smoothing process that initializes the RBF kernel. The default is 1.

    Returns
    -------
    tuple(array, array)
        Mean "noise-free" values of the spectral flux and associated noise, shaped as wl.
    """
    # Manage mask
    if mask is None:
        mask = np.zeros_like(fl)
    nomask = np.where(mask > 0, False, True)
    sel = np.logical_and(nomask, fl > 0.0)

    # Initialize the gaussian process to plausible parameters
    fl_smooth = gaussian_filter1d(fl[sel], nsigma)
    fl_dev = np.abs(fl_smooth - fl[sel]) / nsigma
    # rbf = kernels.RBF(55., (10., 100.))
    # gpr = GaussianProcessRegressor(kernel = rbf)

    # Fit the GP on observations
    # gpr.fit(wl[sel, None], fl[sel])

    # Compute the signal and noise in the observations
    # fl_mean, fl_std = gpr.predict(wl[:, None], return_std=True)
    plt.plot(wl, fl, label="Input data", alpha=0.2, c="k")
    plt.plot(wl[sel], fl_smooth, label=f"{nsigma}-sigma gaussian-filtered spectrum", c="r", lw=0.5)
    # plt.plot(wl[sel], fl_mean[sel], label="GP-predicted signal", c='g', lw=0.5)
    # plt.fill_between(wl[sel], fl_mean[sel]+fl_std[sel], fl_mean[sel]-fl_std[sel], color='g', alpha=0.2, label="GP-predicted noise")
    plt.fill_between(wl[sel], fl_smooth + fl_dev, fl_smooth - fl_dev, color="r", alpha=0.4, label="Associated noise")
    plt.xlabel("Wavelength $[Ang]$")
    plt.ylabel("Spectral flux $[erg . s^{-1} . cm^{-2} . Ang^{-1}]$")
    plt.legend()
    plt.show()

    # Interpolate result to match original shape
    fl_mean = np.interp(wl, wl[sel], fl_smooth, left=0.0, right=0.0)
    fl_std = np.interp(wl, wl[sel], fl_dev, left=0.0, right=0.0)

    return fl_mean, fl_std
