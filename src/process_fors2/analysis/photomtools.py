#!/usr/bin/env python3
"""
Module to prepare files for emission lines identification with GELATO and Star Population Synthesis with DSPS or pCIGALE.

Created on Mon Mar 11 09:56:42 2024

@author: joseph
"""
import os

import astropy.constants as const
import astropy.units as u
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY, luminosity_distance_to_z  # in Mpc
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from sedpy import observate

C_KMS = (const.c).to("km/s").value  # km/s
C_CMS = (const.c).to("cm/s").value  # cm/s
C_AAS = (const.c).to("AA/s").value  # AA/s
C_MS = const.c  # m/s
LSUN = const.L_sun  # Watts
parsec = const.pc  # m
AB0_Lum = (3631.0 * u.Jy * (4 * np.pi * np.power(10 * parsec, 2))).to("W/Hz")

U_LSUNperHz = u.def_unit("Lsun . Hz^{-1}", LSUN * u.Hz**-1)
AB0 = AB0_Lum.to(U_LSUNperHz)  # 3631 Jansky placed at 10 pc in units of Lsun/Hz
U_LSUNperm2perHz = u.def_unit("Lsun . m^{-2} . Hz^{-1}", U_LSUNperHz * u.m**-2)
jy_to_lsun = (1 * u.Jy).to(U_LSUNperm2perHz)

U_FNU = u.def_unit("erg . cm^{-2} . s^{-1} . Hz^{-1}", u.erg / (u.cm**2 * u.s * u.Hz))
U_FL = u.def_unit("erg . cm^{-2} . s^{-1} . AA^{-1}", u.erg / (u.cm**2 * u.s * u.AA))


def convert_flux_torestframe(wl, fl, redshift=0.0):
    """
    Shifts the flux values to restframe wavelengths and scales them accordingly.

    Parameters
    ----------
    wl : array
        Wavelengths (unit unimportant) in the observation frame.
    fl : array
        Flux density (unit unimportant).
    redshift : int or float, optional
        Redshift of the object. The default is 0.

    Returns
    -------
    tuple(array, array)
        The spectrum blueshifted to restframe wavelengths.
    """
    factor = 1.0 + redshift
    return wl / factor, fl  # * factor


def convert_flux_toobsframe(wl, fl, redshift=0.0):
    """
    Shifts the flux values to observed wavelengths and scales them accordingly.

    Parameters
    ----------
    wl : array
        Wavelengths (unit unimportant) in the restframe.
    fl : array
        Flux density (unit unimportant).
    redshift : int or float, optional
        Redshift of the object. The default is 0.

    Returns
    -------
    tuple(array, array)
        The spectrum redshifted to observed wavelengths.
    """
    factor = 1.0 + redshift
    return wl * factor, fl  # / factor


def convertFlambdaToFnu(wl, flambda):
    """
    Convert spectra density flambda to fnu.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param flambda: flux density in erg/s/cm2 /AA or W/cm2/AA
    :type flambda: float

    :return: fnu, flux density in erg/s/cm2/Hz or W/cm2/Hz
    :rtype: float

    Compute Fnu = wl**2/c Flambda
    check the conversion units with astropy units and constants
    """
    fnu = (flambda * U_FL * (wl * u.AA) ** 2 / const.c).to(U_FNU) / (1 * U_FNU)
    return fnu


def convertFnuToFlambda(wl, fnu):
    """
    Convert spectra density fnu to flambda.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param fnu: flux density in erg/s/cm2/Hz or W/cm2/Hz
    :type fnu: float

    :return: flambda, flux density in erg/s/cm2 /AA or W/cm2/AA
    :rtype: float

    Compute Flambda = Fnu / (wl**2/c)
    check the conversion units with astropy units and constants
    """
    flambda = (fnu * U_FNU * const.c / ((wl * u.AA) ** 2)).to(U_FL) / (1 * U_FL)
    return flambda


def convertFlambdaToFnu_noU(wl, flambda):
    """
    Convert spectra density flambda to fnu.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param flambda: flux density in erg/s/cm2 /AA or W/cm2/AA
    :type flambda: float

    :return: fnu, flux density in erg/s/cm2/Hz or W/cm2/Hz
    :rtype: float

    Compute Fnu = wl**2/c Flambda
    check the conversion units with astropy units and constants
    """
    fnu = flambda * jnp.power(wl, 2) / C_AAS
    return fnu


def convertFnuToFlambda_noU(wl, fnu):
    """
    Convert spectra density fnu to flambda.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param fnu: flux density in erg/s/cm2/Hz or W/cm2/Hz
    :type fnu: float

    :return: flambda, flux density in erg/s/cm2 /AA or W/cm2/AA
    :rtype: float

    Compute Flambda = Fnu / (wl**2/c)
    check the conversion units with astropy units and constants
    """
    flambda = fnu * C_AAS / jnp.power(wl, 2)
    return flambda


def scalingToBand(wl, fl, mab, mab_err, mask=None, band="sdss_r0"):
    r"""
    Returns the scaling factor of the spectra to match photometry in a set of bands.

    Parameters
    ----------
    wl : array
        Wavelengths in Angstrom.
    fl : array
        Spectral flux CGS units $erg . cm^{-2} . s^{-1} . \AA^{-1}$.
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


def estimateErrors(wl, fl, mask=None, nsigma=1, makeplots=True):
    r"""
    Returns an estimation of noise in FORS2 measurements using gaussian processes.

    Parameters
    ----------
    wl : array
        Wavelengths in Angstrom.
    fl : array
        Spectral flux CGS units $erg . cm^{-2} . s^{-1} . \AA^{-1}$.
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
    if makeplots:
        plt.plot(wl, fl, label="Input data", alpha=0.2, c="k")
        plt.plot(wl[sel], fl_smooth, label=f"{nsigma}-sigma gaussian-filtered spectrum", c="r", lw=0.5)
        # plt.plot(wl[sel], fl_mean[sel], label="GP-predicted signal", c='g', lw=0.5)
        # plt.fill_between(wl[sel], fl_mean[sel]+fl_std[sel], fl_mean[sel]-fl_std[sel], color='g', alpha=0.2, label="GP-predicted noise")
        plt.fill_between(wl[sel], fl_smooth + fl_dev, fl_smooth - fl_dev, color="r", alpha=0.4, label="Associated noise")
        plt.xlabel(r"Wavelength $[\mathrm{\AA}]$")
        plt.ylabel(r"Spectral flux $[\mathrm{erg} . \mathrm{s}^{-1} . \mathrm{cm}^{-2} . \mathrm{\AA}^{-1}]$")
        plt.legend()
        plt.show()

    # Interpolate result to match original shape
    fl_mean = np.interp(wl, wl[sel], fl_smooth, left=0.0, right=0.0)
    fl_std = np.interp(wl, wl[sel], fl_dev, left=0.0, right=0.0)

    return fl_mean, fl_std


def get_fnu_clean(gelatoh5, tag, zob=None, nsigs=8):
    """
    Computes the clean spectrum in appropriate units for SPS-fitting with DSPS.

    Parameters
    ----------
    gelatoh5 : str or path
        Name or path to the `HDF5` file that contains GELATO outputs.
    tag : str
        Name of the FORS2 spectrum to be considered, normally in the form f"SPEC{number}".
    zob : int or float
        Redshift of the galaxy. If `None`, it will be inferred from GELATO results. The default is None.
    nsigs : int or float
        Number of standard deviations to consider at smoothing. The default is 8.

    Returns
    -------
    dict
        Dictionary containing wavelengths, flux in Lsun/Hz cleaned of spectral lines (smoothed), and corresponding errors, along with background estimations.
    """
    gelatoh5file = os.path.abspath(gelatoh5)
    spec_dict = {}
    with h5py.File(gelatoh5file, "r") as gel5:
        group = gel5.get(tag)
        wls = np.array(group.get("wl_ang"))
        flam = np.array(group.get("flam"))
        flamerr = np.array(group.get("flam_err"))
        if zob is None:
            zob = group.attrs.get("SSP_Redshift") / C_KMS
        dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY) * u.Mpc  # in Mpc
        fnu = ((convertFlambdaToFnu(wls, flam) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        fnuerr = ((convertFlambdaToFnu(wls, flamerr) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        # gpr.fit(wls.reshape(-1, 1), fnu)
        # sm_fnu = gpr.predict(wls.reshape(-1, 1), return_std=False)
        sm_fnu = gaussian_filter1d(fnu, sigma=nsigs)
        deltaY = np.abs(fnu - sm_fnu)
        bkg = np.sqrt(np.median(deltaY**2))
        sel = np.logical_and(deltaY <= nsigs * bkg, fnu > 0)

        spec_dict["wl_cl"] = wls[sel]
        spec_dict["fnu_cl"] = fnu[sel]
        spec_dict["fnuerr_cl"] = fnuerr[sel]
        spec_dict["bg"] = deltaY[sel]
        spec_dict["bg_med"] = bkg
    return spec_dict


def get_fnu(gelatoh5, tag, zob=None):
    """
    Computes the spectrum in appropriate units for SPS-fitting with DSPS.

    Parameters
    ----------
    gelatoh5 : str or path
        Name or path to the `HDF5` file that contains GELATO outputs.
    tag : str
        Name of the FORS2 spectrum to be considered, normally in the form f"SPEC{number}".
    zob : int or float
        Redshift of the galaxy. If `None`, it will be inferred from GELATO results. The default is None.

    Returns
    -------
    dict
        Dictionary containing wavelengths, flux in Lsun/Hz and corresponding errors.
    """
    gelatoh5file = os.path.abspath(gelatoh5)
    spec_dict = {}
    with h5py.File(gelatoh5file, "r") as gel5:
        group = gel5.get(tag)
        wls = np.array(group.get("wl_ang"))
        flam = np.array(group.get("flam"))
        flamerr = np.array(group.get("flam_err"))
        if zob is None:
            zob = group.attrs.get("SSP_Redshift") / C_KMS
        dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY) * u.Mpc  # in meters
        fnu = ((convertFlambdaToFnu(wls, flam) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        fnuerr = ((convertFlambdaToFnu(wls, flamerr) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        spec_dict["wl"] = wls
        spec_dict["fnu"] = fnu
        spec_dict["fnuerr"] = fnuerr
    return spec_dict


def get_gelmod(gelatoh5, tag, zob=None):
    """
    Computes the spectrum in appropriate units for SPS-fitting with DSPS.

    Parameters
    ----------
    gelatoh5 : str or path
        Name or path to the `HDF5` file that contains GELATO outputs.
    tag : str
        Name of the FORS2 spectrum to be considered, normally in the form f"SPEC{number}".
    zob : int or float
        Redshift of the galaxy. If `None`, it will be inferred from GELATO results. The default is None.

    Returns
    -------
    dict
        Dictionary containing wavelengths and several fluxes (in Lsun/Hz) corresponding to each output from GELATO : full model, SSP contribution and spectral lines contributions, as well as errors.
    """
    gelatoh5file = os.path.abspath(gelatoh5)
    spec_dict = {}
    with h5py.File(gelatoh5file, "r") as gel5:
        group = gel5.get(tag)
        wls = np.array(group.get("wl_ang"))
        mod_fl = np.array(group.get("gelato_mod"))
        line_fl = np.array(group.get("gelato_line"))
        ssp_fl = np.array(group.get("gelato_ssp"))
        flamerr = np.array(group.get("flam_err"))
        if zob is None:
            zob = group.attrs.get("SSP_Redshift") / C_KMS
        dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY) * u.Mpc  # in meters
        mod_fnu = ((convertFlambdaToFnu(wls, mod_fl) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        line_fnu = ((convertFlambdaToFnu(wls, line_fl) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        ssp_fnu = ((convertFlambdaToFnu(wls, ssp_fl) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        fnuerr = ((convertFlambdaToFnu(wls, flamerr) * U_FNU).to(u.Jy) * 4 * np.pi * (dl.to(u.m) ** 2) / (1 + zob)).to(U_LSUNperHz).value
        spec_dict["wl"] = wls
        spec_dict["mod"] = mod_fnu
        spec_dict["ssp"] = ssp_fnu
        spec_dict["line"] = line_fnu
        spec_dict["fnuerr"] = fnuerr
    return spec_dict
