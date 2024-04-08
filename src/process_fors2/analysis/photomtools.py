#!/usr/bin/env python3
"""
Module to prepare files for emission lines identification with GELATO and Star Population Synthesis with DSPS or pCIGALE.

Created on Mon Mar 11 09:56:42 2024

@author: joseph
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from sedpy import observate
from tqdm import tqdm

from process_fors2.fetchData import DEFAULTS_DICT

C = 299792.458  # km/s


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
        tagbools = [f"SPEC{specid}" in tag for tag in catin]
        tagsel = taglist[tagbools]
        if f"SPEC{specid}" in tagsel:
            tag = f"SPEC{specid}"
        else:
            try:
                assert len(tagsel) <= 1
                tag = tagsel[0]
            except AssertionError:
                tag = input(f"Several entries found : {tagsel}. Pease type in the one to use.")
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


def tableForGelato(wl, fl, std, mask=None):
    r"""
    Returns a table that contains spectral data formatted for GELATO, *i.e* the log10 of the wavelength in Angstroms,
    the spectral flux density per unit wavelength (flam) and the inverse variance of the fluxes, in corresponding units.

    Parameters
    ----------
    wl : array
        Wavelengths in Angstrom.
    fl : array
        Spectral flux CGS units $erg . cm^{-2} . s^{-1} . \AA^{-1}$.
    std : array
        Spectral flux errors (as standard deviation, or $\sigma$) in units $erg . cm^{-2} . s^{-1} . \AA^{-1}$.
    mask : array, optional
        Where the spectral flux is masked. 0 or False = valid flux. The default is None.

    Returns
    -------
    Table
        Astropy Table containing the formatted data, to be saved as a `FITS` file for use with GELATO. It contains :
        1. The log10 of the wavelengths in Angstroms, column name: "loglam"
        2. The spectral flux density in flam units, column name: "flux"
        3. The inverse variances of the data points, column name: "ivar"
    """
    # Manage mask
    if mask is None:
        mask = np.zeros_like(fl)
    nomask = np.where(mask > 0, False, True)
    sel = np.logical_and(nomask, np.isfinite(fl))
    sel = np.logical_and(sel, fl > 0.0)
    sel = np.logical_and(sel, np.isfinite(std))
    sel = np.logical_and(sel, std > 0.0)

    # Transform data
    wl_gel = np.log10(wl[sel])
    flam_gel = fl[sel]
    inv_var = np.power(std[sel], -2)

    # Create table
    t = Table([wl_gel, flam_gel, inv_var], names=["loglam", "flux", "ivar"])
    return t


def crossmatchToGelato(input_file, output_dir, smoothe=False, nsigma=3):
    """
    Reads data from input file, makes it compatible with GELATO and writes necessary files.

    Parameters
    ----------
    input_file : str or path
        Path to the `HDF5` file resulting from the crossmatch between FORS2 spectra and photometric catalogs.
    output_dir : str or path
        Path to the output directory where to store `FITS` tables for GELATO.
    smoothe : bool, optional
        Whether to use the smoothed spectrum in the `FITS` tables for GELATO or the raw one. The default is False.
    nsigma : int, optional
        Number of sigma to use at smoothing during noise estimation.\
        If `smoothe` is `True`, this also impacts the spectrum that is exported for GELATO. The default is 3.

    Returns
    -------
    tuple(Table, path)
        Astropy Table of the list of objects to be passed to GELATO. It is also saved in `output_dir`.
    """
    xmatchfile = os.path.abspath(input_file)
    if not os.path.isfile(xmatchfile):
        print(f"ABORTED : {xmatchfile} is a not valid HDF5 file.")
        sys.exit(1)

    with h5py.File(xmatchfile, "r") as xfile:
        tags = np.array(list(xfile.keys()))
        nums = []
        for tag in tags:
            group = xfile.get(tag)
            num = group.attrs.get("num")
            nums.append(num)
    nums = np.array(nums)

    # Create filters to scale the photometry. Creation is redondant scalingToBand function, could be improved.
    filts = observate.load_filters(["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0"])
    sdss_u, sdss_g, sdss_r, sdss_i = filts

    all_paths = []
    all_zs = []

    for specid in tqdm(nums):
        # Load data in input file
        datadict = loadDataInH5(specid, h5file=xmatchfile)
        try:
            mags = [datadict["MAG_GAAP_u"], datadict["MAG_GAAP_g"], datadict["MAG_GAAP_r"], datadict["MAG_GAAP_i"]]
            magserr = [datadict["MAGERR_GAAP_u"], datadict["MAGERR_GAAP_g"], datadict["MAGERR_GAAP_r"], datadict["MAGERR_GAAP_i"]]
        except KeyError:
            print(f"ABORTED : {xmatchfile} appears not to contain KiDS photometry.")

        wlf2 = datadict["wl_f2"]
        flf2 = datadict["fl_f2"]
        maskf2 = datadict["mask_f2"]

        # Identify good photometry for scaling
        sel = np.where(maskf2 > 0, False, True)
        good_filts = []
        good_mags = []
        good_magserr = []
        for f, m, err in zip(filts, mags, magserr, strict=False):
            if (f.blue_edge > min(wlf2[sel])) and (f.red_edge < max(wlf2[sel])) and np.isfinite(m) and np.isfinite(err):
                good_filts.append(f)
                good_mags.append(m)
                good_magserr.append(err)

        # Scale flux on photometry
        try:
            flux2phot = scalingToBand(wlf2, flf2, good_mags, good_magserr, mask=maskf2, band=[f.name for f in good_filts])
        except IndexError:
            print("CAUTION : no filter encompasses the full spectra. Review graph and select one as a reference for scaling.")
            f, a = plt.subplots(1, 1)
            aa = a.twinx()
            for filt, c in zip(filts, ["b", "g", "y", "r"], strict=False):
                aa.fill_between(filt.wavelength, filt.transmission, color=c, alpha=0.4, label=filt.name)
                aa.axvline(filt.blue_edge, lw=0.5, c=c, ls=":")
                aa.axvline(filt.red_edge, lw=0.5, c=c, ls=":")
                aa.axvline(filt.wave_effective, lw=0.5, c=c, ls="-")
            a.plot(wlf2[sel], flf2[sel], lw=0.5, c="k", label=datadict["name"])
            a.set_xlabel(r"Wavelength $[ \AA ]$")
            a.set_ylabel("Spectral flux [arbitrary units]")
            aa.set_ylabel("Filter transmission")
            f.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0))
            # plt.show()
            plt.pause(15)
            filt_id = int(input("Type 1 for 'sdss_u0', 2 for 'sdss_g0', 3 for 'sdss_r0', or 4 for 'sdss_i0':")) - 1
            flux2phot = scalingToBand(wlf2, flf2, mags[filt_id], magserr[filt_id], mask=maskf2, band=filts[filt_id].name)
        scaled_flux = flux2phot * flf2

        # Eyeball estimate of noise in flux data
        fl_signal, fl_noise = estimateErrors(wlf2, scaled_flux, mask=maskf2, nsigma=nsigma, makeplots=False)
        sm_noise = gaussian_filter1d(fl_noise, 5)  # smoothing of the noise, just because.

        # Conversion to GELATO format
        t = tableForGelato(wlf2, fl_signal, sm_noise, mask=maskf2) if smoothe else tableForGelato(wlf2, scaled_flux, sm_noise, mask=maskf2)

        # Write data
        outdir = os.path.abspath(output_dir)
        if not os.path.isdir(os.path.join(outdir, "SPECS")):
            os.makedirs(os.path.join(outdir, "SPECS"))

        redz = datadict["redshift"]
        fpath = os.path.join(outdir, "SPECS", f"{datadict['name']}_z{redz:.3f}_GEL.fits")
        t.write(fpath, format="fits", overwrite=True)
        all_paths.append(fpath)
        all_zs.append(redz)

    # Create list of objects
    objlist = Table([all_paths, all_zs], names=["Path", "z"])
    writepath = os.path.join(outdir, "specs_for_GELATO.fits")
    objlist.write(writepath, format="fits", overwrite=True)
    print(f"Done ! List of objects written in {writepath}.")

    return objlist, writepath


def gelatoToH5(outfilename, gelato_run_dir):
    """
    Gathers data from GELATO inputs and outputs and writes them to a HDF5 file to be used as input for Stellar Population Synthesis.

    Parameters
    ----------
    outfilename : str or path
        Name of the `HDF5` file that will be written.
    gelato_run_dir : str or path
        Path to the output directory of the GELATO run to consider.

    Returns
    -------
    path
        Absolute path to the written file - if successful.
    """
    gelatout = os.path.abspath(os.path.join(gelato_run_dir, "resultsGELATO"))
    fileout = os.path.abspath(outfilename)

    if os.path.isdir(gelatout):
        res_tab_path = os.path.join(gelatout, "GELATO-results.fits")
        res_table = Table.read(res_tab_path)
        res_df = res_table.to_pandas()
        res_df["Name"] = np.array([n.decode("UTF-8") for n in res_df["Name"]])
        for col in res_df.columns:
            try:
                res_df[col] = pd.to_numeric(res_df[col])
            except ValueError:
                pass
        with h5py.File(fileout, "w") as h5out:
            for i, row in res_df.iterrows():
                specin = row["Name"]
                fn, ext = os.path.splitext(specin)
                specn = specin.split("_")[0]
                spec_path = os.path.join(gelatout, f"{fn}-results{ext}")
                spec_tab = Table.read(spec_path)
                wlang = np.power(10, spec_tab["loglam"])
                flam = np.array(spec_tab["flux"])
                flamerr = np.power(spec_tab["ivar"], -0.5)
                groupout = h5out.create_group(specn)
                for key, val in row.items():
                    groupout.attrs[key] = val
                groupout.create_dataset("wl_ang", data=wlang, compression="gzip", compression_opts=9)
                groupout.create_dataset("flam", data=flam, compression="gzip", compression_opts=9)
                groupout.create_dataset("flam_err", data=flamerr, compression="gzip", compression_opts=9)
                groupout.create_dataset("gelato_mod", data=np.array(spec_tab["MODEL"]), compression="gzip", compression_opts=9)
                groupout.create_dataset("gelato_ssp", data=np.array(spec_tab["SSP"]), compression="gzip", compression_opts=9)
                groupout.create_dataset("gelato_line", data=np.array(spec_tab["LINE"]), compression="gzip", compression_opts=9)

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret
