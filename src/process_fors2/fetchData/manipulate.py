#!/usr/bin/env python3
"""
Module to load and process data related to our study of the galaxy cluster RXJ0054.0-2823.
In particular, this is used to cross-match catalogs and switch from FITS format to HDF5 and/or pandas.
Data shall be available or queried using the appropriate module.

Created on Tue Feb 27 11:34:33 2024

@author: joseph
"""

import os
import pickle
import re
import sys

import astropy.coordinates as coord
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from dsps.cosmology import DEFAULT_COSMOLOGY, luminosity_distance_to_z  # in Mpc
from jax import numpy as jnp
from scipy.ndimage import gaussian_filter1d
from sedpy import observate
from tqdm import tqdm

from process_fors2.analysis import U_FL, U_FNU, U_LSUNperHz, convert_flux_toobsframe, convertFnuToFlambda, estimateErrors, scalingToBand

from .queries import FORS2DATALOC, _defaults, queryGalexMast, readKids

# FORS2DATALOC = os.path.abspath(os.environ["FORS2DATALOC"])
# FORS2_TABLE = "fors2_catalogue.fits"
# GALEX_TABLE = "queryMAST_GALEXDR6_RXJ0054.0-2823_11arcmin.fits"
# KIDS_TABLE = "queryESO_KiDS_RXJ0054.0-2823_12arcmin.fits"

DEFAULTS_DICT = _defaults.copy()

FORS2_FITS = DEFAULTS_DICT["FITS location"]
GALEX_FITS = DEFAULTS_DICT["GALEX FITS"]
KIDS_FITS = DEFAULTS_DICT["KiDS FITS"]

FORS2_SPECS = os.path.join(FORS2DATALOC, "fors2", "seds")
STARLIGHT_SPECS = os.path.join(FORS2DATALOC, "starlight")

DEFAULTS_DICT.update({"FORS2 spectra": FORS2_SPECS})
DEFAULTS_DICT.update({"Starlight spectra": STARLIGHT_SPECS})

FORS2_H5 = os.path.join(FORS2DATALOC, "fors2", "fors2_spectra.hdf5")
SL_H5 = os.path.join(FORS2DATALOC, "starlight", "starlight_spectra.hdf5")
FILENAME_SSP_DATA = "ssp_data_fsps_v3.2_lgmet_age.h5"
# FILENAME_SSP_DATA = "test_fspsData_v3_2_BASEL.h5"
# FILENAME_SSP_DATA = 'test_fspsData_v3_2_C3K.h5'
FULLFILENAME_SSP_DATA = os.path.abspath(os.path.join(os.path.join(FORS2DATALOC, "sps"), FILENAME_SSP_DATA))

DEFAULTS_DICT.update({"FORS2 HDF5": FORS2_H5})
DEFAULTS_DICT.update({"Starlight HDF5": SL_H5})
DEFAULTS_DICT.update({"DSPS HDF5": FULLFILENAME_SSP_DATA})


def fors2ToH5(infile=FORS2_FITS, inspecs=FORS2_SPECS, outfile=FORS2_H5):
    """
    Gathers FORS2 spectra and FORS2 catalog informations into an hdf5 file.
    All inputs are optional (kwargs defaulting to package's values).

    parameters
        infile          : FITS table of input FORS2 attributes.
        inspecs         : Directory containing spectra in ASCII format. Three columns : wavelength in Angstrom, Flux and mask.
        outfile         : Path to the output file that will contain all data.

    return
        None.
    """
    # if os.path.isfile(outfile):
    #    os.remove(outfile)
    spec_list = sorted(os.listdir(inspecs))
    if "IMG" in spec_list:
        spec_list.remove("IMG")
    all_numbers = [int(re.findall("^SPEC(.*)n[.]txt$", fn)[0]) for fn in spec_list]
    tabl = Table.read(infile)

    from process_fors2.analysis import convertFlambdaToFnu

    with h5py.File(outfile, "w") as hf_outfile:
        for idgal in tabl["ID"]:
            if idgal not in all_numbers:
                pass
            else:
                # print(f"DEBUG: {idgal}")

                wl, fl, msk = np.loadtxt(os.path.join(inspecs, f"SPEC{idgal}n.txt"), unpack=True)
                _sel = tabl["ID"] == idgal
                redshift = tabl[_sel]["z"].value[0]
                lines = tabl[_sel]["Lines"].value[0]
                try:
                    lines = lines.decode("UTF-8")
                except AttributeError:
                    print(f"Gal. {idgal}, has  no detected lines (Lines = {lines})")
                ra = tabl[_sel]["RAJ2000"].value[0]
                dec = tabl[_sel]["DEJ2000"].value[0]
                Rmag = tabl[_sel]["Rmag"].value[0]
                RV = tabl[_sel]["RV"].value[0]
                e_RV = tabl[_sel]["e_RV"].value[0]
                RT = tabl[_sel]["RT"].value[0]
                Nsp = tabl[_sel]["Nsp"].value[0]

                try:
                    h5group = hf_outfile.create_group(f"SPEC{idgal}")
                    h5group.attrs["num"] = idgal
                    h5group.attrs["redshift"] = redshift
                    h5group.attrs["lines"] = lines
                    h5group.attrs["ra"] = ra
                    h5group.attrs["dec"] = dec
                    h5group.attrs["Rmag"] = Rmag
                    h5group.attrs["RV"] = RV
                    h5group.attrs["eRV"] = e_RV
                    h5group.attrs["RT"] = RT
                    h5group.attrs["Nsp"] = Nsp

                    h5group.create_dataset("wl", data=wl, compression="gzip", compression_opts=9)
                    h5group.create_dataset("fl", data=fl, compression="gzip", compression_opts=9)
                    h5group.create_dataset("fnu", data=convertFlambdaToFnu(wl, fl), compression="gzip", compression_opts=9)
                    h5group.create_dataset("mask", data=msk, compression="gzip", compression_opts=9)
                except ValueError:
                    print(f"Galaxy ID.{idgal} appears to be duplicated in the table.\nPlease consider checking input data.")
                    pass
    return None


def starlightToH5(infile=FORS2_FITS, inspecs=STARLIGHT_SPECS, outfile=SL_H5):
    """
    Gathers Starlight spectra and FORS2 catalog informations into an hdf5 file.
    All inputs are optional (kwargs defaulting to package's values).

    parameters
        infile          : FITS table of input FORS2 attributes.
        inspecs         : Directory containing spectra in ASCII format. Two columns : wavelength in Angstrom and Flux.
        outfile         : Path to the output file that will contain all data.

    return
        None.
    """
    sl_nodust = os.path.join(inspecs, "full_spectra")
    sl_dust = os.path.join(inspecs, "full_spectra_ext")

    spec_list = sorted(os.listdir(sl_nodust))
    all_numbers = [int(re.findall("^SPEC(.*)_HZ4_BC[.]txt$", fn)[0]) for fn in spec_list]
    tabl = Table.read(infile)

    from process_fors2.analysis import convertFlambdaToFnu

    with h5py.File(outfile, "w") as hf_outfile:
        for idgal in tabl["ID"]:
            if idgal not in all_numbers:
                pass
            else:
                wl, fl = np.loadtxt(os.path.join(sl_nodust, f"SPEC{idgal}_HZ4_BC.txt"), unpack=True)
                wl_ext, fl_ext = np.loadtxt(os.path.join(sl_dust, f"SPEC{idgal}_HZ4_BC_ext_full.txt"), unpack=True)
                fl_ext_interp = np.interp(wl, wl_ext, fl_ext, left=0.0, right=0.0)

                _sel = tabl["ID"] == idgal
                redshift = tabl[_sel]["z"].value[0]
                lines = tabl[_sel]["Lines"].value[0]
                try:
                    lines = lines.decode("UTF-8")
                except AttributeError:
                    print(f"Gal. {idgal}, has  no detected lines (Lines = {lines})")
                ra = tabl[_sel]["RAJ2000"].value[0]
                dec = tabl[_sel]["DEJ2000"].value[0]
                Rmag = tabl[_sel]["Rmag"].value[0]
                RV = tabl[_sel]["RV"].value[0]
                e_RV = tabl[_sel]["e_RV"].value[0]
                RT = tabl[_sel]["RT"].value[0]
                Nsp = tabl[_sel]["Nsp"].value[0]

                try:
                    h5group = hf_outfile.create_group(f"SPEC{idgal}_SL")
                    h5group.attrs["num"] = idgal
                    h5group.attrs["redshift"] = redshift
                    h5group.attrs["lines"] = lines
                    h5group.attrs["ra"] = ra
                    h5group.attrs["dec"] = dec
                    h5group.attrs["Rmag"] = Rmag
                    h5group.attrs["RV"] = RV
                    h5group.attrs["eRV"] = e_RV
                    h5group.attrs["RT"] = RT
                    h5group.attrs["Nsp"] = Nsp

                    h5group.create_dataset("wl", data=wl, compression="gzip", compression_opts=9)
                    h5group.create_dataset("fl", data=fl, compression="gzip", compression_opts=9)
                    h5group.create_dataset("fnu", data=convertFlambdaToFnu(wl, fl), compression="gzip", compression_opts=9)
                    h5group.create_dataset("fl_ext", data=fl_ext_interp, compression="gzip", compression_opts=9)
                    h5group.create_dataset("fnu_ext", data=convertFlambdaToFnu(wl, fl_ext_interp), compression="gzip", compression_opts=9)
                except ValueError:
                    print(f"Galaxy ID.{idgal} appears to be duplicated in the table.\nPlease consider checking input data.")
                    pass
    return None


def GetColumnHfData(hff, list_of_keys, nameval):
    """
    Extracts the values of one attribute for all listed keys.

    parameters
      hff           : descriptor of h5 file
      list_of_keys  : list of exposures
      nameval       : name of the attribute

    return
       The array of values in the order of appearance.
    """
    all_data = []
    for key in list_of_keys:
        group = hff.get(key)
        val = group.attrs[nameval]
        all_data.append(val)
    return all_data


def readH5FileAttributes(input_file_h5):
    """
    Reads attributes from an hdf5 file.

    parameters
        input_file_h5 : path to the input file

    return
        The Pandas DataFrame of attributes for all groups in the file.
    """
    with h5py.File(input_file_h5, "r") as hf:
        list_of_keys = list(hf.keys())

        # pick one key
        key_sel = list_of_keys[0]

        # pick one group
        group = hf.get(key_sel)

        # pickup all attribute names
        all_subgroup_keys = []
        for k in group.attrs.keys():
            all_subgroup_keys.append(k)

        # create info
        dico_to_pd = {}
        for key in all_subgroup_keys:
            arr = GetColumnHfData(hf, list_of_keys, key)
            dico_to_pd.update({key: arr})

        df_info = pd.DataFrame(dico_to_pd)
    df_info = df_info.sort_values(by="num", ascending=True)
    df_info_num = df_info["num"].values
    key_tags = [f"SPEC{num}" for num in df_info_num]
    df_info["name"] = key_tags
    df_info.reset_index(drop=True, inplace=True)

    for col in df_info.columns:
        try:
            df_info[col] = pd.to_numeric(df_info[col])
        except ValueError:
            pass

    return df_info


def crossmatchFors2KidsGalex(outfile, fors2=FORS2_H5, starlight=SL_H5, kids=KIDS_FITS, galex=GALEX_FITS):
    """
    Performs a cross-match between catalogs. Writes an hdf5 file containing all attributes + inputs spectra and returns the DataFrame of attributes.

    parameters
        outfile     : hdf5 file where the results of cross-match will be written.
        **kwargs    : (optional) locations of input catalogs. Defaults to package's values for FORS2 (+ Starlight) inputs and 9-bands KiDS + GALEX.

    return
        The Pandas DataFrame of attributes for all groups in the file.
    """
    df_fors2 = readH5FileAttributes(fors2)
    df_galex = queryGalexMast(outpath=galex).to_pandas()
    df_kids = readKids(path=kids).to_pandas()
    radec_galex = coord.SkyCoord(df_galex["ra_galex"].values * u.deg, df_galex["dec_galex"].values * u.deg)
    radec_kids = coord.SkyCoord(df_kids["ra_kids"].values * u.deg, df_kids["dec_kids"].values * u.deg)

    SelectedColumns_kids = [
        "KiDS_ID",
        "KIDS_TILE",
        "ra_kids",
        "dec_kids",
        "FLUX_RADIUS",
        "CLASS_STAR",
        "Z_B",
        "Z_ML",
        "MAG_GAAP_u",
        "MAG_GAAP_g",
        "MAG_GAAP_r",
        "MAG_GAAP_i",
        "MAG_GAAP_Z",
        "MAG_GAAP_Y",
        "MAG_GAAP_J",
        "MAG_GAAP_H",
        "MAG_GAAP_Ks",
        "MAGERR_GAAP_u",
        "MAGERR_GAAP_g",
        "MAGERR_GAAP_r",
        "MAGERR_GAAP_i",
        "MAGERR_GAAP_Z",
        "MAGERR_GAAP_Y",
        "MAGERR_GAAP_J",
        "MAGERR_GAAP_H",
        "MAGERR_GAAP_Ks",
        "FLUX_GAAP_u",
        "FLUX_GAAP_g",
        "FLUX_GAAP_r",
        "FLUX_GAAP_i",
        "FLUX_GAAP_Z",
        "FLUX_GAAP_Y",
        "FLUX_GAAP_J",
        "FLUX_GAAP_H",
        "FLUX_GAAP_Ks",
        "FLUXERR_GAAP_u",
        "FLUXERR_GAAP_g",
        "FLUXERR_GAAP_r",
        "FLUXERR_GAAP_i",
        "FLUXERR_GAAP_Z",
        "FLUXERR_GAAP_Y",
        "FLUXERR_GAAP_J",
        "FLUXERR_GAAP_H",
        "FLUXERR_GAAP_Ks",
        "EXTINCTION_u",
        "EXTINCTION_g",
        "EXTINCTION_r",
        "EXTINCTION_i",
    ]
    df_kids = df_kids.filter(items=SelectedColumns_kids, axis=1)

    SelectedColumns_galex = [
        "ra_galex",
        "dec_galex",
        "fuv_mag",
        "nuv_mag",
        "fuv_magerr",
        "nuv_magerr",
        "fuv_flux",
        "nuv_flux",
        "fuv_fluxerr",
        "nuv_fluxerr",
    ]
    df_galex = df_galex.filter(items=SelectedColumns_galex, axis=1)

    all_idx_k = []  # index of the match
    all_d2d_k = []  # distance in arcsec
    all_idx_g = []  # index of the match
    all_d2d_g = []  # distance in arcsec

    df_photometry = pd.DataFrame(index=df_fors2.index, columns=SelectedColumns_kids + SelectedColumns_galex)
    for index, row in tqdm(df_fors2.iterrows()):
        c = coord.SkyCoord(row["ra"] * u.degree, row["dec"] * u.degree)
        idx_k, d2d_k, _ = c.match_to_catalog_sky(radec_kids)
        idx_g, d2d_g, _ = c.match_to_catalog_sky(radec_galex)
        all_idx_k.append(int(idx_k))
        all_idx_g.append(int(idx_g))
        all_d2d_k.append(coord.Angle(d2d_k[0]).arcsec)
        all_d2d_g.append(coord.Angle(d2d_g[0]).arcsec)
        df_photometry.loc[index, SelectedColumns_kids] = df_kids.iloc[idx_k]
        df_photometry.loc[index, SelectedColumns_galex] = df_galex.iloc[idx_g]

    all_idx_k = np.array(all_idx_k, dtype=int)
    all_idx_g = np.array(all_idx_g, dtype=int)
    all_d2d_k = np.array(all_d2d_k)
    all_d2d_g = np.array(all_d2d_g)
    df_photometry["id_galex"] = all_idx_g
    df_photometry["id_kids"] = all_idx_k
    df_photometry["asep_galex"] = all_d2d_g
    df_photometry["asep_kids"] = all_d2d_k

    df_concatenated = pd.concat((df_fors2, df_photometry), axis=1)

    for col in df_concatenated.columns:
        try:
            df_concatenated[col] = pd.to_numeric(df_concatenated[col])
        except ValueError:
            pass

    f2in = h5py.File(fors2, "r")
    tags = np.array(list(f2in.keys()))
    f2in.close()

    slin = h5py.File(starlight, "r")
    tags_sl = np.array(list(slin.keys()))
    slin.close()

    # print(f"DEBUG : {tags[:10]}")
    with h5py.File(outfile, "w") as h5out:
        for idx, row in tqdm(df_concatenated.iterrows()):
            _sel = [f"SPEC{row['num']}" in t for t in tags]
            tag = tags[_sel][0]
            _sel = [f"SPEC{row['num']}" in t for t in tags_sl]
            tag_sl = tags_sl[_sel][0]
            # print(f"DEBUG : tag {tag}, tagSL {tag_sl}, num {row['num']}")

            h5group_out = h5out.create_group(tag)

            parameter_names = list(row.index)
            parameter_values = row.values

            for name, val in zip(parameter_names, parameter_values, strict=False):
                h5group_out.attrs[name] = val

            with h5py.File(fors2, "r") as f2in:
                f2group_in = f2in.get(tag)
                # print(f"DEBUG : {f2group_in}")
                h5group_out.create_dataset("wl_f2", data=np.array(f2group_in.get("wl")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fl_f2", data=np.array(f2group_in.get("fl")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fnu_f2", data=np.array(f2group_in.get("fnu")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("mask_f2", data=np.array(f2group_in.get("mask")), compression="gzip", compression_opts=9)

            with h5py.File(starlight, "r") as slin:
                # print(f"DEBUG : {list(slin.keys())[:10]}")
                slgroup_in = slin.get(tag_sl)
                h5group_out.create_dataset("wl_sl", data=np.array(slgroup_in.get("wl")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fl_sl", data=np.array(slgroup_in.get("fl")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fnu_sl", data=np.array(slgroup_in.get("fnu")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fl_sl_ext", data=np.array(slgroup_in.get("fl_ext")), compression="gzip", compression_opts=9)
                h5group_out.create_dataset("fnu_sl_ext", data=np.array(slgroup_in.get("fnu_ext")), compression="gzip", compression_opts=9)

    return df_concatenated


def filterCrossMatch(input_file, asep_max, z_bias=None, asep_galex=None):
    """
    Returns a subset of the input table that matches the criteria in angular separation (spectra vs. photometry catalogs) and redshift (spectroscopic vs. photometric from KiDS).

    Parameters
    ----------
        input_file : path or str
            Path to HDF5 file resulting from cross-match between spectroscopic dataset and photometry catalogs. Must contain data from KiDs.
        asep_max : int or float
            Maximum angular separation in arcseconds to accept a match.
        z_bias : int or float, optional
            If not None, filters out matches that do not respect $\frac{ \\left| z_{photo} - z_{spectro} \right| }{1+z_{spectro}} < $ `z_bias`. An indicative good value is 0.1. The default is None.
        asep_galex : int or float, optional
            If not None, value used specifically for the angular separation criterion with GALEX matches, which tend to be less close than KiDS. Else, no filtering. The default is None.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing only objects that match defined criteria. Corresponding HDF5 file written to disk as `[original name]_filtered.h5`.
    """
    xmatchfile = os.path.abspath(input_file)
    if not os.path.isfile(xmatchfile):
        print(f"ABORTED : {xmatchfile} is a not valid HDF5 file.")
        sys.exit(1)

    dicts_to_keep = {}
    with h5py.File(xmatchfile, "r") as xfile:
        for tag in xfile:
            group = xfile.get(tag)

            # Gather all spectral data in a dictionary
            datadict = {key: np.array(group.get(key)) for key in group}

            # Add attributes in the dictionary
            for key in group.attrs:
                datadict.update({f"attrs.{key}": group.attrs.get(key)})

            sel_asep_kids = datadict["attrs.asep_kids"] < asep_max

            sel_asep_galex = True
            if asep_galex is not None:
                sel_asep_galex = datadict["attrs.asep_galex"] < asep_galex

            sel_z = True
            if z_bias is not None:
                zs, zb, zml = datadict["attrs.redshift"], datadict["attrs.Z_B"], datadict["attrs.redshift"] - datadict["attrs.Z_ML"]
                bias_b = np.abs(zs - zb) / (1 + zs)
                bias_ml = np.abs(zs - zml) / (1 + zs)
                sel_z = min(bias_b, bias_ml) < z_bias

            if sel_asep_kids and sel_asep_galex and sel_z:
                dicts_to_keep.update({tag: datadict})

    rep, fn = os.path.split(xmatchfile)
    fn, ext = os.path.splitext(fn)
    new_fn = f"{fn}_filtered{ext}"
    outpath = os.path.join(rep, new_fn)

    with h5py.File(outpath, "w") as outfile:
        for tag, dico in dicts_to_keep.items():
            h5group_out = outfile.create_group(tag)

            for key, val in dico.items():
                if "attrs." in key:
                    attr = key.split(".")[-1]
                    h5group_out.attrs[attr] = val
                else:
                    h5group_out.create_dataset(key, data=np.array(val), compression="gzip", compression_opts=9)

    return readH5FileAttributes(outpath)


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


def cleanGalexData(input_file, asep_galex):
    """
    Removes GALEX data from entries with an angular separation larger than `asep_max`. This seems more sensible than removing the whole entry (valuable photometry mostly comes from KiDS).

    Parameters
    ----------
        input_file : path or str
            Path to HDF5 file resulting from cross-match between spectroscopic dataset and photometry catalogs. Must contain data from GALEX.
        asep_galex : int or float
            Maximum angular separation in arcseconds to keep GALEX data.

    Returns
    -------
    DataFrame
        Pandas DataFrame cleaned from GALEX data for bad matches. Corresponding HDF5 file written to disk as `[original name]_cleanGALEX.h5`.
    """
    xmatchfile = os.path.abspath(input_file)
    if not os.path.isfile(xmatchfile):
        print(f"ABORTED : {xmatchfile} is a not valid HDF5 file.")
        sys.exit(1)

    SelectedColumns_galex = [
        "fuv_mag",
        "nuv_mag",
        "fuv_magerr",
        "nuv_magerr",
        "fuv_flux",
        "nuv_flux",
        "fuv_fluxerr",
        "nuv_fluxerr",
    ]

    clean_dicts = {}
    with h5py.File(xmatchfile, "r") as xfile:
        for tag in xfile:
            group = xfile.get(tag)

            # Gather all spectral data in a dictionary
            datadict = {key: np.array(group.get(key)) for key in group}

            # Add attributes in the dictionary
            for key in group.attrs:
                datadict.update({f"attrs.{key}": group.attrs.get(key)})

            if datadict["attrs.asep_galex"] > asep_galex:
                for galkey in SelectedColumns_galex:
                    datadict.update({f"attrs.{galkey}": np.nan})
                datadict.update({"attrs.id_galex": "CLEANED"})

            clean_dicts.update({tag: datadict})

    rep, fn = os.path.split(xmatchfile)
    fn, ext = os.path.splitext(fn)
    new_fn = f"{fn}_cleanGALEX{ext}"
    outpath = os.path.join(rep, new_fn)

    with h5py.File(outpath, "w") as outfile:
        for tag, dico in clean_dicts.items():
            h5group_out = outfile.create_group(tag)

            for key, val in dico.items():
                # print(f"DEBUG {key}, {val}")
                if "attrs." in key:
                    attr = key.split(".")[-1]
                    h5group_out.attrs[attr] = val
                else:
                    h5group_out.create_dataset(key, data=val, compression="gzip", compression_opts=9)

    return readH5FileAttributes(outpath)


def gelato_xmatch_todict(gelatoh5, xmatchh5):
    """
    Merges attributes from cross-matched data and GELATO output.

    Parameters
    ----------
    gelatoh5 : str or path
        Name or path to the `HDF5` file that contains GELATO outputs.
    xmatchh5 : str or path
        Name or path to the `HDF5` file that contains cross-matched data.

    Returns
    -------
    dict
        Dictionary of dictionaries, containing all attributes in the form {tag: {attr: val, ..}, ..}.
    """
    gelatofile = os.path.abspath(gelatoh5)
    xmatchfile = os.path.abspath(xmatchh5)
    gelatout = readH5FileAttributes(gelatofile)
    xmatchout = readH5FileAttributes(xmatchfile)
    merged_df = xmatchout.merge(right=gelatout, how="outer", on=["name", "num"])
    merged_df.set_index("name", drop=False, inplace=True)
    merged_df.sort_values("num", inplace=True)
    merged_attrs = merged_df.to_dict(orient="index")
    return merged_attrs


def dsps_to_gelato(wls_ang, params_dict, z_obs=0.0, ssp_file=None):
    """
    Returns a table that contains spectral data from DSPS formatted for GELATO, *i.e* the log10 of the wavelength in Angstroms,
    the spectral flux density per unit wavelength (flam) and the inverse variance of the fluxes, in corresponding units.

    Parameters
    ----------
    wls_ang : array
        Wavelengths in Angstrom.
    params_dict : dict
        Parameters that result from SPS-fitting procedure with DSPS.
    z_obs : int or float, optional
        Redshift of the object. The default is 0.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    Table
        Astropy Table containing the formatted data, to be saved as a `FITS` file for use with GELATO. It contains :
        1. The log10 of the wavelengths in Angstroms, column name: "loglam"
        2. The spectral flux density in flam units, column name: "flux"
        3. The inverse variances of the data points, column name: "ivar"
    """
    from process_fors2.stellarPopSynthesis import mean_spectrum

    dl = luminosity_distance_to_z(z_obs, *DEFAULT_COSMOLOGY) * u.Mpc  # in Mpc
    dsps_fnu_r = mean_spectrum(wls_ang, params_dict, z_obs, ssp_file) * U_LSUNperHz / (4 * np.pi * dl.to(u.m) ** 2) / (1 + z_obs)
    dsps_flambda_r = convertFnuToFlambda(wls_ang, dsps_fnu_r.to(U_FNU).value) * U_FL
    wls_o, dsps_flambda_o = convert_flux_toobsframe(wls_ang, dsps_flambda_r, z_obs)
    nsig = 1
    dsps_signal, dsps_noise = estimateErrors(wls_o, dsps_flambda_o.value, nsigma=nsig, makeplots=False)
    t_gel = tableForGelato(wls_o, dsps_flambda_o, dsps_noise)
    return t_gel


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
        res_df["FITS"] = np.array([n.decode("UTF-8") for n in res_df["Name"]])
        # res_df["name"] = np.array([n.split('_')[0] for n in res_df["FITS"]]) -- Added in the readH5FileAttributes function
        specs = np.array([n.split("_")[0] for n in res_df["FITS"]])
        nums = np.array([int(s.split("SPEC")[-1]) for s in specs], dtype=int)
        res_df["num"] = nums
        res_df.drop(columns="Name", inplace=True)
        for col in res_df.columns:
            try:
                res_df[col] = pd.to_numeric(res_df[col])
            except ValueError:
                pass
        with h5py.File(fileout, "w") as h5out:
            for i, row in res_df.iterrows():
                specin = row["FITS"]
                fn, ext = os.path.splitext(specin)
                specn = fn.split("_")[0]
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


def gelato_tables_from_dsps(dsps_pickle_dir, ssp_file=None):
    """
    Generates all files and folder structure for a GELATO run on DSPS outputs. Essentially a wrapper for a loop on `dsps_to_gelato()`.

    Parameters
    ----------
    dsps_pickle_dir : path or str
        Path to the directory containing DSPS outputs as pickle files.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    None
    """
    mod = dsps_pickle_dir.split("fit_")[-1]
    dsps_pickle_dir = os.path.abspath(dsps_pickle_dir)
    dict_of_dsps_fits = {}
    for file in os.listdir(dsps_pickle_dir):
        if f"{mod}" in file and ".pickle" in file:
            dspsfile = os.path.abspath(os.path.join(dsps_pickle_dir, file))
            with open(dspsfile, "rb") as _pkl:
                _dict = pickle.load(_pkl)
                dict_of_dsps_fits.update(_dict)
    geldir = f"prep_gelato_fromDSPS_{mod}"
    dn = os.path.dirname(dsps_pickle_dir)
    outdir = os.path.join(dn, geldir)
    wls_ang = np.arange(500, 30000, 10)
    from process_fors2.stellarPopSynthesis import SSPParametersFit, paramslist_to_dict

    _pars = SSPParametersFit()

    if not os.path.isdir(os.path.join(outdir, "SPECS")):
        os.makedirs(os.path.join(outdir, "SPECS"))

    all_paths = []
    all_zs = []
    for tag, dsps_fit_tag in tqdm(dict_of_dsps_fits.items()):
        redz = dsps_fit_tag["zobs"]
        dict_of_pars = paramslist_to_dict(dsps_fit_tag["fit_params"], _pars.PARAM_NAMES_FLAT)
        gelin = dsps_to_gelato(wls_ang, dict_of_pars, dsps_fit_tag["zobs"], ssp_file)
        fpath = os.path.join(outdir, "SPECS", f"{tag}_z{redz:.3f}_GEL.fits")
        gelin.write(fpath, format="fits", overwrite=True)
        all_paths.append(os.path.relpath(fpath, start=outdir))
        all_zs.append(redz)

    # Create list of objects
    objlist = Table([all_paths, all_zs], names=["Path", "z"])
    writepath = os.path.join(outdir, "specs_for_GELATO.fits")
    objlist.write(writepath, format="fits", overwrite=True)
    print(f"Done ! List of objects written in {writepath}.")


def templatesToHDF5(outfilename, templ_dict):
    """
    Writes the SED templates used for photo-z in an HDF5 for a quicker use in future runs.
    Mimics the structure of the class SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ.

    Parameters
    ----------
    outfilename : str or path
        Name of the `HDF5` file that will be written.
    templ_dict : dict
        Dictionary object containing the SED templates.

    Returns
    -------
    path
        Absolute path to the written file - if successful.
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        for key, templ in templ_dict.items():
            groupout = h5out.create_group(key)
            groupout.attrs["name"] = templ.name
            groupout.attrs["redshift"] = templ.redshift
            groupout.create_dataset("z_grid", data=templ.z_grid, compression="gzip", compression_opts=9)
            groupout.create_dataset("i_mag", data=templ.i_mag, compression="gzip", compression_opts=9)
            groupout.create_dataset("colors", data=templ.colors, compression="gzip", compression_opts=9)
            groupout.create_dataset("nuvk", data=templ.nuvk, compression="gzip", compression_opts=9)

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readTemplatesHDF5(h5file):
    """readTemplatesHDF5 loads the SED templates for photo-z from the specified HDF5 and returns them as a dictionary of objects
    SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ

    :param h5file: Path to the HDF5 containing the SED templates data.
    :type h5file: str or path-like object
    :return: The dictionary of SPS_Templates objects.
    :rtype: dictionary
    """
    from process_fors2.photoZ import SPS_Templates

    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key in h5in:
            grp = h5in.get(key)
            out_dict.update(
                {
                    key: SPS_Templates(
                        grp.attrs.get("name"), grp.attrs.get("redshift"), jnp.array(grp.get("z_grid")), jnp.array(grp.get("i_mag")), jnp.array(grp.get("colors")), jnp.array(grp.get("nuvk"))
                    )
                }
            )
    return out_dict


def photoZtoHDF5(outfilename, pz_list):
    """photoZtoHDF5 Saves the pytree of photo-z results (list of dicts) in an HDF5 file.

    :param outfilename: Name of the `HDF5` file that will be written.
    :type outfilename: str or path-like object
    :param pz_list: List of dictionaries containing the photo-z results.
    :type pz_list: list
    :return: Absolute path to the written file - if successful.
    :rtype: str or path-like object
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        for i, posts_dic in enumerate(pz_list):
            groupout = h5out.create_group(f"{i}")
            groupout.attrs["z_spec"] = posts_dic.pop("z_spec")
            groupout.create_dataset("PDZ", data=posts_dic.pop("PDZ"), compression="gzip", compression_opts=9)
            for templ, tdic in posts_dic.items():
                groupout.attrs[f"{templ} evidence"] = tdic["SED evidence"]

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readPhotoZHDF5(h5file):
    """readPhotoZHDF5 Reads the photo-z results file and generates the corresponding pytree (list of dictionaries) for analysis.

    :param h5file: Path to the HDF5 containing the photo-z results.
    :type h5file: str or path-like object
    :return: List of photo-z results dicts as computed by process_fors2.photoZ.
    :rtype: list
    """
    filein = os.path.abspath(h5file)
    out_list = []
    with h5py.File(filein, "r") as h5in:
        for key in h5in:
            grp = h5in.get(key)
            obs_dict = {"PDZ": jnp.array(grp.get("PDZ")), "z_spec": grp.attrs.get("z_spec")}
            for attr in grp.attrs:
                if "evidence" in attr:
                    templ = attr.split(" ")[0]
                    obs_dict.update({templ: {"SED evidence": grp.attrs.get(attr)}})
            out_list.append(obs_dict)
    return out_list


def dspsFitToHDF5(outfilename, dsps_dict):
    """dspsFitToHDF5 _summary_

    :param outfilename: _description_
    :type outfilename: _type_
    :param dsps_dict: _description_
    :type dsps_dict: _type_
    :return: _description_
    :rtype: _type_
    """
    fileout = os.path.abspath(outfilename)

    from process_fors2.stellarPopSynthesis import SSPParametersFit, paramslist_to_dict

    _pars = SSPParametersFit()

    with h5py.File(fileout, "w") as h5out:
        for key, gal in dsps_dict.items():
            groupout = h5out.create_group(key, track_order=True)
            params_dict = paramslist_to_dict(gal["fit_params"], _pars.PARAM_NAMES_FLAT)
            params_dict.update({"redshift": gal["zobs"], "tag": key})
            for kkey, val in params_dict.items():
                groupout.attrs[kkey] = val

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readDSPSHDF5(h5file):
    """readDSPSHDF5 _summary_

    :param h5file: _description_
    :type h5file: _type_
    :return: _description_
    :rtype: _type_
    """
    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key, grp in h5in.items():
            out_dict.update({key: {_k: _v for _k, _v in grp.attrs.items()}})
    return out_dict


def _recursive_dict_to_hdf5(group, attrs):
    for key, item in attrs.items():
        if isinstance(item, dict):
            sub_group = group.create_group(key, track_order=True)
            _recursive_dict_to_hdf5(sub_group, item)
        else:
            group.attrs[key] = item


def dspsBootstrapToHDF5(outfilename, dsps_bs_dict):
    """dspsBootstrapToHDF5 _summary_

    :param outfilename: _description_
    :type outfilename: _type_
    :param dsps_bs_dict: _description_
    :type dsps_bs_dict: _type_
    :return: _description_
    :rtype: _type_
    """
    fileout = os.path.abspath(outfilename)

    from process_fors2.stellarPopSynthesis import SSPParametersFit, paramslist_to_dict

    _pars = SSPParametersFit()

    with h5py.File(fileout, "w") as h5out:
        for key, fit in dsps_bs_dict.items():
            fitgroupout = h5out.create_group(key, track_order=True)
            for kkey, gal in fit.items():
                galgroupout = fitgroupout.create_group(kkey, track_order=True)
                params_dict = paramslist_to_dict(gal["fit_params"], _pars.PARAM_NAMES_FLAT)
                params_dict.update({"redshift": gal["zobs"], "tag": key})
                for kkkey, val in params_dict.items():
                    galgroupout.attrs[kkkey] = val

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readDSPSBootstrapHDF5(h5file):
    """readDSPSBootstrapHDF5 _summary_

    :param h5file: _description_
    :type h5file: _type_
    :return: _description_
    :rtype: _type_
    """
    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key, grp in h5in.items():
            fit_dict = {}
            for kkey, ggrp in grp.items():
                fit_dict.update({kkey: {_k: _v for _k, _v in ggrp.attrs.items()}})
            out_dict.update({key: fit_dict})
    return out_dict
