#!/usr/bin/env python3
"""
Module to load and process data related to our study of the galaxy cluster RXJ0054.0-2823.
In particular, this is used to cross-match catalogs and switch from FITS format to HDF5 and/or pandas.
Data shall be available or queried using the appropriate module.

Created on Tue Feb 27 11:34:33 2024

@author: joseph
"""

import os
import re
import sys

import astropy.constants as const
import astropy.coordinates as coord
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm

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

DEFAULTS_DICT.update({"FORS2 HDF5": FORS2_H5})
DEFAULTS_DICT.update({"Starlight HDF5": SL_H5})

U_FNU = u.def_unit("fnu", u.erg / (u.cm**2 * u.s * u.Hz))
U_FL = u.def_unit("fl", u.erg / (u.cm**2 * u.s * u.AA))


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
        df_info = pd.DataFrame()
        for key in all_subgroup_keys:
            arr = GetColumnHfData(hf, list_of_keys, key)
            df_info[key] = arr
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

            for name, val in zip(parameter_names, parameter_values):
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
