#!/usr/bin/env python3
"""
Module to query public data related to the galaxy cluster RXJ0054.0-2823.

Created on Mon Feb 26 17:17:01 2024

@author: joseph
"""

import json
import os
from io import BytesIO

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from dl import queryClient as qc
from dl import storeClient as sc
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    FORS2DATALOC = os.environ["FORS2DATALOC"]
except KeyError:
    try:
        FORS2DATALOC = input("Please type in the path to FORS2 data, e.g. /home/usr/process_fors2/src/data")
        os.environ["FORS2DATALOC"] = FORS2DATALOC
    except Exception:
        FORS2DATALOC = os.path.join(_script_dir, "..", "data")
        os.environ["FORS2DATALOC"] = FORS2DATALOC

TARGET = "RXJ0054.0-2823"
AUTHOR = "GIRAUD"
OBJ_SIMBAD = "BAX 013.5117-28.3994"
CATALOG_VIZIER = "J/other/RAA/11.245"
OUTFILENAME = "fors2_catalogue.fits"
TABLE_PATH = os.path.abspath(os.path.join(FORS2DATALOC, "fors2", OUTFILENAME))

BOX_SIZE = (11 * u.arcmin).to(u.deg)

GALEX_TABLE = "queryMAST_GALEXDR6_RXJ0054.0-2823_11arcmin.fits"
GLXTBL_PATH = os.path.join(FORS2DATALOC, "catalogs", GALEX_TABLE)
KIDS_TABLE = "queryESO_KiDS_RXJ0054.0-2823_rad15arcmin_SG1.fits"
KIDSTBL_PATH = os.path.join(FORS2DATALOC, "catalogs", KIDS_TABLE)

_defaults = {"Target": TARGET, "Simbad name": OBJ_SIMBAD, "Vizier catalog": CATALOG_VIZIER, "FITS location": TABLE_PATH, "GALEX FITS": GLXTBL_PATH, "KiDS FITS": KIDSTBL_PATH, "Box size": BOX_SIZE}


def json_to_inputs(conf_json):
    """
    Load JSON configuration file and return inputs dictionary.

    Parameters
    ----------
    conf_json : path or str
        Path to the configuration file in JSON format.

    Returns
    -------
    dict
        Dictionary of inputs `{param_name: value}`.
    """
    conf_json = os.path.abspath(conf_json)
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    return inputs


def queryTargetInSimbad(target=TARGET):
    """
    Looks for the target name in Simbad database.
    All inputs are optional (kwargs defaulting to package's values).

    parameters
        target  : (str) target name in Simbad
    return
        Table list (can be read with astropy Table).
    """
    return Simbad.query_object(target)


def getFors2FitsTable(catalog=CATALOG_VIZIER, outpath=TABLE_PATH):
    """
    Looks for the catalog in Vizier database.
    All inputs are optional (kwargs defaulting to package's values).
    If the file already exists, it is directly loaded and its content returned.

    parameters
        catalog     : (str) catalog name in Vizier database
        outpath     : (str or path-like) FITS file to write the resulting catalog.
    return
        Table (can be read with astropy Table).
    """
    if os.path.isfile(outpath):
        tabl = Table.read(outpath)
    else:
        Vizier.ROW_LIMIT = -1
        res_query = Vizier.get_catalogs(catalog)
        tabl = Table(res_query[0])
        Vizier.ROW_LIMIT = 50
        try:
            tabl.write(outpath, format="fits")
        except ValueError:
            print(tabl.meta["description"])
            tabl.meta["description"] = input("Summarise the above in fewer than 80 characters:")
            tabl.write(outpath, format="fits")
    return tabl


def _getTargetCoordinates():
    """
    Converts the coordinates from Simbad query to degrees using astropy utilities.
    Only implemented for internal use - does not accept any argument.

    parameters

    return
        Sky coordinates of the target in astropy.coordinates format.
    """
    fromsimbad = queryTargetInSimbad()
    ra_str = "{} hours".format(fromsimbad["RA"][0])
    dec_str = "{} degree".format(fromsimbad["DEC"][0])
    radec = coord.SkyCoord(ra_str, dec_str)
    return radec


def queryGalexMast(target=OBJ_SIMBAD, outpath=GLXTBL_PATH, boxsize=BOX_SIZE.value):
    """
    Looks for the catalog in Vizier database.
    All inputs are optional (kwargs defaulting to package's values).
    If the file already exists, it is directly loaded and its content returned.

    parameters
        target      : (str) target name in Simbad
        outpath     : (str or path-like) FITS file to write the resulting catalog.
        boxsize     : size of the box covered by the mast query (in degrees) of radius = sqrt(2)*boxsize
    return
        Table (can be read with astropy Table).
    """
    if os.path.isfile(outpath):
        catalog_data = Table.read(outpath)
    else:
        catalog_data = Catalogs.query_object(target, catalog="Galex", data_release="DR6", radius=boxsize * np.sqrt(2.0))
        catalog_data.rename_column("ra", "ra_galex")
        catalog_data.rename_column("dec", "dec_galex")
        try:
            catalog_data.write(outpath, format="fits")
        except ValueError:
            print(catalog_data.meta["description"])
            catalog_data.meta["description"] = input("Summarise the above in fewer than 80 characters:")
            catalog_data.write(outpath, format="fits")
    return catalog_data


def readKids(path=KIDSTBL_PATH):
    """
    Reads existing FITS file containing results from a query of the 9-band KiDS catalog from the ESO archives website.
    All inputs are optional (kwargs defaulting to package's values).

    parameters
        path     : (str or path-like) FITS file containing the results of the query.
    return
        Table (can be read with astropy Table).
    """
    assert os.path.isfile(path), "Please query appropriate data from ESO archives and save it as a FITS file, prior to execute this function."
    catalog_data = Table.read(path)
    catalog_data.rename_column("ID", "KiDS_ID")
    catalog_data.rename_column("RAJ2000", "ra_kids")
    catalog_data.rename_column("DECJ2000", "dec_kids")
    return catalog_data


# twoddir = 'gogreen_dr1://SPECTROSCOPY/TwoD/'  # 2-d spectra
# imdir = 'gogreen_dr1://PHOTOMETRY/IMAGES/'    # photometry and images


def get_gogreen_merged_table(outfile):
    """get_gogreen_merged_table Queries GOGREEN data from NOIRLAB Astro Data Lab and saves it to disk as a pandas DataFrame.
    Quality cuts are operated in order to limit the number of objects that will be queried as individual spectra.

    :param outfile: HDF5 file name for the output
    :type outfile: str or path-like
    :return: The absolute path the the written file, if successful, else None.
    :rtype: str or path-like or None
    """
    outfile = os.path.abspath(outfile)
    cluster_table = qc.query("select * from gogreen_dr1.clusters", fmt="pandas")
    phot_table = qc.query("select * from gogreen_dr1.photo", fmt="pandas")
    redshift_table = qc.query("select * from gogreen_dr1.redshift", fmt="pandas")

    # this way avoids duplicate columns (ie dont need to specify suffixes)
    merge_col = ["specid"]
    cols_to_use = phot_table.columns.difference(redshift_table.columns).tolist() + merge_col
    matched_table = pd.merge(redshift_table, phot_table[cols_to_use], how="left", left_on=["specid"], right_on=merge_col)

    merge_col = ["cluster"]
    # Here attach suffix _c to distinguish between galaxy values (Redshift) and cluster values (Redshift_c)
    matched_table = pd.merge(matched_table, cluster_table, how="left", left_on=["cluster"], right_on=merge_col, suffixes=["", "_c"])

    sel = (matched_table["redshift_quality"] == 4) * (matched_table["objclass"] == 1) * (matched_table["spec_flag"] < 1) * (matched_table["star"] != 1) * (np.isfinite(matched_table["zspec"]))
    sel_table = matched_table[sel]
    fmt_df = format_gogreen_data(sel_table)
    fmt_df.to_hdf(outfile, key="gogreen")
    if os.path.isfile(outfile):
        print(f"File successfully written to {outfile}.")
        return outfile
    else:
        print("Unable to write GOGREEN data to disk.")
        return None


def get_gogreen_wavelength_from_hdu(hdr):
    """
    get_wavelength_from_hdu(hdr)

    :param hdr: Fits table header

    Reads 'CRVAL1' 'NAXIS1' and 'CD1_1' to compute wavelength coverage
    """
    return np.arange(hdr["CRVAL1"], hdr["CRVAL1"] + hdr["NAXIS1"] * hdr["CD1_1"], hdr["CD1_1"])


def get_gogreen_spectrum(hdu, extver, units="fl", return_frame="observed", redshift=0.0):
    """
    get_gogreen_spectrum returns the spectrum and associated noise in desired frame and units from GOGREEN data
    stored at NOIRlab using noirlab.datalab access tools.

    :param hdu:       Fits table hdu object
    :param extver:     Extension of science frame
    :param return_frame:  What frame to return the wavelength units in. If 'rest', a redshift is required.
    :param redshift:  Redshift of galaxy to convert to rest-frame, if redshift = 0 returns observed-frame
    :param units:     Units of output spectrum, case insensitive

    :type units:      string, "Fl" or "Fnu" or "maggies"
    :returns:         wavelengths, the spectrum, and the variance

    Access spectrum from fits file, convert to specified units in rest frame

    Note: Input spectrum must be in units erg cm^-2 s^-1 A^-1
    Note: Header values must give wavelength in Angstroms
    """
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    extver = int(extver)

    units = units.lower()
    assert units in ["fl", "maggies", "fnu"], 'Error, units must be "Fl" or "maggies" or "Fnu"'
    return_frame = return_frame.lower()
    assert return_frame in ["rest", "observed"], 'Error, return_frame must be either "rest" or "observed"'

    scale = hdu["SCI", extver].header["FLUXSCAL"]
    spec = hdu["SCI", extver].data / scale
    var = hdu["VAR", extver].data / scale**2

    lam = get_gogreen_wavelength_from_hdu(hdu["SCI", extver].header)

    if return_frame == "rest":  # convert from observed to rest wavelength
        assert redshift >= 0, f"ERROR: redshift must be positive, is {redshift}"
        dl = (cosmo.luminosity_distance(redshift).to(u.pc).value / 10.0) ** (-2)
        spec *= (1.0 + redshift) / dl
        var *= ((1.0 + redshift) / dl) ** 2
        lam /= 1.0 + redshift

    if units == "maggies":
        convers = (3.34e4 * lam**2) / 3631.0
        spec *= convers
        var *= convers**2
    elif units == "fnu":
        convers = 3.34e4 * lam**2
        spec *= convers
        var *= convers**2

    return lam, spec, np.sqrt(var)


def gogreen_to_gelato(gg_infile, output_dir):
    """gogreen_to_gelato Queries individual spectra matching the data in the input file and writes them to disk as FITS files for use with GELATO.

    :param gg_infile: HDF5 file containing the GOGREEN data as a pandas DataFrame.
    :type gg_infile: str or path-like
    :param output_dir: Directory where to store the GELATO inputs as FITS files
    :type output_dir: str or path-like
    :return: The list of FITS spectra as an Astropy Table and the path to the output directory
    :rtype: tuple(Table, str)
    """
    from process_fors2.fetchData import tableForGelato

    gg_df = pd.read_hdf(gg_infile, key="gogreen")
    all_paths = []
    all_zs = []
    oneddir = "gogreen_dr1://SPECTROSCOPY/OneD/"  # 1-d spectra

    for _, row in tqdm(gg_df.iterrows(), total=gg_df.shape[0]):
        fits_path = oneddir + row["cluster"] + "_final.fits"

        with fits.open(BytesIO(sc.get(fits_path))) as hdu:
            lam, spec, std = get_gogreen_spectrum(hdu, row["extver"], return_frame="observed")  # get observed frame spectra

        # Conversion to GELATO format
        t = tableForGelato(lam, spec, std)

        # Write data
        outdir = os.path.abspath(output_dir)
        if not os.path.isdir(os.path.join(outdir, "SPECS")):
            os.makedirs(os.path.join(outdir, "SPECS"))

        redz = row["redshift"]
        fpath = os.path.join(outdir, "SPECS", f"{row['cluster']}_{row['specid']}_z{redz:.3f}_GEL.fits")
        t.write(fpath, format="fits", overwrite=True)
        all_paths.append(fpath)
        all_zs.append(redz)

    # Create list of objects
    objlist = Table([all_paths, all_zs], names=["Path", "z"])
    writepath = os.path.join(outdir, "specs_for_GELATO.fits")
    objlist.write(writepath, format="fits", overwrite=True)
    print(f"Done ! List of objects written in {writepath}.")

    return objlist, writepath


def format_gogreen_data(gg_df_in):
    """format_gogreen_data Transforms the photometry data (names and values) from GOGREEN to AB magnitudes for use with DSPS.

    :param gg_df_in: GOGREEN catalogue data as loaded and merged from astro datalab.
    :type gg_df_in: pandas DataFrame
    :return: Formatted and transformed data
    :rtype: pandas Dataframe
    """
    filt_corresp_dict = {
        "b": "subaru_suprimecam_B",
        "fuv": "galex_FUV",
        "g": "hsc_g",
        "h": "vista_vircam_H",
        "i": "hsc_i",
        "ia484": "subaru_suprimecam_ia484",
        "ia527": "subaru_suprimecam_ia527",
        "ia624": "subaru_suprimecam_ia624",
        "ia679": "subaru_suprimecam_ia679",
        "ia738": "subaru_suprimecam_ia738",
        "ia767": "subaru_suprimecam_ia767",
        "ib427": "subaru_suprimecam_ia427",
        "ib464": "subaru_suprimecam_ia464",
        "ib505": "subaru_suprimecam_ia505",
        "ib574": "subaru_suprimecam_ia574",
        "ib709": "subaru_suprimecam_ia709",
        "ib827": "subaru_suprimecam_ia827",
        "irac1": "spitzer_irac_ch1",
        "irac2": "spitzer_irac_ch2",
        "irac3": "spitzer_irac_ch3",
        "irac4": "spitzer_irac_ch4",
        "j": "vista_vircam_J",
        "k": "ukirt_wfcam_K",
        "ks": "vista_vircam_Ks",
        "mips24": "spitzer_mips_24",
        "nuv": "galex_NUV",
        "r": "hsc_r",
        "u": "decam_u",
        "v": "subaru_suprimecam_V",
        "y": "hsc_y",
        "z": "hsc_z",
    }
    filt_cols = [c for c in gg_df_in.columns if "_tot" in c]
    mags_cols, magerrs_cols = [c for c in filt_cols if c[0] != "e"], [c for c in filt_cols if c[0] == "e"]
    new_mag_names = {oldn: f"mag_{filt_corresp_dict[oldn.split('_')[0]]}" for oldn in mags_cols}
    new_magerr_names = {oldn: f"magerr_{filt_corresp_dict[oldn.split('_')[0][1:]]}" for oldn in magerrs_cols}
    gg_df = gg_df_in.rename(columns=new_mag_names, inplace=False)
    gg_df.rename(columns=new_magerr_names, inplace=True)
    gg_df["num"] = gg_df["specid"]
    new_mag_cols = [new_mag_names[k] for k in mags_cols]
    new_magerr_cols = [new_magerr_names[k] for k in magerrs_cols]
    for mcol, merrcol in zip(new_mag_cols, new_magerr_cols, strict=True):
        flux, fluxerr = np.array(gg_df[mcol]), np.array(gg_df[merrcol])
        gg_df[mcol] = -2.5 * np.log10(flux) + 25
        gg_df[merrcol] = np.abs(-2.5 * np.log10(1 + fluxerr / flux))
    return gg_df


def load_filters_from_ggdf(catalogue_df, wls=None):
    """load_filters_from_ggdf _summary_

    :param catalogue_df: _description_
    :type catalogue_df: _type_
    :param wls: _description_, defaults to None
    :type wls: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    from interpax import interp1d
    from jax import numpy as jnp
    from sedpy import observate

    mags_cols = [c for c in catalogue_df.columns if "mag" in c.lower() and "err" not in c.lower() and "image" not in c.lower()]
    spy_filt_names = ["_".join(m.split("_")[1:]) for m in mags_cols]
    spy_filts = observate.load_filters(spy_filt_names)
    if wls is None:
        wls = jnp.arange(100.0, 1.0e5, 10)
    transm_list = [interp1d(wls, f.wavelength, f.transmission, method="linear", extrap=0.0) for f in spy_filts]
    wlmean_list = [f.wave_mean for f in spy_filts]
    return wls, jnp.array(transm_list), jnp.array(wlmean_list)
