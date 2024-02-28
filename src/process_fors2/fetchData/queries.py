#!/usr/bin/env python3
"""
Module to query public data related to the galaxy cluster RXJ0054.0-2823.

Created on Mon Feb 26 17:17:01 2024

@author: joseph
"""

import os

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.table import Table
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

FORS2DATALOC = os.environ["FORS2DATALOC"]

TARGET = "RXJ0054.0-2823"
AUTHOR = "GIRAUD"
OBJ_SIMBAD = "BAX 013.5117-28.3994"
CATALOG_VIZIER = "J/other/RAA/11.245"
OUTFILENAME = "fors2_catalogue.fits"
_script_dir = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.abspath(os.path.join(FORS2DATALOC, "fors2", OUTFILENAME))

BOX_SIZE = (11 * u.arcmin).to(u.deg)

GALEX_TABLE = "queryMAST_GALEXDR6_RXJ0054.0-2823_11arcmin.fits"
GLXTBL_PATH = os.path.join(FORS2DATALOC, "catalogs", GALEX_TABLE)
KIDS_TABLE = "queryESO_KiDS_RXJ0054.0-2823_12arcmin.fits"
KIDSTBL_PATH = os.path.join(FORS2DATALOC, "catalogs", KIDS_TABLE)


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
    return catalog_data
