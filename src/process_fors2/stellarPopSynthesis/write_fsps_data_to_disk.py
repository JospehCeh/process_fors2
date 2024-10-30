"""
Utilitary from A. Hearin to run FSPS - through python-fsps - to generate the `HDF5` file containing the
Simple Stellar Populations used in DSPS.
"""
import argparse
import os
import sys

import h5py
from dsps.data_loaders import retrieve_ssp_data_from_fsps


def write_fsps_data(outname):
    """write_fsps_data retrieves SSP from FSPS and writes it to the specified file.

    :param outname: path to the output file
    :type outname: _type_
    """
    ssp_data = retrieve_ssp_data_from_fsps()

    fnout = os.path.abspath(outname)
    with h5py.File(fnout, "w") as hdf:
        for key, arr in zip(ssp_data._fields, ssp_data):  # noqa: B905
            hdf[key] = arr
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", help="Name of the output file")
    args = parser.parse_args()
    sys.exit(write_fsps_data(args.outname))
