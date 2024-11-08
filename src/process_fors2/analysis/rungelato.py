#! /usr/bin/env python3

"""
Wrapper for mulitple gelato runs, adapted from the original convenience function provided in GELATO.

Created on Thu Mar 14 14:14:14 2024

@author: joseph
"""

# Packages
import copy
import os

# GELATO
import gelato
from gelato import ConstructParams, Utility


def run_gelato_single(pars, spec, z):
    """
    Run GELATO from within python to identify lines in spectra. This version is limited to a single spectrum.

    Parameters
    ----------
    pars : path or str
        Path to the parameters file (json).
    spec : path or str
        Path to the spectrum file (fits table of 3 columns : log10(wl in angstrom), flux in erg/cm²/s/angstrom, inverse variance of the flux).
    z :  int or float
        Redshift of the spectrum

    Returns
    -------
    int
        0 if exited correctly, else another value.
    """
    # Parameters
    _pars = os.path.abspath(pars)
    _spec = os.path.abspath(spec)
    p = ConstructParams.construct(_pars)

    ## Create Directory for Output
    outpath = os.path.abspath(p["OutFolder"])
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if p["Verbose"]:
        now = Utility.header()

    # Single Mode
    gelato.gelato(_pars, _spec, z)

    if p["Verbose"]:
        Utility.footer(now)
    return 0


def run_gelato():
    """
    Run GELATO to identify lines in spectra.

    Parameters
    ----------
    None

    Returns
    -------
    int
        0 if exited correctly, else another value.
    """
    # Parse Argument
    args = Utility.parseArgs()

    # Parameters
    _p = ConstructParams.construct(args.Parameters)
    p = _p["gelato"]  # To account for the multilayer JSON configuration file

    ## Create Directory for Output
    outpath = os.path.abspath(p["OutFolder"])
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if p["Verbose"]:
        now = Utility.header()

    # Single Mode
    if args.single:
        gelato.gelato(args.Parameters, args.Spectrum, args.Redshift)

    # Multi Mode
    else:
        # Assemble Objects
        objects = Utility.loadObjects(args.ObjectList)

        ## Run gelato ##
        if p["NProcess"] > 1:  # Mutlithread
            import multiprocessing as mp

            pool = mp.Pool(processes=p["NProcess"])
            inputs = [(copy.deepcopy(args.Parameters), o["Path"], o["z"]) for o in objects]
            pool.starmap(gelato.gelato, inputs)
            pool.close()
            pool.join()
        else:  # Single Thread
            for o in objects:
                gelato.gelato(copy.deepcopy(args.Parameters), o["Path"], o["z"])

        # Concatenate Results
        if p["Concatenate"]:
            from gelato import Concatenate

            Concatenate.concatfromresults(p, objects)

    if p["Verbose"]:
        Utility.footer(now)
    return 0


# Main Function
if __name__ == "__main__":
    import sys

    sys.exit(run_gelato())
