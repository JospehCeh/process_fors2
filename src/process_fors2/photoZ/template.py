#!/usr/bin/env python3
"""
Module to specify and use SED templates for photometric redshifts estimation algorithms.
Insipired by previous developments in [EmuLP](https://github.com/JospehCeh/EmuLP).

Created on Thu Aug 1 12:59:33 2024

@author: joseph
"""

import os
import pickle
from collections import namedtuple

import pandas as pd
from jax import jit, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map

from process_fors2.stellarPopSynthesis import SSPParametersFit, mean_spectrum, paramslist_to_dict, ssp_spectrum_fromparam

_DUMMY_P_ADQ = SSPParametersFit()

BaseTemplate = namedtuple("BaseTemplate", ["name", "flux", "z_sps"])
SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"])


def read_params(pickle_file):
    """read_params Reads the parameters to syntheticise an SED in DSPS from the specified pickle file.

    :param pickle_file: Pickle file location
    :type pickle_file: str or path-like
    :return: DSPS input parameters
    :rtype: dict
    """
    new_dict = {}
    with open(pickle_file, "rb") as pkl:
        par_dict = pickle.load(pkl)
    for tag, dico in par_dict.items():
        params_dict = paramslist_to_dict(dico["fit_params"], _DUMMY_P_ADQ.PARAM_NAMES_FLAT)
        params_dict.update({"redshift": dico["zobs"], "tag": tag})
        new_dict.update({tag: params_dict})
    return new_dict


def read_h5_table(templ_h5_file):
    """read_h5_table _summary_

    :param templ_h5_file: _description_
    :type templ_h5_file: _type_
    :return: _description_
    :rtype: _type_
    """
    templ_df = pd.read_hdf(os.path.abspath(templ_h5_file), key="fit_dsps")
    templ_pars_arr = jnp.array(templ_df[_DUMMY_P_ADQ.PARAM_NAMES_FLAT])
    return templ_pars_arr  # placeholder, finish the function later to return the proper array of parameters


@jit
def templ_mags(X, params, z_obs, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    from dsps import calc_obs_mag, calc_rest_mag
    from dsps.cosmology import DEFAULT_COSMOLOGY

    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    obs_mags = tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters[:-2], list_transm_filters[:-2])
    rest_mags = tree_map(lambda x, y: calc_rest_mag(ssp_wave, sed_attenuated, x, y), list_wls_filters[-2:], list_transm_filters[-2:])

    mags_predictions = jnp.concatenate((jnp.array(obs_mags), jnp.array(rest_mags)))

    return mags_predictions


v_mags = vmap(templ_mags, in_axes=(None, None, 0, None))


# @jit
def calc_nuvk(wls, params_dict, zobs, ssp_data):
    """calc_nuvk Computes the theoretical emitted NUV-IR color index of a reference galaxy.

    :param wls: Wavelengths
    :type wls: array
    :param params_dict: DSPS input parameters to compute the restframe NUV and NIR photometry.
    :type params_dict: dict
    :param zobs: Redshift value
    :type zobs: float
    :return: NUV-NIR color index
    :rtype: float
    """
    from process_fors2.photoZ import NIR_filt, NUV_filt, ab_mag

    rest_sed = mean_spectrum(wls, params_dict, zobs, ssp_data)
    nuv = ab_mag(NUV_filt.wavelengths, NUV_filt.transmission, wls, rest_sed)
    nir = ab_mag(NIR_filt.wavelengths, NIR_filt.transmission, wls, rest_sed)
    return nuv - nir


v_nuvk = vmap(calc_nuvk, in_axes=(None, None, 0, None))


def make_sps_templates(params_dict, filt_tup, redz, ssp_data, id_imag=3):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_dict: DSPS input parameters
    :type params_dict: dict
    :param filt_tup: Filters in which to compute the photometry of the templates, given as a tuple of two arrays : one for wavelengths, one for transmissions.
    :type filt_tup: tuple of arrays
    :param redz: redshift grid on which to compute the templates photometry
    :type redz: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: SPS_Templates object (namedtuple)
    """
    name = params_dict.pop("tag")
    z_sps = params_dict.pop("redshift")
    template_mags = v_mags(filt_tup, params_dict, redz, ssp_data)
    nuvk = template_mags[:, -2] - template_mags[:, -1]
    colors = template_mags[:, :-3] - template_mags[:, 1:-2]
    i_mag = template_mags[:, id_imag]
    return SPS_Templates(name, z_sps, redz, i_mag, colors, nuvk)


@jit
def templ_mags_legacy(X, params, z_ref, z_obs, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    from dsps import calc_obs_mag, calc_rest_mag
    from dsps.cosmology import DEFAULT_COSMOLOGY

    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_ref, ssp_data)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    obs_mags = tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters[:-2], list_transm_filters[:-2])
    rest_mags = tree_map(lambda x, y: calc_rest_mag(ssp_wave, sed_attenuated, x, y), list_wls_filters[-2:], list_transm_filters[-2:])

    mags_predictions = jnp.concatenate((jnp.array(obs_mags), jnp.array(rest_mags)))

    return mags_predictions


v_mags_legacy = vmap(templ_mags_legacy, in_axes=(None, None, None, 0, None))


def make_legacy_templates(params_dict, filt_tup, redz, ssp_data, id_imag=3):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.
    Contrary to `make_sps_template`, this methods only shifts the restframe SED and does not reevaluate the stellar population at each redshift.
    Mainly used for comparative studies as other existing photoZ codes such as BPZ and LEPHARE will do this and more.

    :param params_dict: DSPS input parameters
    :type params_dict: dict
    :param filt_tup: Filters in which to compute the photometry of the templates, given as a tuple of two arrays : one for wavelengths, one for transmissions.
    :type filt_tup: tuple of arrays
    :param redz: redshift grid on which to compute the templates photometry
    :type redz: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, NOT accounting for the Star Formation History up to the redshift value.
    :rtype: SPS_Templates object (namedtuple)
    """
    name = params_dict.pop("tag")
    z_sps = params_dict.pop("redshift")
    template_mags = v_mags_legacy(filt_tup, params_dict, z_sps, redz, ssp_data)
    nuvk = template_mags[:, -2] - template_mags[:, -1]
    colors = template_mags[:, :-3] - template_mags[:, 1:-2]
    i_mag = template_mags[:, id_imag]
    return SPS_Templates(name, z_sps, redz, i_mag, colors, nuvk)


"""OLD FUNCTIONS FOR REFERENCE
def make_base_template(ident, specfile, wl_grid):
    wl, _lums = np.loadtxt(os.path.abspath(specfile), unpack=True)
    _inds = jnp.argsort(wl)
    wls = wl[_inds]
    lum = _lums[_inds]
    wavelengths, lums = jnp.array(wls), jnp.array([l if l>0. else 1.0e-20 for l in lum])
    lumins = jnp.interp(wl_grid, wavelengths, lums, left=0., right=0., period=None)
    return ident, lumins

def nojit_no_ext_make_template(base_temp_lums, filts, z, cosmo, wl_grid):
    lumins = base_temp_lums
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    print(f"Dist. modulus = {d_modulus}")
    mags = jnp.array([filter.noJit_ab_mag(filt.wavelengths, filt.transmission, zshift_wls, lumins) + d_modulus for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_scaled_template(base_temp_lums, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid,  wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))

    scale = calc_scale_arrs(f_ab, galax_fab, galax_fab_err)
    print(f"Scale={scale}")
    scaled_lumins = ext_lumins*scale
    scaled_mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, scaled_lumins) + d_modulus for filt in filts])
    scaled_f_ab = jnp.power(10., -0.4*(scaled_mags+48.6))
    return scaled_mags

@partial(jit, static_argnums=4)
#@partial(vmap, in_axes=(None, None, 0, 0, None, None))
def make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    #ext_lumins = base_temp_lums*extinc_arr
    #zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    #d_modulus = Cosmology.calc_distMod(cosmo, z)
    #d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid*(1.+z),
    base_temp_lums*extinc_arr*opacities) + cosmology.distMod(cosmo, z) for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def make_dusty_template(base_temp_lums, filts, extinc_arr, wl_grid):
    #ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid, calc_dusty_transm(base_temp_lums, extinc_arr)) for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def calc_fab(filts, wvls, lums, d_mod=0.):
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wvls, lums) + d_mod for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def make_scaled_template(base_temp_lums, filts, extinc_arr, galax_fab, galax_fab_err, z, wl_grid, d_modulus, opacities):
    ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr) * opacities
    zshift_wls = (1.+z)*wl_grid
    #f_ab = calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus)
    #scale = calc_scale_arrs(calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus), galax_fab, galax_fab_err)
    #scaled_lumins = ext_lumins*scale
    return calc_fab(filts,
                    zshift_wls,
                    ext_lumins*calc_scale_arrs(calc_fab(filts, zshift_wls, ext_lumins, d_modulus), galax_fab, galax_fab_err),
                    d_modulus)

#@partial(jit, static_argnums=(0,1,2))
@jit
def calc_scale_arrs(f_templ, f_gal, err_gal):
    #_sel1 = jnp.isfinite(f_gal)
    #_sel2 = jnp.isfinite(f_templ)
    #_sel3 = jnp.isfinite(err_gal)
    #_sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
    #_sel = _sel1 * _sel2 * _sel3
    #if len(f_templ[_sel]) > 0 :
        # Scaling as in LEPHARE

    arr_o = f_gal/err_gal
    arr_t = f_templ/err_gal
    #avmago = jnp.sum(arr_o*arr_t)
    #avmagt = jnp.sum(jnp.power(arr_t, 2.))
    return jnp.sum(arr_o*arr_t)/jnp.sum(jnp.power(arr_t, 2.))
    #else:
    #    _scale = 1.
"""
