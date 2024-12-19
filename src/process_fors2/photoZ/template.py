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

from process_fors2.stellarPopSynthesis import SSPParametersFit, istuple, mean_spectrum, paramslist_to_dict, ssp_spectrum_fromparam, vmap_calc_obs_mag, vmap_calc_rest_mag

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
    zref_arr = jnp.array(templ_df["redshift"])
    return templ_pars_arr, zref_arr  # placeholder, finish the function later to return the proper array of parameters


@jit
def templ_mags(params, wls, filt_trans_arr, z_obs, anu, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: 1D JAX-array of floats of length (nb bands+2)
    """
    _pars = params.at[13].set(anu)
    # get the restframe spectra without and with dust attenuation
    ssp_wave, _, sed_attenuated = ssp_spectrum_fromparam(_pars, z_obs, ssp_data)
    _mags = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr[:-2], z_obs)
    _nuvk = vmap_calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr[-2:])

    mags_predictions = jnp.concatenate((_mags, _nuvk))

    return mags_predictions


vmap_mags_anu = vmap(templ_mags, in_axes=(None, None, None, None, 0, None))
vmap_mags_zobs = vmap(vmap_mags_anu, in_axes=(None, None, None, 0, None, None))
vmap_mags_pars = vmap(vmap_mags_zobs, in_axes=(0, None, None, None, None, None))


def templ_clrs_nuvk(params, wls, filt_trans_arr, z_obs, anu, ssp_data):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands-1), float)
    """
    _mags = templ_mags(params, wls, filt_trans_arr, z_obs, anu, ssp_data)
    return _mags[:-3] - _mags[1:-2], _mags[-2] - _mags[-1]


vmap_clrs_anu = vmap(templ_clrs_nuvk, in_axes=(None, None, None, None, 0, None))
vmap_clrs_zobs = vmap(vmap_clrs_anu, in_axes=(None, None, None, 0, None, None))
vmap_clrs_pars = vmap(vmap_clrs_zobs, in_axes=(0, None, None, None, None, None))


def templ_iclrs_nuvk(params, wls, filt_trans_arr, z_obs, anu, ssp_data, id_imag):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands), float)
    """
    _mags = templ_mags(params, wls, filt_trans_arr, z_obs, anu, ssp_data)
    return _mags[:-2] - _mags[id_imag], _mags[-2] - _mags[-1]


vmap_iclrs_anu = vmap(templ_iclrs_nuvk, in_axes=(None, None, None, None, 0, None, None))
vmap_iclrs_zobs = vmap(vmap_iclrs_anu, in_axes=(None, None, None, 0, None, None, None))
vmap_iclrs_pars = vmap(vmap_iclrs_zobs, in_axes=(0, None, None, None, None, None, None))


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


def make_sps_templates(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param anu_arr: Attenuation grid on which to compute the templates photometry
    :type anu_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-3] - template_mags[:, :, :, 1:-2]
    templ_tupl = [tuple(_pars) for _pars in params_arr]
    reslist_of_tupl = tree_map(lambda partup: vmap_clrs_zobs(jnp.array(partup), wls, transm_arr, redz_arr, anu_arr, ssp_data), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_clrs_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    return reslist_of_tupl


def make_sps_itemplates(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag=3):
    """make_sps_itemplates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param anu_arr: Attenuation grid on which to compute the templates photometry
    :type anu_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # i_mag = template_mags[:, :, :, id_imag]
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-2] - i_mag
    templ_tupl = [tuple(_pars) for _pars in params_arr]
    reslist_of_tupl = tree_map(lambda partup: vmap_iclrs_zobs(jnp.array(partup), wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_iclrs_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag)
    return reslist_of_tupl


@jit
def templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, anu, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: 1D JAX-array of floats of length (nb bands+2)

    """
    _pars = params.at[13].set(anu)
    # get the restframe spectra without and with dust attenuation
    ssp_wave, _, sed_attenuated = ssp_spectrum_fromparam(_pars, z_ref, ssp_data)
    _mags = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr[:-2], z_obs)
    _nuvk = vmap_calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr[-2:])

    mags_predictions = jnp.concatenate((_mags, _nuvk))

    return mags_predictions


vmap_mags_anu_legacy = vmap(templ_mags_legacy, in_axes=(None, None, None, None, None, 0, None))
vmap_mags_zobs_legacy = vmap(vmap_mags_anu_legacy, in_axes=(None, None, None, None, 0, None, None))
vmap_mags_pars_legacy = vmap(vmap_mags_zobs_legacy, in_axes=(0, 0, None, None, None, None, None))


def templ_clrs_nuvk_legacy(params, z_ref, wls, filt_trans_arr, z_obs, anu, ssp_data):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands-1), float)
    """
    _mags = templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, anu, ssp_data)
    return _mags[:-3] - _mags[1:-2], _mags[-2] - _mags[-1]


vmap_clrs_anu_legacy = vmap(templ_clrs_nuvk_legacy, in_axes=(None, None, None, None, None, 0, None))
vmap_clrs_zobs_legacy = vmap(vmap_clrs_anu_legacy, in_axes=(None, None, None, None, 0, None, None))
vmap_clrs_pars_legacy = vmap(vmap_clrs_zobs_legacy, in_axes=(0, 0, None, None, None, None, None))


def templ_iclrs_nuvk_legacy(params, z_ref, wls, filt_trans_arr, z_obs, anu, ssp_data, id_imag):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param anu: Attenuation parameter in dust law
    :type anu: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands), float)
    """
    _mags = templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, anu, ssp_data)
    return _mags[:-2] - _mags[id_imag], _mags[-2] - _mags[-1]


vmap_iclrs_anu_legacy = vmap(templ_iclrs_nuvk_legacy, in_axes=(None, None, None, None, None, 0, None, None))
vmap_iclrs_zobs_legacy = vmap(vmap_iclrs_anu_legacy, in_axes=(None, None, None, None, 0, None, None, None))
vmap_iclrs_pars_legacy = vmap(vmap_iclrs_zobs_legacy, in_axes=(0, 0, None, None, None, None, None, None))


def make_legacy_templates(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data):
    """make_legacy_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param z_ref: array of redshift of the galaxy used as template
    :type z_ref: JAX-array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param anu_arr: Attenuation grid on which to compute the templates photometry
    :type anu_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-3] - template_mags[:, :, :, 1:-2]
    templ_tupl = [tuple(_pars) + tuple([z]) for _pars, z in zip(params_arr, zref_arr, strict=True)]
    reslist_of_tupl = tree_map(lambda partup: vmap_clrs_zobs_legacy(jnp.array(partup[:-1]), partup[-1], wls, transm_arr, redz_arr, anu_arr, ssp_data), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_clrs_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    return reslist_of_tupl


def make_legacy_itemplates(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag=3):
    """make_legacy_itemplates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param z_ref: array of redshift of the galaxy used as template
    :type z_ref: JAX-array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param anu_arr: Attenuation grid on which to compute the templates photometry
    :type anu_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # i_mag = template_mags[:, :, :, id_imag]
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-2] - i_mag
    templ_tupl = [tuple(_pars) + tuple([z]) for _pars, z in zip(params_arr, zref_arr, strict=True)]
    reslist_of_tupl = tree_map(lambda partup: vmap_iclrs_zobs_legacy(jnp.array(partup[:-1]), partup[-1], wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_iclrs_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag)
    return reslist_of_tupl


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
