#!/usr/bin/env python

# # Fit Fors2 Spectra and Photometry with DSPS
# Restricted to FORS2 galaxies with GALEX photometry

# Implement this fit using this `fors2tostellarpopsynthesis` package
#
# - Author Joseph Chevalier
# - Afflilation : IJCLab/IN2P3/CNRS
# - Organisation : LSST-DESC
# - creation date : 2024-01-10
# - last update : 2024-01-10 : Initial version
#
# Most functions are inside the package. This code is a synthetic rewrite of the `fit_loop.py` module.

import os
from functools import partial

import jaxopt
import numpy as np
import pandas as pd
from diffmah.defaults import DiffmahParams
from diffstar import calc_sfh_singlegal  # sfh_singlegal
from diffstar.defaults import DiffstarUParams  # , DEFAULT_Q_PARAMS
from dsps import calc_obs_mag
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from interpax import interp1d
from jax import jit, vmap
from jax import numpy as jnp
from jax.scipy.optimize import minimize

from process_fors2.analysis import bpt_classif
from process_fors2.stellarPopSynthesis import SSPParametersFit
from process_fors2.stellarPopSynthesis.met_weights_age_dep import calc_rest_sed_sfh_table_lognormal_mdf_agedep

try:
    from jax.numpy import trapezoid as trapz
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid as trapz
    except ImportError:
        from jax.numpy import trapz


_DUMMY_P_ADQ = SSPParametersFit()
PARS_DF = pd.DataFrame(index=_DUMMY_P_ADQ.PARAM_NAMES_FLAT, columns=["Init", "Min", "Max"])
PARS_DF["Init"] = _DUMMY_P_ADQ.INIT_PARAMS
PARS_DF["Min"] = _DUMMY_P_ADQ.PARAMS_MIN
PARS_DF["Max"] = _DUMMY_P_ADQ.PARAMS_MAX
INIT_PARAMS = jnp.array(PARS_DF["Init"])
PARAMS_MIN = jnp.array(PARS_DF["Min"])
PARAMS_MAX = jnp.array(PARS_DF["Max"])

TODAY_GYR = 13.8
T_ARR = jnp.linspace(0.1, TODAY_GYR, 100)


def prepare_data_arr(attrs_df, selected_tags, wls_arr):
    """prepare_data_arr _summary_

    :param attrs_df: _description_
    :type attrs_df: _type_
    :param selected_tags: _description_
    :type selected_tags: _type_
    :param wls_arr: _description_
    :type wls_arr: _type_
    :return: _description_
    :rtype: _type_
    """
    from process_fors2.stellarPopSynthesis import FilterInfo

    rews_list = sorted([col for col in list(attrs_df.columns) if "REW" in col])
    li_names = np.unique([li.split("_REW")[0] for li in rews_list])
    li_wls = jnp.array([float(ln.split("_")[-1]) for ln in li_names])

    columns = [
        "num",
        "redshift",
        "ra",
        "dec",
        "Classification",
        "fuv_mag",
        "nuv_mag",
        "MAG_GAAP_u",
        "MAG_GAAP_g",
        "MAG_GAAP_r",
        "MAG_GAAP_i",
        "MAG_GAAP_Z",
        "MAG_GAAP_Y",
        "MAG_GAAP_J",
        "MAG_GAAP_H",
        "MAG_GAAP_Ks",
        "fuv_magerr",
        "nuv_magerr",
        "MAGERR_GAAP_u",
        "MAGERR_GAAP_g",
        "MAGERR_GAAP_r",
        "MAGERR_GAAP_i",
        "MAGERR_GAAP_Z",
        "MAGERR_GAAP_Y",
        "MAGERR_GAAP_J",
        "MAGERR_GAAP_H",
        "MAGERR_GAAP_Ks",
    ] + rews_list

    sel_df = attrs_df.loc[selected_tags, columns]

    mags_arr = jnp.array(sel_df[["fuv_mag", "nuv_mag", "MAG_GAAP_u", "MAG_GAAP_g", "MAG_GAAP_r", "MAG_GAAP_i", "MAG_GAAP_Z", "MAG_GAAP_Y", "MAG_GAAP_J", "MAG_GAAP_H", "MAG_GAAP_Ks"]])

    magerrs_arr = jnp.array(
        sel_df[["fuv_magerr", "nuv_magerr", "MAGERR_GAAP_u", "MAGERR_GAAP_g", "MAGERR_GAAP_r", "MAGERR_GAAP_i", "MAGERR_GAAP_Z", "MAGERR_GAAP_Y", "MAGERR_GAAP_J", "MAGERR_GAAP_H", "MAGERR_GAAP_Ks"]]
    )

    rews_arr = jnp.array(sel_df[[c for c in rews_list if "err" not in c]])

    rewerrs_arr = jnp.array(sel_df[[c for c in rews_list if "err" in c]])

    ps = FilterInfo()
    wls, trans = ps.get_2lists()
    transm_arr = jnp.array([interp1d(wls_arr, wl, tr, method="linear", extrap=0.0) for wl, tr in zip(wls, trans, strict=True)])
    list_wlmean_f_sel = jnp.array([f.wave_mean for f in ps.filters_transmissionlist])

    return sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr


@jit
def mean_sfr(params):
    """Model of the SFR

    :param params: Fitted parameter dictionnary
    :type params: float as a dictionnary

    :return: array of the star formation rate
    :rtype: float

    """
    # decode the parameters
    param_mah = params[:4]
    param_ms = params[4:9]
    param_q = params[9:13]

    # compute SFR
    tup_param_sfh = DiffstarUParams(param_ms, param_q)
    tup_param_mah = DiffmahParams(*param_mah)

    sfh_gal = calc_sfh_singlegal(tup_param_sfh, tup_param_mah, T_ARR)

    return sfh_gal


vmap_mean_sfr = vmap(mean_sfr)


@jit
def ssp_spectrum_fromparam(params, z_obs, ssp_data):
    """ssp_spectrum_fromparam _summary_

    :param params: _description_
    :type params: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # compute the SFR
    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
    t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

    gal_sfr_table = mean_sfr(params)

    # age-dependant metallicity
    gal_lgmet_young = 2.0  # log10(Z)
    gal_lgmet_old = -3.0  # params["LGMET_OLD"] # log10(Z)
    gal_lgmet_scatter = 0.2  # params["LGMETSCATTER"] # lognormal scatter in the metallicity distribution function

    # compute the SED_info object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf_agedep(
        T_ARR, gal_sfr_table, gal_lgmet_young, gal_lgmet_old, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs
    )
    # dust attenuation parameters
    Av = params[13]
    uv_bump = params[14]
    plaw_slope = params[15]
    # list_param_dust = [Av, uv_bump, plaw_slope]

    # compute dust attenuation
    wave_spec_micron = ssp_data.ssp_wave / 10000
    k = sbl18_k_lambda(wave_spec_micron, uv_bump, plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k, Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    return ssp_data.ssp_wave, sed_info.rest_sed, sed_attenuated


@jit
def mean_spectrum(wls, params, z_obs, ssp_data):
    """mean_spectrum _summary_

    :param wls: _description_
    :type wls: _type_
    :param params: _description_
    :type params: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # interpolate with interpax which is differentiable
    # Fobs = jnp.interp(wls, ssp_data.ssp_wave, sed_attenuated)
    Fobs = interp1d(wls, ssp_wave, sed_attenuated, method="cubic")

    return Fobs


vmap_mean_spectrum = vmap(mean_spectrum, in_axes=(None, 0, 0, None))


@partial(vmap, in_axes=(None, None, None, 0, None))
def vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs):
    """vmap_calc_obs_mag _summary_

    :param ssp_wave: _description_
    :type ssp_wave: _type_
    :param sed_attenuated: _description_
    :type sed_attenuated: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :return: _description_
    :rtype: _type_
    """
    return calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs, *DEFAULT_COSMOLOGY)


@jit
def mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data):
    """mean_mags _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    mags_predictions = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs)
    # mags_predictions = tree_map(
    #    lambda trans : calc_obs_mag(
    #        ssp_wave,
    #        sed_attenuated,
    #        wls,
    #        trans,
    #        z_obs,
    #        *DEFAULT_COSMOLOGY
    #    ),
    #    tuple(t for t in filt_trans_arr)
    # )

    return jnp.array(mags_predictions)


@jit
def mean_colors(params, wls, filt_trans_arr, z_obs, ssp_data):
    """mean_colors _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    mags = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    return mags[:-1] - mags[1:]


vmap_mean_mags = vmap(mean_mags, in_axes=(0, None, None, 0, None))

vmap_mean_colors = vmap(mean_colors, in_axes=(0, None, None, 0, None))


@jit
def calc_eqw(sur_wls, sur_spec, lin):
    r"""
    Computes the equivalent width of the specified spectral line.

    Parameters
    ----------
    p : array
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`.
    sur_wls : array
        Wavelengths in angstrom - should be oversampled so that spectral lines can be sampled with a sufficiently high resolution (step of 0.1 angstrom is recommended)
    sur_spec : array
        Flux densities in Lsun/Hz - should be oversampled to match `sur_wls`.
    lin : int or float
        Central wavelength (in angstrom) of the line to be studied.

    Returns
    -------
    float
        Value of the nequivalent width of spectral line at $\lambda=$`lin`.
    """
    from process_fors2.analysis import C_KMS

    line_wid = lin * 300 / C_KMS / 2
    cont_wid = lin * 1500 / C_KMS / 2
    nancont = jnp.where(jnp.logical_or(jnp.logical_and(sur_wls > lin - cont_wid, sur_wls < lin - line_wid), jnp.logical_and(sur_wls > lin + line_wid, sur_wls < lin + cont_wid)), sur_spec, jnp.nan)
    height = jnp.nanmean(nancont)
    vals = jnp.where(jnp.logical_and(sur_wls > lin - line_wid, sur_wls < lin + line_wid), sur_spec / height - 1.0, 0.0)
    ew = trapz(vals, x=sur_wls)
    return ew


vmap_calc_eqw = vmap(calc_eqw, in_axes=(None, None, 0))


@jit
def chi_term(ref, obs, sig):
    """chi_term _summary_

    :param ref: _description_
    :type ref: _type_
    :param obs: _description_
    :type obs: _type_
    :param sig: _description_
    :type sig: _type_
    :return: _description_
    :rtype: _type_
    """
    return jnp.power((ref - obs) / sig, 2)


@jit
def red_chi2(ref_arr, obs_arr, sig_arr):
    """red_chi2 _summary_

    :param ref_arr: _description_
    :type ref_arr: _type_
    :param obs_arr: _description_
    :type obs_arr: _type_
    :param sig_arr: _description_
    :type sig_arr: _type_
    :return: _description_
    :rtype: _type_
    """
    non_nan_obs = jnp.where(jnp.logical_and(jnp.isfinite(sig_arr), jnp.logical_and(jnp.isfinite(ref_arr), jnp.isfinite(obs_arr))), obs_arr, 0.0)

    non_nan_sig = jnp.where(jnp.logical_and(jnp.isfinite(sig_arr), jnp.logical_and(jnp.isfinite(ref_arr), jnp.isfinite(obs_arr))), sig_arr, 1.0)

    non_nan_ref = jnp.where(jnp.logical_and(jnp.isfinite(sig_arr), jnp.logical_and(jnp.isfinite(ref_arr), jnp.isfinite(obs_arr))), ref_arr, 0.0)

    chi2s = chi_term(non_nan_ref, non_nan_obs, non_nan_sig)
    no_nan = jnp.where(jnp.logical_and(jnp.isfinite(sig_arr), jnp.logical_and(jnp.isfinite(ref_arr), jnp.isfinite(obs_arr))), 1, 0)
    _count = jnp.sum(no_nan)
    return jnp.where(_count > 0, jnp.nansum(chi2s) / _count, 1.0e15)


@jit
def lik_rew(params, surwls, rews_wls, rews, rews_err, z_obs, ssp_data):
    """lik_rew _summary_

    :param params: _description_
    :type params: _type_
    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    spec = mean_spectrum(surwls, params, z_obs, ssp_data)
    rew_predictions = vmap_calc_eqw(surwls, spec, rews_wls)
    redchi2 = red_chi2(rew_predictions, rews, rews_err)
    return redchi2


@jit
def lik_mag(params, wls, filt_trans_arr, mags_measured, sigma_mag_obs, z_obs, ssp_data):
    """lik_mag _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param mags_measured: _description_
    :type mags_measured: _type_
    :param sigma_mag_obs: _description_
    :type sigma_mag_obs: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    all_mags_predictions = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_mags_predictions, mags_measured, sigma_mag_obs)
    return redchi2


@jit
def lik_mag_z_anu(z_anu, fixed_pars, wls, filt_trans_arr, mags_measured, sigma_mag_obs, ssp_data):
    """lik_mag_z_anu _summary_

    :param z_anu: _description_
    :type z_anu: _type_
    :param fixed_pars: _description_
    :type fixed_pars: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param mags_measured: _description_
    :type mags_measured: _type_
    :param sigma_mag_obs: _description_
    :type sigma_mag_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    z_obs, anu = z_anu
    params = jnp.column_stack((fixed_pars[:13], jnp.array(anu), fixed_pars[-2:]))
    all_mags_predictions = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_mags_predictions, mags_measured, sigma_mag_obs)
    return redchi2


@jit
def lik_colr(params, wls, filt_trans_arr, clrs_measured, sigma_clr_obs, z_obs, ssp_data):
    """lik_mag _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param clrs_measured: _description_
    :type clrs_measured: _type_
    :param sigma_clr_obs: _description_
    :type sigma_clr_obs: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    all_clrs_predictions = mean_colors(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_clrs_predictions, clrs_measured, sigma_clr_obs)
    return redchi2


@jit
def lik_colr_z_anu(z_anu, fixed_pars, wls, filt_trans_arr, clrs_measured, sigma_clr_obs, ssp_data):
    """lik_mag_z_anu _summary_

    :param z_anu: _description_
    :type z_anu: _type_
    :param fixed_pars: _description_
    :type fixed_pars: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param clrs_measured: _description_
    :type clrs_measured: _type_
    :param sigma_clr_obs: _description_
    :type sigma_clr_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    z_obs, anu = z_anu
    params = jnp.column_stack((fixed_pars[:13], jnp.array(anu), fixed_pars[-2:]))
    all_clrs_predictions = mean_colors(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_clrs_predictions, clrs_measured, sigma_clr_obs)
    return redchi2


@jit
def lik_mag_rew(params, wls, filt_trans_arr, mags_measured, sigma_mag_obs, surwls, rews_wls, rews, rews_err, z_obs, ssp_data, weight_mag):
    """lik_mag_rew _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param mags_measured: _description_
    :type mags_measured: _type_
    :param sigma_mag_obs: _description_
    :type sigma_mag_obs: _type_
    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param weight_mag: _description_
    :type weight_mag: _type_
    :return: _description_
    :rtype: _type_
    """
    resid_spec = lik_rew(params, surwls, rews_wls, rews, rews_err, z_obs, ssp_data)
    resid_phot = lik_mag(params, wls, filt_trans_arr, mags_measured, sigma_mag_obs, z_obs, ssp_data)

    return weight_mag * resid_phot + (1 - weight_mag) * resid_spec


def fit_mags(fwls, filts_transm, omags, omagerrs, zobs, ssp_data):
    """fit_mags _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param omags: _description_
    :type omags: _type_
    :param omagerrs: _description_
    :type omagerrs: _type_
    :param zobs: _description_
    :type zobs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    lbfgsb_mag = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=1000)
    pars_m, stat_m = lbfgsb_mag.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), fwls, filts_transm, omags, omagerrs, zobs, ssp_data)

    # Convert fitted parameters into a dictionnary
    # params_m = res_m.params

    # convert into a dictionnary
    # dict_out = {"fit_params": params_m, "zobs": zobs}
    return pars_m  # params_m


# vmap_fit_mags = vmap(fit_mags, in_axes=(None, None, 0, 0, 0, None))


def fit_rews(surwls, rews_wls, rews, rews_err, z_obs, ssp_data):
    """fit_rews _summary_

    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    lbfgsb_rews = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=1000)
    pars_r, stat_r = lbfgsb_rews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), surwls, rews_wls, rews, rews_err, z_obs, ssp_data)

    # Convert fitted parameters into a dictionnary
    # params_m = res_m.params

    # convert into a dictionnary
    # dict_out = {"fit_params": params_m, "zobs": zobs}
    return pars_r  # params_m


# vmap_fit_rews = vmap(fit_rews, in_axes=(None, None, 0, 0, 0, None))


def fit_mags_rews(fwls, filts_transm, omags, omagerrs, surwls, rews_wls, rews, rews_err, z_obs, ssp_data, weight_mag):
    """fit_mags_rews _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param omags: _description_
    :type omags: _type_
    :param omagerrs: _description_
    :type omagerrs: _type_
    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param weight_mag: _description_
    :type weight_mag: _type_
    :return: _description_
    :rtype: _type_
    """
    lbfgsb_rews = jaxopt.ScipyBoundedMinimize(fun=lik_mag_rew, method="L-BFGS-B", maxiter=1000)
    pars_mr, stat_mr = lbfgsb_rews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), fwls, filts_transm, omags, omagerrs, surwls, rews_wls, rews, rews_err, z_obs, ssp_data, weight_mag)

    # Convert fitted parameters into a dictionnary
    # params_m = res_m.params

    # convert into a dictionnary
    # dict_out = {"fit_params": params_m, "zobs": zobs}
    return pars_mr  # params_m


# vmap_fit_mags_rews = vmap(fit_mags_rews, in_axes=(None, None, 0, 0, None, None, 0, 0, 0, None, None))


def vmap_fit_mags(fwls, filts_transm, omags, omagerrs, zobs, ssp_data):
    """vmap_fit_mags _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param omags: _description_
    :type omags: _type_
    :param omagerrs: _description_
    :type omagerrs: _type_
    :param zobs: _description_
    :type zobs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    """

    @jit
    def solve(_omags, _oerrs, _oz):
        res_m = minimize(lik_mag, INIT_PARAMS, (fwls, filts_transm, _omags, _oerrs, _oz, ssp_data), method="BFGS")
        return res_m.x

    vsolve = vmap(solve, in_axes=(0, 0, 0))
    return vsolve(omags, omagerrs, zobs)  # params_m


def vmap_fit_mags_z_anu(fixed_pars, fwls, filts_transm, omags, omagerrs, ssp_data):
    """vmap_fit_mags _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param omags: _description_
    :type omags: _type_
    :param omagerrs: _description_
    :type omagerrs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    """

    @jit
    def solve(_omags, _oerrs):
        res_m = minimize(lik_mag_z_anu, (0.5, INIT_PARAMS[13]), (fixed_pars, fwls, filts_transm, _omags, _oerrs, ssp_data), method="BFGS")
        return res_m.x

    vsolve = vmap(solve, in_axes=(0, 0))
    return vsolve(omags, omagerrs)  # params_m


def vmap_fit_colrs_z_anu(fixed_pars, fwls, filts_transm, ocolrs, ocolrerrs, ssp_data):
    """vmap_fit_mags _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param ocolrs: _description_
    :type ocolrs: _type_
    :param ocolrerrs: _description_
    :type ocolrerrs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    """

    @jit
    def solve(_ocolrs, _oerrs):
        res_m = minimize(lik_colr_z_anu, (0.5, INIT_PARAMS[13]), (fixed_pars, fwls, filts_transm, _ocolrs, _oerrs, ssp_data), method="BFGS")
        return res_m.x

    vsolve = vmap(solve, in_axes=(0, 0))
    return vsolve(ocolrs, ocolrerrs)  # params_m


def vmap_fit_rews(surwls, rews_wls, rews, rews_err, zobs, ssp_data):
    """vmap_fit_rews _summary_

    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param zobs: _description_
    :type zobs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    """

    @jit
    def solve(_rews, _rewerrs, _z):
        res_r = minimize(lik_rew, INIT_PARAMS, (surwls, rews_wls, _rews, _rewerrs, _z, ssp_data), method="BFGS")
        return res_r.x

    vsolve = vmap(solve, in_axes=(0, 0, 0))
    return vsolve(rews, rews_err, zobs)


def vmap_fit_mags_rews(fwls, filts_transm, omags, omagerrs, surwls, rews_wls, rews, rews_err, zobs, ssp_data, weight_mag):
    """vmap_fit_mags_rews _summary_

    :param fwls: _description_
    :type fwls: _type_
    :param filts_transm: _description_
    :type filts_transm: _type_
    :param omags: _description_
    :type omags: _type_
    :param omagerrs: _description_
    :type omagerrs: _type_
    :param surwls: _description_
    :type surwls: _type_
    :param rews_wls: _description_
    :type rews_wls: _type_
    :param rews: _description_
    :type rews: _type_
    :param rews_err: _description_
    :type rews_err: _type_
    :param zobs: _description_
    :type zobs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param weight_mag: _description_
    :type weight_mag: _type_
    """

    @jit
    def solve(_omags, _oerrs, _rews, _rewerrs, _oz):
        res_mr = minimize(lik_mag_rew, INIT_PARAMS, (fwls, filts_transm, _omags, _oerrs, surwls, rews_wls, _rews, _rewerrs, _oz, ssp_data, weight_mag), method="BFGS")
        return res_mr.x

    vsolve = vmap(solve, in_axes=(0, 0, 0, 0, 0))
    return vsolve(omags, omagerrs, rews, rews_err, zobs)  # params_m


def filter_tags_df(attrs_df, remove_visible=False, remove_galex=False, remove_galex_fuv=True):
    """filter_tags_df Function to filter galaxies to fit according to their available photometry.

    :param attrs_df: DataFrame of attributes - must contain KiDS and GALEX photometry keywords in its columns.
    :type attrs_df: DataFrame
    :param remove_visible: Whether to remove galaxies with photometry in the visible range of the EM spectrum, defaults to False
    :type remove_visible: bool, optional
    :param remove_galex: Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum, defaults to False
    :type remove_galex: bool, optional
    :param remove_galex_fuv: Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum, defaults to True
    :type remove_galex_fuv: bool, optional
    :return: List of applicable tags after filtering, to be used as indices in the original `attrs_df` DataFrame for instance.
    :rtype: list
    """
    # ## Select applicable spectra
    filtered_tags = []
    for tag, fors2_attr in attrs_df.iterrows():
        bool_viz = remove_visible or (
            not (remove_visible)
            and np.isfinite(fors2_attr["MAG_GAAP_u"])
            and np.isfinite(fors2_attr["MAG_GAAP_g"])
            and np.isfinite(fors2_attr["MAG_GAAP_r"])
            and np.isfinite(fors2_attr["MAG_GAAP_i"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_u"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_g"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_r"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_i"])
        )

        bool_fuv = (remove_galex or remove_galex_fuv) or (not (remove_galex or remove_galex_fuv) and np.isfinite(fors2_attr["fuv_mag"]) and np.isfinite(fors2_attr["fuv_magerr"]))

        bool_nuv = remove_galex or (not (remove_galex) and np.isfinite(fors2_attr["nuv_mag"]) and np.isfinite(fors2_attr["nuv_magerr"]))

        if bool_viz and bool_fuv and bool_nuv:
            filtered_tags.append(tag)
    print(f"Number of galaxies in the sample : {len(filtered_tags)}.")
    return filtered_tags


def fit_vmap(xmatch_h5, gelato_h5, fit_type="mags", low_bound=0, high_bound=None, ssp_file=None, weight_mag=0.5, remove_visible=False, remove_galex=False, remove_galex_fuv=True, quiet=False):
    """fit_vmap Function to fit a stellar population onto observations of galaxies, using a vmapped algorithm on JAX arrays.

    :param xmatch_h5: Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    :type xmatch_h5: path or str
    :param gelato_h5: Path to the HDF5 file gathering outputs from GELATO run.
    :type gelato_h5: path or str
    :param fit_type: Data to fit the SPS on. Must be one of :
            - 'mags' to fit on KiDS+VIKING+GALEX photometry
            - 'rews' to fit on Restframe Equivalent Widths of spectral emission/absorption lines as detected and computed by GELATO
            - 'mags+rews' to fit on both magnitudes and Restframe Equivalent Widths. The weight associated to each likelihood can be controlled with the optional parameter `weight_mag`.
            Defaults to "mags"
    :type fit_type: str, optional
    :param low_bound: If fitting a slice of the original data : the index of the first element (natural count : starts at 1, ends at nb of elements), defaults to 0
    :type low_bound: int, optional
    :param high_bound: If fitting a slice of the original data : the index of the last element (natural count : starts at 1, ends at nb of elements).
            If None, all galaxies are fitted starting with `low_bound`, defaults to None
    :type high_bound: int, optional
    :param ssp_file: SSP library location. If None, loads the defaults file from `process_fors2.fetchData`, defaults to None
    :type ssp_file: path or str, optional
    :param weight_mag: Weight of the fit on photometry. 1-weight_mag is affected to the fit on rest equivalent widths. Must be between 0.0 and 1.0, defaults to 0.5
    :type weight_mag: float, optional
    :param remove_visible: Whether to remove galaxies with photometry in the visible range of the EM spectrum, defaults to False
    :type remove_visible: bool, optional
    :param remove_galex: Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum, defaults to False
    :type remove_galex: bool, optional
    :param remove_galex_fuv: Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum, defaults to True
    :type remove_galex_fuv: bool, optional
    :param quiet: Whether to silence some prints (for convenience while running in loops for instance), defaults to False
    :type quiet: bool, optional
    :return: The properties of fitted galaxies in a dataframe, the array of SPS parameters and the boundaries of the selected slice of the set of galaxies.
    :rtype: tuple of (DataFrame, array, int, int)
    """
    from process_fors2.stellarPopSynthesis import load_ssp

    ssp_data = load_ssp(ssp_file)
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs_df = bpt_classif(gelatoh5, xmatchh5, use_nc=False, return_dict=False)

    # ## Select applicable spectra
    filtered_tags = filter_tags_df(merged_attrs_df, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)

    if high_bound is None:
        high_bound = len(filtered_tags)
    else:
        high_bound = min(high_bound, len(filtered_tags))
        high_bound = max(1, high_bound)
    low_bound = max(0, low_bound - 1)
    low_bound = min(low_bound, high_bound - 1)

    selected_tags = filtered_tags[low_bound:high_bound]
    if not quiet:
        print(f"Number of galaxies to be fitted : {len(selected_tags)}.")

    wls_interp = jnp.arange(100.0, 25000.1, 10)
    wls_rews = jnp.arange(1000.0, 10000, 0.1)

    sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr = prepare_data_arr(merged_attrs_df, selected_tags, wls_interp)
    zs = jnp.array(sel_df["redshift"])

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results_arr = vmap_fit_mags_rews(wls_interp, transm_arr, mags_arr, magerrs_arr, wls_rews, li_wls, rews_arr, rewerrs_arr, zs, ssp_data, weight_mag)
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results_arr = vmap_fit_rews(wls_rews, li_wls, rews_arr, rewerrs_arr, zs, ssp_data)
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        fit_results_arr = vmap_fit_mags(wls_interp, transm_arr, mags_arr, magerrs_arr, zs, ssp_data)

    return sel_df, fit_results_arr, low_bound, high_bound


def fit_treemap(xmatch_h5, gelato_h5, fit_type="mags", low_bound=0, high_bound=None, ssp_file=None, weight_mag=0.5, remove_visible=False, remove_galex=False, remove_galex_fuv=True, quiet=False):
    """fit_treemap _summary_

    :param xmatch_h5: _description_
    :type xmatch_h5: _type_
    :param gelato_h5: _description_
    :type gelato_h5: _type_
    :param fit_type: _description_, defaults to "mags"
    :type fit_type: str, optional
    :param low_bound: _description_, defaults to 0
    :type low_bound: int, optional
    :param high_bound: _description_, defaults to None
    :type high_bound: _type_, optional
    :param ssp_file: _description_, defaults to None
    :type ssp_file: _type_, optional
    :param weight_mag: _description_, defaults to 0.5
    :type weight_mag: float, optional
    :param remove_visible: _description_, defaults to False
    :type remove_visible: bool, optional
    :param remove_galex: _description_, defaults to False
    :type remove_galex: bool, optional
    :param remove_galex_fuv: _description_, defaults to True
    :type remove_galex_fuv: bool, optional
    :param quiet: _description_, defaults to False
    :type quiet: bool, optional
    :return: _description_
    :rtype: _type_
    """
    from jax.tree_util import tree_map

    from process_fors2.stellarPopSynthesis import load_ssp

    ssp_data = load_ssp(ssp_file)
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs_df = bpt_classif(gelatoh5, xmatchh5, use_nc=False, return_dict=False)

    # ## Select applicable spectra
    filtered_tags = filter_tags_df(merged_attrs_df, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)

    if high_bound is None:
        high_bound = len(filtered_tags)
    else:
        high_bound = min(high_bound, len(filtered_tags))
        high_bound = max(1, high_bound)
    low_bound = max(0, low_bound - 1)
    low_bound = min(low_bound, high_bound - 1)

    selected_tags = filtered_tags[low_bound:high_bound]
    if not quiet:
        print(f"Number of galaxies to be fitted : {len(selected_tags)}.")

    wls_interp = jnp.arange(100.0, 25000.1, 10)
    wls_rews = jnp.arange(1000.0, 10000, 0.1)

    sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr = prepare_data_arr(merged_attrs_df, selected_tags, wls_interp)
    zs = jnp.array(sel_df["redshift"])

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_magrews = jaxopt.ScipyBoundedMinimize(fun=lik_mag_rew, method="L-BFGS-B", maxiter=1000)

        @jit
        def solve(arg_tupl):
            omags, omagerrs, rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_magrews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data, weight_mag)
            return pars

        fit_results_tree = tree_map(lambda otupl: solve(otupl), tuple([(ma, mer, rew, rer, z) for ma, mer, rew, rer, z in zip(mags_arr, magerrs_arr, rews_arr, rewerrs_arr, zs, strict=True)]))
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_rews = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=1000)

        @jit
        def solve(arg_tupl):
            rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_rews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data)
            return pars

        fit_results_tree = tree_map(lambda otupl: solve(otupl), tuple([(rew, rer, z) for rew, rer, z in zip(rews_arr, rewerrs_arr, zs, strict=True)]))
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        lbfgsb_mags = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=1000)

        @jit
        def solve(arg_tupl):
            omags, omagerrs, zobs = arg_tupl
            pars, stat = lbfgsb_mags.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, zobs, ssp_data)
            return pars

        fit_results_tree = tree_map(lambda otupl: solve(otupl), tuple([(ma, mer, z) for ma, mer, z in zip(mags_arr, magerrs_arr, zs, strict=True)]))

    return sel_df, fit_results_tree, low_bound, high_bound


def vmapFitsToHDF5(df_outfilename, ref_df, fit_res_arr):
    """vmapFitsToHDF5 _summary_

    :param df_outfilename: _description_
    :type df_outfilename: _type_
    :param ref_df: _description_
    :type ref_df: _type_
    :param fit_res_arr: _description_
    :type fit_res_arr: _type_
    :return: _description_
    :rtype: _type_
    """
    res_df = pd.DataFrame(index=ref_df.index, columns=_DUMMY_P_ADQ.PARAM_NAMES_FLAT, data=fit_res_arr)
    out_df = ref_df.join(res_df, how="inner")
    outpath = os.path.abspath(df_outfilename)
    out_df.to_hdf(outpath, key="fit_dsps")
    ret = outpath
    if not os.path.isfile(outpath):
        ret = f"Unable to write file to {outpath}. Please check that the run finished correctly."
    return ret


def readVmapFitsFromHDF5(dspsFitsH5, group="fit_dsps"):
    """readVmapFitsFromHDF5 _summary_

    :param dspsFitsH5: _description_
    :type dspsFitsH5: _type_
    :param group: _description_, defaults to "fit_dsps"
    :type group: str, optional
    :return: _description_
    :rtype: _type_
    """
    fitres_df = pd.read_hdf(os.path.abspath(dspsFitsH5), key=group)
    fitres_df = fitres_df[_DUMMY_P_ADQ.PARAM_NAMES_FLAT + ["redshift"]]
    sps_params_dict = fitres_df.to_dict("index")
    for key, dico in sps_params_dict.items():
        dico.update({"tag": key})
    return sps_params_dict


def main(args):
    """
    Function that goes through the whole fitting process, callable from outside.

    Parameters
    ----------
    args : list, tuple or array
        Arguments to be passed to the function as command line arguments.
        Mandatory arguments are 1- path to the HDF5 file of cross-matched data and 2- path to the HDF5 file of GELATO outputs.
        Optional argument is 3- path to a `JSON` configuration file similar to that in `$FORS2DATALOC/defaults.json`.

    Returns
    -------
    int
        0 if exited correctly.
    """
    from process_fors2.fetchData import json_to_inputs
    from process_fors2.fetchData.queries import FORS2DATALOC

    conf_json = args[3] if len(args) > 3 else os.path.join(FORS2DATALOC, "defaults.json")  # attention à la localisation du fichier !

    xmatchh5 = args[1]  # le premier argument de args est toujours `__main__.py`
    gelatoh5 = args[2]
    inputs = json_to_inputs(conf_json)["fitDSPS"]
    _fit_type = inputs["fit_type"]
    _weight_mag = inputs["weight_mag"]  # Only for combined fit : mags + rews
    _ssp_file = None if (inputs["ssp_file"].lower() == "default" or inputs["ssp_file"] == "" or inputs["ssp_file"] is None) else os.path.abspath(inputs["ssp_file"])

    _low = inputs["first_spec"]
    _high = None if inputs["last_spec"] < 0 else inputs["last_spec"]

    sel_df, fit_results_arr, low_bound, high_bound = fit_vmap(
        xmatchh5,
        gelatoh5,
        fit_type=_fit_type,
        low_bound=_low,
        high_bound=_high,
        ssp_file=_ssp_file,
        weight_mag=_weight_mag,
        remove_visible=inputs["remove_visible"],
        remove_galex=inputs["remove_galex"],
        remove_galex_fuv=inputs["remove_fuv"],
        quiet=False,
    )

    outdir = os.path.abspath(f"./DSPS_hdf5_VMAPfit_{_fit_type}")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    filename_params = os.path.join(outdir, f"fitparams_{_fit_type}_{low_bound+1}_to_{high_bound}.h5")
    status = vmapFitsToHDF5(filename_params, sel_df, fit_results_arr)
    print(status)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
