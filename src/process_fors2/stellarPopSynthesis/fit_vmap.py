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

import copy
import os
from functools import partial

import h5py
import jax
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diffmah.defaults import DiffmahParams
from diffstar import calc_sfh_singlegal  # sfh_singlegal
from diffstar.defaults import DiffstarUParams  # , DEFAULT_Q_PARAMS
from dsps import calc_obs_mag, calc_rest_mag, load_ssp_templates
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z, age_at_z0
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from interpax import interp1d
from jax import jit, vmap
from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax.tree_util import tree_map
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from process_fors2.analysis import C_KMS, bpt_classif
from process_fors2.stellarPopSynthesis import SSPParametersFit
from process_fors2.stellarPopSynthesis.met_weights_age_dep import calc_rest_sed_sfh_table_lognormal_mdf_agedep

try:
    from jax.numpy import trapezoid as trapz
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid as trapz
    except ImportError:
        from jax.numpy import trapz

jax.config.update("jax_enable_x64", True)

plt.style.use("default")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.labelsize"] = "x-large"
plt.rcParams["axes.titlesize"] = "x-large"
plt.rcParams["xtick.labelsize"] = "x-large"
plt.rcParams["ytick.labelsize"] = "x-large"
plt.rcParams["legend.fontsize"] = 12

_DUMMY_P_ADQ = SSPParametersFit()
PARS_DF = pd.DataFrame(index=_DUMMY_P_ADQ.PARAM_NAMES_FLAT, columns=["Init", "Min", "Max"])
PARS_DF["Init"] = _DUMMY_P_ADQ.INIT_PARAMS
PARS_DF["Min"] = _DUMMY_P_ADQ.PARAMS_MIN
PARS_DF["Max"] = _DUMMY_P_ADQ.PARAMS_MAX
INIT_PARAMS = jnp.array(PARS_DF["Init"])
PARAMS_MIN = jnp.array(PARS_DF["Min"])
PARAMS_MAX = jnp.array(PARS_DF["Max"])

TODAY_GYR = age_at_z0(*DEFAULT_COSMOLOGY)  # 13.8
T_ARR = jnp.linspace(0.1, TODAY_GYR, 100)


def load_ssp(ssp_file=None):
    """load_ssp _summary_

    :param ssp_file: _description_, defaults to None
    :type ssp_file: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if ssp_file == "" or ssp_file is None or "default" in ssp_file.lower():
        from process_fors2.fetchData import DEFAULTS_DICT

        fullfilename_ssp_data = DEFAULTS_DICT["DSPS HDF5"]
    else:
        fullfilename_ssp_data = os.path.abspath(ssp_file)
    ssp_data = load_ssp_templates(fn=fullfilename_ssp_data)
    return ssp_data


def istuple(tree):
    """istuple _summary_

    :param tree: _description_
    :type tree: _type_
    :return: _description_
    :rtype: _type_
    """
    return isinstance(tree, tuple)


def has_redshift(dic):
    """
    Utility to detect a leaf in a dictionary (tree) based on the assumption that a leaf is a dictionary that contains individual information linked to a spectrum, such as the redshift of the galaxy.

    Parameters
    ----------
    dic : dictionary
        Dictionary with data. Within the context of this function's use, this is an output of the catering of data to fit on DSPS.
        This function is applied to a global dictionary (tree) and its sub-dictionaries (leaves - as identified by this function).

    Returns
    -------
    bool
        `True` if `'redshift'` is in `dic.keys()` - making it a leaf - `False` otherwise.
    """
    return "redshift" in list(dic.keys())


def prepare_data_arr(attrs_df, selected_tags, wls_arr, source="FORS2"):
    """prepare_data_arr _summary_

    :param attrs_df: _description_
    :type attrs_df: _type_
    :param selected_tags: _description_
    :type selected_tags: _type_
    :param wls_arr: _description_
    :type wls_arr: _type_
    :param source: _description_, defaults to "FORS2"
    :type source: str, optional
    :return: _description_
    :rtype: _type_
    """

    rews_list = sorted([col for col in list(attrs_df.columns) if "rew" in col.lower()])
    li_names = np.unique([li.split("_REW")[0] for li in rews_list])
    li_wls = jnp.array([float(ln.split("_")[-1]) for ln in li_names])

    mags_list = sorted([col for col in list(attrs_df.columns) if "mag" in col.lower() and "image" not in col.lower()])

    if "fors2" in source.lower():
        mags_list = [c for c in mags_list if "Rmag" not in c]

    columns = (
        [
            "num",
            "redshift",
            "ra",
            "dec",
            "Classification",
            "rChi2",
            "CAT_NII",
            "CAT_SII",
            "CAT_OI",
            "CAT_OIII/OIIvsOI",
        ]
        + mags_list
        + rews_list
    )

    if "gogreen" in source.lower():
        columns = ["cluster", "specid"] + columns

    if "desi" in source.lower():
        columns = ["survey", "program", "specid", "morphtype", "LRG", "ELG", "QSO", "BGS", "MWS", "SCND"] + columns

    sel_df = attrs_df.loc[selected_tags, columns]

    mags_arr = jnp.array(sel_df[[c for c in mags_list if "err" not in c.lower()]])
    magerrs_arr = jnp.array(sel_df[[c for c in mags_list if "err" in c.lower()]])

    rews_arr = jnp.array(sel_df[[c for c in rews_list if "err" not in c.lower()]])
    rewerrs_arr = jnp.array(sel_df[[c for c in rews_list if "err" in c.lower()]])

    if "fors2" in source.lower():
        from process_fors2.fetchData import load_filters_from_f2df

        _, transm_arr, list_wlmean_f_sel = load_filters_from_f2df(sel_df, wls_arr)
    else:  # elif "gogreen" in source.lower(): # the DESI case should be covered by any of these two functions, let's pick GOGREEN.
        from process_fors2.fetchData import load_filters_from_ggdf

        _, transm_arr, list_wlmean_f_sel = load_filters_from_ggdf(sel_df, wls_arr)

    return sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr


def prepare_bootstrap_arr(attrs_df, selected_tags, wls_arr, source="FORS2", n_fits=10, bs_type="mags"):
    """prepare_bootstrap_arr _summary_

    :param attrs_df: _description_
    :type attrs_df: _type_
    :param selected_tags: _description_
    :type selected_tags: _type_
    :param wls_arr: _description_
    :type wls_arr: _type_
    :param source: _description_, defaults to "FORS2"
    :type source: str, optional
    :param n_fits: _description_, defaults to 10
    :type n_fits: int, optional
    :param bs_type: _description_, defaults to "mags"
    :type bs_type: str, optional
    :return: _description_
    :rtype: _type_
    """
    rews_list = sorted([col for col in list(attrs_df.columns) if "rew" in col.lower()])
    li_names = np.unique([li.split("_REW")[0] for li in rews_list])
    li_wls = jnp.array([float(ln.split("_")[-1]) for ln in li_names])

    mags_list = sorted([col for col in list(attrs_df.columns) if "mag" in col.lower() and "image" not in col.lower()])

    if "fors2" in source.lower():
        mags_list = [c for c in mags_list if "Rmag" not in c]

    columns = (
        [
            "num",
            "redshift",
            "ra",
            "dec",
            "Classification",
            "rChi2",
            "CAT_NII",
            "CAT_SII",
            "CAT_OI",
            "CAT_OIII/OIIvsOI",
        ]
        + mags_list
        + rews_list
    )

    if "gogreen" in source.lower():
        columns = ["cluster", "specid"] + columns

    if "desi" in source.lower():
        columns = ["survey", "program", "specid", "morphtype", "LRG", "ELG", "QSO", "BGS", "MWS", "SCND"] + columns

    sel_df = attrs_df.loc[selected_tags, columns]

    mags_arr = jnp.array(sel_df[[c for c in mags_list if "err" not in c.lower()]])
    magerrs_arr = jnp.array(sel_df[[c for c in mags_list if "err" in c.lower()]])

    rews_arr = jnp.array(sel_df[[c for c in rews_list if "err" not in c.lower()]])
    rewerrs_arr = jnp.array(sel_df[[c for c in rews_list if "err" in c.lower()]])

    if "fors2" in source.lower():
        from process_fors2.fetchData import load_filters_from_f2df

        _, transm_arr, list_wlmean_f_sel = load_filters_from_f2df(sel_df, wls_arr)
    else:  # elif "gogreen" in source.lower(): # the DESI case should be covered by any of these two functions, let's pick GOGREEN.
        from process_fors2.fetchData import load_filters_from_ggdf

        _, transm_arr, list_wlmean_f_sel = load_filters_from_ggdf(sel_df, wls_arr)

    jkey = jax.random.key(717)

    all_mags_arr, all_magerrs_arr, all_rews_arr, all_rewerrs_arr = [], [], [], []

    for idx, (_meanmag, _stdmag, _meanrew, _stdrew) in enumerate(zip(mags_arr, magerrs_arr, rews_arr, rewerrs_arr, strict=True)):
        # indiv_df = pd.DataFrame(columns=[mags_list, rews_list])
        _mags = jnp.tile(_meanmag, reps=(n_fits, 1))
        _merrs = jnp.tile(_stdmag, reps=(n_fits, 1))
        if "mag" in bs_type.lower():
            jkey, jsubkey = jax.random.split(jkey)
            rnd_norm = jax.random.normal(jsubkey, shape=_mags.shape)
            rnd_mags = rnd_norm * _merrs + _mags
            all_mags_arr.append(rnd_mags)
        else:
            all_mags_arr.append(_mags)
        all_magerrs_arr.append(_merrs)

        _rews = jnp.tile(_meanrew, reps=(n_fits, 1))
        _rerrs = jnp.tile(_stdrew, reps=(n_fits, 1))
        if "rew" in bs_type.lower():
            jkey, jsubkey = jax.random.split(jkey)
            rnd_norm = jax.random.normal(jsubkey, shape=_rews.shape)
            rnd_rews = rnd_norm * _rerrs + _rews
            all_rews_arr.append(rnd_rews)
        else:
            all_rews_arr.append(_rews)
        all_rewerrs_arr.append(_rerrs)

    return sel_df, tuple(all_mags_arr), tuple(all_magerrs_arr), tuple(all_rews_arr), tuple(all_rewerrs_arr), li_wls, list_wlmean_f_sel, transm_arr


@jit
def mean_sfr(params):
    """Model of the SFR

    :param params: Fitted parameters array
    :type params: array of floats

    :return: array of the star formation rate
    :rtype: array

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

    # age-dependant metallicity, log10(Z)
    gal_lgmet_young = params.at[16].get()  # 2.0
    gal_lgmet_old = params.at[17].get()  # -3.0  # params["LGMET_OLD"]
    gal_lgmet_scatter = 0.2  # params["LGMETSCATTER"] # lognormal scatter in the metallicity distribution function

    # compute the SED_info object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf_agedep(
        T_ARR, gal_sfr_table, gal_lgmet_young, gal_lgmet_old, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs
    )
    # dust attenuation parameters
    Av = params.at[13].get()
    uv_bump = params.at[14].get()
    plaw_slope = params.at[15].get()
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
    Fobs = interp1d(wls, ssp_wave, sed_attenuated, method="akima", extrap=False)

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


@partial(vmap, in_axes=(None, None, None, 0))
def vmap_calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr):
    """vmap_calc_obs_mag _summary_

    :param ssp_wave: _description_
    :type ssp_wave: _type_
    :param sed_attenuated: _description_
    :type sed_attenuated: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :return: _description_
    :rtype: _type_
    """
    return calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr)


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
    from process_fors2.analysis import C_KMS, lsunPerHz_to_flam_noU

    line_wid = lin * 400 / C_KMS / 2
    cont_wid = lin * 15000 / C_KMS / 2
    sur_flam = lsunPerHz_to_flam_noU(sur_wls, sur_spec, 0.001)
    nancont = jnp.where(jnp.logical_or(jnp.logical_and(sur_wls > lin - cont_wid, sur_wls < lin - line_wid), jnp.logical_and(sur_wls > lin + line_wid, sur_wls < lin + cont_wid)), sur_flam, jnp.nan)
    height = jnp.nanmean(nancont)
    vals = jnp.where(jnp.logical_and(sur_wls > lin - line_wid, sur_wls < lin + line_wid), sur_flam / height - 1.0, 0.0)
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
    _cond = jnp.logical_and(jnp.isfinite(sig_arr), jnp.logical_and(jnp.isfinite(ref_arr), jnp.isfinite(obs_arr)))
    non_nan_obs = jnp.where(_cond, obs_arr, 0.0)

    non_nan_sig = jnp.where(_cond, sig_arr, 1.0)

    non_nan_ref = jnp.where(_cond, ref_arr, 0.0)

    chi2s = chi_term(non_nan_ref, non_nan_obs, non_nan_sig)
    no_nan = jnp.where(_cond, 1, 0)
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
        return res_m

    vsolve = vmap(solve, in_axes=(0, 0, 0))
    return vsolve(omags, omagerrs, zobs)  # params_m


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
        return res_r

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
        return res_mr

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
        bool_viz = not (remove_visible) or (
            remove_visible
            and np.isfinite(fors2_attr["mag_sdss_u0"])
            and np.isfinite(fors2_attr["mag_sdss_g0"])
            and np.isfinite(fors2_attr["mag_sdss_r0"])
            and np.isfinite(fors2_attr["mag_sdss_i0"])
            and np.isfinite(fors2_attr["magerr_sdss_u0"])
            and np.isfinite(fors2_attr["magerr_sdss_g0"])
            and np.isfinite(fors2_attr["magerr_sdss_r0"])
            and np.isfinite(fors2_attr["magerr_sdss_i0"])
        )

        bool_fuv = not (remove_galex_fuv) or (remove_galex_fuv and np.isfinite(fors2_attr["mag_galex_FUV"]) and np.isfinite(fors2_attr["magerr_galex_FUV"]))

        bool_nuv = not (remove_galex) or (remove_galex and np.isfinite(fors2_attr["mag_galex_NUV"]) and np.isfinite(fors2_attr["magerr_galex_NUV"]))

        if bool_viz and bool_fuv and bool_nuv:
            filtered_tags.append(tag)
    print(f"Number of galaxies in the sample : {len(filtered_tags)}.")
    return filtered_tags


def fit_vmap(
    xmatch_h5,
    gelato_h5,
    fit_type="mags",
    low_bound=0,
    high_bound=None,
    ssp_file=None,
    weight_mag=0.5,
    remove_visible=False,
    remove_galex=False,
    remove_galex_fuv=True,
    quiet=False,
    source="FORS2",
    selsplit=None,
):
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
    :param source: Origin of the spectroscopic and photometric data. Mostly used to identify the filters used in the photometry, defaults to "FORS2"
    :type source: str, optional
    :param selsplit:  Whether to deal with the crossmatch input as a splitted entry between 'valid' and 'invalid' data. If None, the default behaviour is not to look for splitted data in the file.
            Defaults to None.
    :type selplit: str, optional
    :return: The properties of fitted galaxies in a dataframe, the array of SPS parameters and the boundaries of the selected slice of the set of galaxies.
    :rtype: tuple of (DataFrame, array, int, int)
    """
    ssp_data = load_ssp(ssp_file)
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs_df = bpt_classif(gelatoh5, xmatchh5, source=source, selsplit=selsplit, use_nc=False, return_dict=False)

    # ## Select applicable spectra
    if "fors2" in source.lower():
        filtered_tags = filter_tags_df(merged_attrs_df, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)
    else:
        filtered_tags = list(merged_attrs_df.index)

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

    wls_interp = jnp.arange(1300.0, 323110.0, 50.0) if "gogreen" in source.lower() else jnp.arange(3400.0, 285610.0, 50.0) if "desi" in source.lower() else jnp.arange(1300.0, 24310.0, 10.0)
    wls_rews = jnp.arange(1300.0, 8000.1, 0.1)

    sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr = prepare_data_arr(merged_attrs_df, selected_tags, wls_interp, source=source)
    zs = jnp.array(sel_df["redshift"])

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results = vmap_fit_mags_rews(wls_interp, transm_arr, mags_arr, magerrs_arr, wls_rews, li_wls, rews_arr, rewerrs_arr, zs, ssp_data, weight_mag)
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results = vmap_fit_rews(wls_rews, li_wls, rews_arr, rewerrs_arr, zs, ssp_data)
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        fit_results = vmap_fit_mags(wls_interp, transm_arr, mags_arr, magerrs_arr, zs, ssp_data)

    fit_res_dict = fit_results._asdict()
    fit_results_arr = fit_res_dict.pop("x")
    fit_res_dict.pop("hess_inv", None)
    fit_res_dict.pop("jac", None)
    fit_res_df = pd.DataFrame.from_dict(fit_res_dict)
    fit_res_df.set_index(sel_df.index, inplace=True)
    sel_df = sel_df.join(fit_res_df, how="inner")

    return sel_df, fit_results_arr, low_bound, high_bound


def fit_treemap(
    xmatch_h5,
    gelato_h5,
    fit_type="mags",
    low_bound=0,
    high_bound=None,
    ssp_file=None,
    weight_mag=0.5,
    remove_visible=False,
    remove_galex=False,
    remove_galex_fuv=True,
    quiet=False,
    source="FORS2",
    selsplit=None,
):
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
    :param source: Origin of the spectroscopic and photometric data. Mostly used to identify the filters used in the photometry, defaults to "FORS2"
    :type source: str, optional
    :param selsplit:  Whether to deal with the crossmatch input as a splitted entry between 'valid' and 'invalid' data. If None, the default behaviour is not to look for splitted data in the file.
            Defaults to None.
    :type selsplit: str, optional
    :return: _description_
    :rtype: _type_
    """
    ssp_data = load_ssp(ssp_file)
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs_df = bpt_classif(gelatoh5, xmatchh5, source=source, selsplit=selsplit, use_nc=False, return_dict=False)

    # ## Select applicable spectra
    if "fors2" in source.lower():
        filtered_tags = filter_tags_df(merged_attrs_df, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)
    else:
        filtered_tags = list(merged_attrs_df.index)

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

    wls_interp = jnp.arange(1300.0, 323110.0, 50.0) if "gogreen" in source.lower() else jnp.arange(3400.0, 285610.0, 50.0) if "desi" in source.lower() else jnp.arange(1300.0, 24310.0, 10.0)
    wls_rews = jnp.arange(1300.0, 8000.1, 0.1)

    sel_df, mags_arr, magerrs_arr, rews_arr, rewerrs_arr, li_wls, list_wlmean_f_sel, transm_arr = prepare_data_arr(merged_attrs_df, selected_tags, wls_interp, source=source)
    zs = jnp.array(sel_df["redshift"])

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_magrews = jaxopt.ScipyBoundedMinimize(fun=lik_mag_rew, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            omags, omagerrs, rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_magrews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data, weight_mag)
            return pars, stat

        _arglist = [tuple((ma, mer, rew, rer, z)) for ma, mer, rew, rer, z in zip(mags_arr, magerrs_arr, rews_arr, rewerrs_arr, zs, strict=True)]
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_rews = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_rews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data)
            return pars, stat

        _arglist = [tuple((rew, rer, z)) for rew, rer, z in zip(rews_arr, rewerrs_arr, zs, strict=True)]
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        lbfgsb_mags = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            omags, omagerrs, zobs = arg_tupl
            pars, stat = lbfgsb_mags.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, zobs, ssp_data)
            return pars, stat

        _arglist = [tuple((ma, mer, z)) for ma, mer, z in zip(mags_arr, magerrs_arr, zs, strict=True)]
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)

    pars_list, stats_list = zip(*fit_results_tree, strict=True)
    stats_list = [s._asdict() for s in stats_list]
    stats_df = pd.DataFrame.from_records(stats_list, index=sel_df.index)
    stats_df.drop(columns=["hess_inv"], inplace=True)
    sel_df = sel_df.join(stats_df, how="inner")

    return sel_df, jnp.array(pars_list), low_bound, high_bound


def fit_bootstrap(
    xmatch_h5,
    gelato_h5,
    fit_type="mags",
    bs_tags=None,
    bs_classif=None,
    bs_type="mags",
    n_fits=10,
    ssp_file=None,
    weight_mag=0.5,
    remove_visible=False,
    remove_galex=False,
    remove_galex_fuv=True,
    quiet=False,
    source="FORS2",
    selsplit=None,
):
    """fit_bootstrap _summary_

    :param xmatch_h5: _description_
    :type xmatch_h5: _type_
    :param gelato_h5: _description_
    :type gelato_h5: _type_
    :param fit_type: _description_, defaults to "mags"
    :type fit_type: str, optional
    :param bs_tags: _description_, defaults to None
    :type bs_tags: _type_, optional
    :param bs_classif: _description_, defaults to None
    :type bs_classif: _type_, optional
    :param bs_type: _description_, defaults to "mags"
    :type bs_type: str, optional
    :param n_fits: _description_, defaults to 10
    :type n_fits: int, optional
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
    :param source: _description_, defaults to "FORS2"
    :type source: str, optional
    :param selsplit:  Whether to deal with the crossmatch input as a splitted entry between 'valid' and 'invalid' data. If None, the default behaviour is not to look for splitted data in the file.
            Defaults to None.
    :type selsplit: str, optional
    :return: _description_
    :rtype: _type_
    """
    ssp_data = load_ssp(ssp_file)
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs_df = bpt_classif(gelatoh5, xmatchh5, source=source, selsplit=selsplit, use_nc=False, return_dict=False)

    classif_tags = []
    for tag, row in merged_attrs_df.iterrows():
        if (row["Classification"].lower() == bs_classif.lower()) or bs_classif == "" or bs_classif is None:
            classif_tags.append(tag)
    classif_tags = np.array(classif_tags)

    # ## Select applicable spectra
    if "fors2" in source.lower():
        filtered_tags = filter_tags_df(merged_attrs_df, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)
    else:
        filtered_tags = list(merged_attrs_df.index)

    list_tags = np.intersect1d(np.array(filtered_tags), classif_tags)

    selected_tags = list_tags if (bs_tags is None or len(bs_tags) == 0) else np.intersect1d(list_tags, np.array(bs_tags))
    if not quiet:
        print(f"Number of galaxies to be fitted : {len(selected_tags)}. Number of bootstrap draws : {n_fits}.")

    wls_interp = jnp.arange(1300.0, 323110.0, 50.0) if "gogreen" in source.lower() else jnp.arange(3400.0, 285610.0, 50.0) if "desi" in source.lower() else jnp.arange(1300.0, 24310.0, 10.0)
    wls_rews = jnp.arange(1300.0, 8000.1, 0.1)

    sel_df, mags_tupl, magerrs_tupl, rews_tupl, rewerrs_tupl, li_wls, list_wlmean_f_sel, transm_arr = prepare_bootstrap_arr(
        merged_attrs_df, selected_tags, wls_interp, bs_type=bs_type, n_fits=n_fits, source=source
    )
    zs = jnp.array(sel_df["redshift"])

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_magrews = jaxopt.ScipyBoundedMinimize(fun=lik_mag_rew, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            omags, omagerrs, rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_magrews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data, weight_mag)
            return pars, stat

        _arglist = []
        for mags_arr, magerrs_arr, rews_arr, rewerrs_arr, z in zip(mags_tupl, magerrs_tupl, rews_tupl, rewerrs_tupl, zs, strict=True):
            _arglist.append([tuple((ma, mer, rew, rer, z)) for ma, mer, rew, rer in zip(mags_arr, magerrs_arr, rews_arr, rewerrs_arr, strict=True)])
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        lbfgsb_rews = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            rews_arr, rewerrs_arr, zobs = arg_tupl
            pars, stat = lbfgsb_rews.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_rews, li_wls, rews_arr, rewerrs_arr, zobs, ssp_data)
            return pars, stat

        _arglist = []
        for rews_arr, rewerrs_arr, z in zip(rews_tupl, rewerrs_tupl, zs, strict=True):
            _arglist.append([tuple((rew, rer, z)) for rew, rer in zip(rews_arr, rewerrs_arr, strict=True)])
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        lbfgsb_mags = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=2000)

        # @jit
        def solve(arg_tupl):
            omags, omagerrs, zobs = arg_tupl
            pars, stat = lbfgsb_mags.run(INIT_PARAMS, (PARAMS_MIN, PARAMS_MAX), wls_interp, transm_arr, omags, omagerrs, zobs, ssp_data)
            return pars, stat

        _arglist = []
        for mags_arr, magerrs_arr, z in zip(mags_tupl, magerrs_tupl, zs, strict=True):
            _arglist.append([tuple((ma, mer, z)) for ma, mer in zip(mags_arr, magerrs_arr, strict=True)])
        fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)

    all_means = []
    all_stds = []
    all_pars = {}
    all_succ_counts = []
    all_succ = []
    all_fun_vals = []

    for _tag, _fitresults in zip(sel_df.index, fit_results_tree, strict=True):
        pars_list, stats_list = zip(*_fitresults, strict=True)
        succ = jnp.array([_s.success for _s in stats_list], dtype=bool)
        funvals = jnp.array([_s.fun_val for _s in stats_list], dtype=jnp.float64)
        status = [_s.status for _s in stats_list]
        if jnp.any(jnp.array(succ)):
            pars_arr = jnp.array([_p for _p, _s in zip(pars_list, succ, strict=True) if _s])
            gal_pars_mean = jnp.nanmean(pars_arr, axis=0)
            gal_pars_std = jnp.nanstd(pars_arr, axis=0)
            fun_mean = jnp.nanmean(funvals)
            all_means.append(gal_pars_mean)
            all_stds.append(gal_pars_std)
            all_succ_counts.append(pars_arr.shape[0])
            all_fun_vals.append(fun_mean)
            all_succ.append(True)
            all_pars.update({_tag: {"bs_pars": pars_arr}})
        else:
            all_succ.append(False)
            all_fun_vals.append(jnp.nan)
            all_succ_counts.append(0)
            all_means.append(jnp.full(pars_list[0].shape, jnp.nan))
            all_stds.append(jnp.full(pars_list[0].shape, jnp.nan))
            all_pars.update({_tag: {"bs_pars": None}})
            sel_df.loc[_tag, "status"] = status
    sel_df["success"] = jnp.array(all_succ)
    sel_df["success_count"] = jnp.array(all_succ_counts)
    sel_df["fun_val"] = jnp.array(all_fun_vals)

    return sel_df, jnp.array(all_means), jnp.array(all_stds), all_pars


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


def bootstrapFitsToHDF5(df_outfilename, ref_df, fit_means_arr, fit_stds_arr, fit_pars_dict):
    """bootstrapFitsToHDF5 _summary_

    :param df_outfilename: _description_
    :type df_outfilename: _type_
    :param ref_df: _description_
    :type ref_df: _type_
    :param fit_means_arr: _description_
    :type fit_means_arr: _type_
    :param fit_stds_arr: _description_
    :type fit_stds_arr: _type_
    :param fit_pars_list: _description_
    :type fit_pars_list: _type_
    :return: _description_
    :rtype: _type_
    """
    res_df = pd.DataFrame(index=ref_df.index, columns=_DUMMY_P_ADQ.PARAM_NAMES_FLAT + [f"{_p}_ERR" for _p in _DUMMY_P_ADQ.PARAM_NAMES_FLAT], data=jnp.column_stack((fit_means_arr, fit_stds_arr)))
    out_df = ref_df.join(res_df, how="inner")
    outpath = os.path.abspath(df_outfilename)
    with h5py.File(outpath, "w") as h5f:
        grp = h5f.create_group("boot_dsps")
        for _tag, row in out_df.iterrows():
            sgrp = grp.create_group(_tag)
            sgrp.create_dataset("bs_pars", data=np.array(fit_pars_dict.pop(_tag).pop("bs_pars"), dtype=np.float64))
            for key, val in row.to_dict().items():
                sgrp.attrs[key] = val
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


def readBootstrapFitsFromHDF5(dspsFitsH5, group="boot_dsps"):
    """readBootstrapFitsFromHDF5 _summary_

    :param dspsFitsH5: _description_
    :type dspsFitsH5: _type_
    :param group: _description_, defaults to "boot_dsps"
    :type group: str, optional
    :return: _description_
    :rtype: _type_
    """
    dico_to_df = {}
    dico_pars = {}
    with h5py.File(os.path.abspath(dspsFitsH5), "r") as h5f:
        grp = h5f.get(group)
        for _tag in grp:
            sgrp = grp.get(_tag)
            dico_to_df.update({_tag: {_k: sgrp.attrs.get(_k) for _k in sgrp.attrs}})
            dico_pars.update({_tag: jnp.array(sgrp.get("bs_pars"), dtype=jnp.float64)})
    fitres_df = pd.DataFrame.from_dict(dico_to_df, orient="index")
    return fitres_df, dico_pars


def func_strip_name(x):
    """
    Strip string of filters name for shorter name plotting
    :param x: name
    :type x: string
    """
    return x.split("_")[-1]


def plot_figs_to_PDF(pdf_file, fig_list):
    """
    Gather figures in a PDF file.

    Parameters
    ----------
    pdf_file : str or path
        Path to the PDF file where to store figures.
    fig_list : list
        List of matplotlib figures to print in PDF file.
    """
    with PdfPages(pdf_file) as pdf:
        for fig in fig_list:
            pdf.savefig(fig)
            plt.close()
    return None


def make_vmapfit_plots(sel_df, gelato_h5, wls_arr, ssp_data, source="FORS2", outpdf=None):
    """make_vmapfit_plots _summary_

    :param sel_df: _description_
    :type sel_df: _type_
    :param gelato_h5: _description_
    :type gelato_h5: _type_
    :param wls_arr: _description_
    :type wls_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param source: _description_, defaults to "FORS2"
    :type source: str, optional
    :param outpdf: Name or path to the PDF output file, defaults to None
    :type oudf: str or path-like, optional
    """
    from process_fors2.analysis import convert_flux_toobsframe, convert_flux_torestframe, convertFlambdaToFnu, lsunPerHz_to_fnu

    gelatoh5 = os.path.abspath(gelato_h5)

    rews_list = [col for col in list(sel_df.columns) if ("rew" in col.lower())]
    mags_list = [col for col in list(sel_df.columns) if ("mag" in col.lower() and "image" not in col.lower())]

    # For all REWs
    li_names = np.unique([li.split("_REW")[0] for li in rews_list])
    li_wls = jnp.array([float(ln.split("_")[-1]) for ln in li_names])

    # For selected lines
    lines = jnp.array([6564.61, 4862.68, 5008.24])
    lines_names = [
        "Balmer_HI_6564.61",
        "Balmer_HI_4862.68",
        "AGN_[OIII]_5008.24",
    ]
    line_wids = lines * 400 / C_KMS / 2
    cont_wids = lines * 15000 / C_KMS / 2

    if "fors2" in source.lower():
        from process_fors2.fetchData import load_filters_from_f2df

        _, transm_arr, list_wlmean_f_sel = load_filters_from_f2df(sel_df, wls_arr)
    else:  # elif "gogreen" in source.lower(): # the DESI case should be covered by any of these two functions, let's pick GOGREEN.
        from process_fors2.fetchData import load_filters_from_ggdf

        _, transm_arr, list_wlmean_f_sel = load_filters_from_ggdf(sel_df, wls_arr)
    list_name_f_sel = ["_".join(m.split("_")[1:]) for m in mags_list if "err" not in m.lower()]

    list_of_figs = []

    for _tag, row in tqdm(sel_df.iterrows(), total=sel_df.shape[0]):
        f, _axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
        a_sfh, ax_spec, ax_rew, ax_ha, ax_hb, ax_oiii = _axes[0, 0], _axes[1, 0], _axes[2, 0], _axes[0, 1], _axes[1, 1], _axes[2, 1]
        z_obs = row["redshift"]
        if "fors2" in source.lower():
            tag = _tag
        elif "gogreen" in source.lower():
            tag = f"{row['cluster']}_{row['specid']}"
        elif "desi" in source.lower():
            tag = f"{row['survey']}_{row['program']}"
            if row["BGS"]:
                tag += "_BGS"
            if row["ELG"]:
                tag += "_ELG"
            if row["LRG"]:
                tag += "_LRG"
            if row["QSO"]:
                tag += "_QSO"
            tag += f"_{row['specid']}"
        title_spec = f"{tag} z = {z_obs:.3f}"
        # spec_obs = get_fnu(gelatoh5, tag, zob=z_obs)
        # Xs = spec_obs["wl"]
        # Ys = spec_obs["fnu"]
        # EYs = spec_obs["fnuerr"]

        with h5py.File(gelatoh5, "r") as gel5:
            group = gel5.get(tag)
            wlo = jnp.array(group.get("wl_ang"))
            flamo = jnp.array(group.get("flam"))
            flamoerr = jnp.array(group.get("flam_err"))
            glamo = jnp.array(group.get("gelato_mod"))

        wlr, flamr = convert_flux_torestframe(wlo, flamo, z_obs)
        _, flamrerr = convert_flux_torestframe(wlo, flamoerr, z_obs)
        _, glamr = convert_flux_torestframe(wlo, glamo, z_obs)

        fnur = convertFlambdaToFnu(wlr, flamr)
        fnurerr = convertFlambdaToFnu(wlr, flamrerr)
        gnur = convertFlambdaToFnu(wlr, glamr)

        _, fnuo = convert_flux_toobsframe(wlr, fnur, z_obs)
        _, fnuoerr = convert_flux_toobsframe(wlr, fnurerr, z_obs)
        _, gnuo = convert_flux_toobsframe(wlr, gnur, z_obs)

        rchi2 = row["rChi2"]

        # get the Gelato model
        # gel_obs = get_gelmod(gelatoh5, tag, zob=z_obs)
        # gemod = interp1d(wls, gel_obs["wl"], gel_obs["mod"], method="akima", extrap=False)

        params_arr = jnp.array(row[_DUMMY_P_ADQ.PARAM_NAMES_FLAT].values, dtype=jnp.float64)

        mags_arr = jnp.array(row[[c for c in mags_list if "err" not in c.lower()]].values, dtype=jnp.float64)
        magerrs_arr = jnp.array(row[[c for c in mags_list if "err" in c.lower()]].values, dtype=jnp.float64)

        rews_arr = jnp.array(row[[c for c in rews_list if "err" not in c.lower()]].values, dtype=jnp.float64)
        rewerrs_arr = jnp.array(row[[c for c in rews_list if "err" in c.lower()]].values, dtype=jnp.float64)

        # Plot SFH
        sfh_gal = mean_sfr(params_arr)
        t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
        t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

        a_sfh.plot(T_ARR, sfh_gal, "-k", lw=2)
        a_sfh.axvline(t_obs, color="red")
        a_sfh.text(t_obs, sfh_gal.max(), f"z={z_obs:.3f}", color="red")

        sfr_max = sfh_gal.max() * 1.1
        sfr_min = 0.0
        a_sfh.set_ylim(sfr_min, sfr_max)

        a_sfh.set_title("Fitted SFH")
        a_sfh.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
        a_sfh.set_ylabel(r"${\rm SFR\ [M_{\odot}/yr]}$")
        a_sfh.grid()

        # Plot Photometry
        x, y_nodust, y_dust = ssp_spectrum_fromparam(params_arr, z_obs, ssp_data)
        fnu_dsps = lsunPerHz_to_fnu(y_dust, z_obs)
        fnu_dsps_nodust = lsunPerHz_to_fnu(y_nodust, z_obs)

        mags_predictions = vmap_calc_obs_mag(x, y_dust, wls_arr, transm_arr, z_obs)

        ax_phot = ax_spec.twinx()
        ax_spec.set_yscale("log")
        ax_spec.set_xscale("log")

        # plot Fors2 data
        (l2,) = ax_spec.plot(wlo, fnuo, "b-", lw=0.2, label="Obs.\nspectrum")

        # plot SED model
        (l0,) = ax_spec.plot(*convert_flux_toobsframe(x, fnu_dsps, z_obs), "-", color="green", lw=1, label="DSPS output\nwith dust")
        (l1,) = ax_spec.plot(*convert_flux_toobsframe(x, fnu_dsps_nodust, z_obs), "-", color="red", lw=1, label="DSPS output\nwithout dust")

        # plot photometric data
        label = "Catalog\nphotometry"
        valid_phot = jnp.logical_and(jnp.isfinite(mags_arr), jnp.isfinite(magerrs_arr))
        l3 = ax_phot.errorbar(list_wlmean_f_sel[valid_phot], mags_arr[valid_phot], yerr=magerrs_arr[valid_phot], fmt=".", color="black", ecolor="black", markersize=20, label=label)
        l4 = ax_phot.scatter(list_wlmean_f_sel[valid_phot], mags_predictions[valid_phot], s=100, marker="s", c="orange", label="Modeled\nphotometry")

        ax_spec.set_title(rf"DSPS fit (obs. frame) - $\chi^2=${row['fun_val']:.2f}")
        # ax.legend()  # (loc="upper left", bbox_to_anchor=(1.1, 1.0))

        ymax = max(fnu_dsps_nodust.max() / (1 + z_obs), fnuo.max())
        ymin = fnuo.min()
        ylim_max = ymax * 2.0
        ylim_min = ymin / 1.5

        filter_tags = [func_strip_name(n) for n, b in zip(list_name_f_sel, valid_phot, strict=True) if b]
        ax_spec.set_xticks(list_wlmean_f_sel[valid_phot], labels=filter_tags, minor=True)
        ax_spec.tick_params(
            axis="x",
            which="minor",
            bottom=False,
            top=True,
            labelbottom=False,
            labeltop=True,
            grid_color="tab:blue",
            grid_linestyle=":",
            colors="tab:blue",
            length=4,
        )

        # for idf, ftag in enumerate(filter_tags):
        #    ax_spec.text(list_wlmean_f_sel[valid_phot][idf], 2.0 * ymax - (idf % 2) * 0.5 * ymax, ftag, fontsize=10, fontweight="bold", horizontalalignment="center", verticalalignment="center")
        #    ax_spec.axvline(list_wlmean_f_sel[valid_phot][idf], linestyle=":")

        ax_spec.set_xlabel("$\\lambda\\ [\\AA]$")
        # ax_spec.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")
        ax_spec.set_ylabel("$F_\\nu\\ [\\mathrm{erg . s^{-1} . cm^{-2} . Hz^{-1}}]$")
        ax_phot.set_ylabel("$m_{AB}$")
        # ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

        ax_spec.set_xlim(jnp.min(list_wlmean_f_sel[valid_phot]) * 0.9, jnp.max(list_wlmean_f_sel[valid_phot]) * 1.1)
        ax_spec.set_ylim(ylim_min, ylim_max)

        m_min = min(mags_arr[valid_phot].min(), mags_predictions[valid_phot].min())
        m_max = max(mags_arr[valid_phot].max(), mags_predictions[valid_phot].max())
        ax_phot.set_ylim(m_max + 1, m_min - 1)

        ax_spec.grid()
        plt.legend(handles=[l0, l1, l2, l3, l4], loc="upper left", bbox_to_anchor=(1.1, 1.0))

        # Plot Equivalent widths + GELATO
        # ax_rew.set_yscale("log")
        # ax_rew.set_xscale("log")

        (lf,) = ax_rew.plot(wlr, fnur, "b-", lw=0.2, label="Obs. spectrum")
        ax_rew.fill_between(wlr, fnur - fnurerr, fnur + fnurerr, color="b", alpha=0.2)

        (ld,) = ax_rew.plot(x, fnu_dsps, "-", color="green", lw=1, label="DSPS output\nwith dust")
        (lg,) = ax_rew.plot(wlr, gnur, color="maroon", lw=1, alpha=0.7, label="GELATO model")

        srwls = jnp.arange(1300.0, 8000.1, 0.1)
        surspec = interp1d(srwls, x, fnu_dsps, method="akima", extrap=False)
        mod_rews = vmap_calc_eqw(srwls, surspec, li_wls)
        ax_rews = ax_rew.twinx()

        valid_rew = jnp.logical_and(jnp.isfinite(rews_arr), jnp.isfinite(rewerrs_arr))

        label = "Restframe\nEq. Widths"
        lrg = ax_rews.errorbar(li_wls[valid_rew], rews_arr[valid_rew], yerr=rewerrs_arr[valid_rew], fmt=".", color="black", ecolor="black", markersize=20, label=label)
        lrd = ax_rews.scatter(li_wls[valid_rew], mod_rews[valid_rew], s=100, marker="s", c="orange", label="Modeled REWs")

        ymax = jnp.nanmax(fnur)
        ymin = jnp.nanmin(fnur)
        ylim_max = ymax * 1.2
        ylim_min = ymin - (0.2 * ymax)

        min_rew = jnp.nanmin(rews_arr[valid_rew]) - 3
        max_rew = jnp.nanmax(rews_arr[valid_rew]) + 3

        lnams = ["_".join(etag.split("_")[1:3]) for etag in li_names[valid_rew]]
        ax_rew.set_xticks(li_wls[valid_rew], labels=lnams, minor=True)
        ax_rew.tick_params(axis="x", which="minor", grid_color="tab:blue", grid_linestyle=":", colors="tab:blue", length=16, labelrotation=90.0)

        # for ide, etag in enumerate(li_names[valid_rew]):
        #    _lnam = "_".join(etag.split("_")[:2])  # f"${li_wls[ide]:.2f}\ \AA$"
        #    ax_rews.text(
        #        li_wls[valid_rew][ide],
        #        min_rew * 0.5,  # (1 - ide % 2) + max_rew * (ide % 2),
        #        _lnam,
        #        fontsize=8,
        #        fontweight="bold",
        #        horizontalalignment="left",
        #        verticalalignment="baseline",
        #        rotation="vertical",
        #    )
        #    ax_rews.axvline(li_wls[valid_rew][ide], linestyle=":")

        ax_rew.set_xlabel("$\\lambda\\ [\\AA]$")
        ax_rew.set_ylabel("$F_\\nu\\ [\\mathrm{erg . s^{-1} . cm^{-2} . Hz^{-1}}]$")
        ax_rews.set_ylabel(r"${\rm Restframe Eq. Width\ [\AA]}$")
        # ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

        ax_rew.set_xlim(min(wlr) - 200.0, max(wlr) + 200.0)
        ax_rew.set_ylim(ylim_min, ylim_max)
        ax_rews.set_ylim(min_rew, max_rew)
        # ax_rews.set_ylim(29, 18)

        ax_rew.grid()
        ax_rew.set_title(rf"GELATO fit (restframe) - $\chi^2=${rchi2:.2f}")
        f.suptitle(title_spec)
        plt.legend(handles=[lg, lrg, lrd], loc="upper left", bbox_to_anchor=(1.1, 1.0))

        # Plot detailed lines
        for _il, (_ax, _liwl, _licont, _liwid, _liname) in enumerate(zip([ax_ha, ax_hb, ax_oiii], lines, cont_wids, line_wids, lines_names, strict=True)):
            sel = jnp.logical_and(wlr >= _liwl - _licont - 1, wlr <= _liwl + _licont + 1)

            (lff,) = _ax.plot(wlr[sel], fnur[sel], "b-", lw=0.5, label="Obs. spectrum")
            # ax_rew.fill_between(wlr[sel], fnur[sel] - fnurerr[sel], fnur[sel] + fnurerr[sel], color="b", alpha=0.2)

            selx = jnp.logical_and(x >= _liwl - _licont - 1, x <= _liwl + _licont + 1)
            (ldd,) = _ax.plot(x[selx], fnu_dsps[selx], "-", color="green", lw=2, label="DSPS output\nwith dust")
            (lgg,) = _ax.plot(wlr[sel], gnur[sel], color="maroon", lw=2, label="GELATO model")

            _mod_rew = calc_eqw(srwls, surspec, _liwl)
            idx_rew = np.argwhere(li_names == _liname)[0][0]

            _gel_rew = rews_arr[idx_rew]

            # _ax.axvline(_liwl - _licont, ls=":", color="orange", label="Continuum bounds")
            # _ax.axvline(_liwl + _licont, ls=":", color="orange")
            # _ax.axvline(_liwl - _liwid, ls=":", color="r", label="Line bounds")
            # _ax.axvline(_liwl + _liwid, ls=":", color="r")
            # _ax.axvline(_liwl, ls="-", lw=1, color="black", label=_liname)
            _ax.set_xticks(np.array([_liwl - _licont, _liwl - _liwid, _liwl, _liwl + _liwid, _liwl + _licont]))
            _ax.tick_params(axis="x", which="major", grid_linestyle=":", label_rotation=90.0)

            _ax.fill_between(
                x[selx],
                fnu_dsps[selx],
                where=np.logical_and(x[selx] > _liwl - 0.5 * _mod_rew, x[selx] < _liwl + 0.5 * _mod_rew),
                color="cyan",
                alpha=0.2,
                label=r"REW-DSPS $=$" + f"{_mod_rew:.2f}" + r"$\mathrm{\AA}$",
            )

            if np.isfinite(_gel_rew):
                _ax.fill_between(
                    wlr[sel],
                    gnur[sel],
                    where=np.logical_and(wlr[sel] > _liwl - 0.5 * _gel_rew, wlr[sel] < _liwl + 0.5 * _gel_rew),
                    ec="maroon",
                    fc=(0.0, 1.0, 0.0, 0.0),
                    # alpha=0.4,
                    hatch="//",
                    label=r"REW-GELATO $=$" + f"{_gel_rew:.2f}" + r"$\mathrm{\AA}$",
                )

            _ax.set_xlabel("$\\lambda\\ [\\AA]$")
            _ax.set_ylabel("$F_\\nu\\ [\\mathrm{erg . s^{-1} . cm^{-2} . Hz^{-1}}]$")
            _ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))
            _ax.set_title(_liname)
            _ax.grid()

        list_of_figs.append(copy.deepcopy(f))
    pdfoutputfilename = f"{source}_dsps_and_gelato_plots_valid_fits.pdf" if outpdf is None else os.path.abspath(".".join([os.path.splitext(outpdf)[0], "pdf"]))
    _ = plot_figs_to_PDF(pdfoutputfilename, list_of_figs)


def make_bootstrap_plots(sel_df, params_dict, gelato_h5, wls_arr, ssp_data, source="FORS2", outpdf=None):
    """make_bootstrap_plots _summary_

    :param sel_df: _description_
    :type sel_df: _type_
    :param params_dict: _description_
    :type params_dict: _type_
    :param gelato_h5: _description_
    :type gelato_h5: _type_
    :param wls_arr: _description_
    :type wls_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param source: _description_, defaults to "FORS2"
    :type source: str, optional
    :param outpdf: _description_, defaults to None
    :type outpdf: _type_, optional
    """
    from process_fors2.analysis import convert_flux_toobsframe, convert_flux_torestframe, convertFlambdaToFnu, lsunPerHz_to_fnu

    gelatoh5 = os.path.abspath(gelato_h5)

    rews_list = [col for col in list(sel_df.columns) if ("rew" in col.lower())]
    mags_list = [col for col in list(sel_df.columns) if ("mag" in col.lower() and "image" not in col.lower())]
    li_names = np.unique([li.split("_REW")[0] for li in rews_list])
    li_wls = jnp.array([float(ln.split("_")[-1]) for ln in li_names])

    if "fors2" in source.lower():
        from process_fors2.fetchData import load_filters_from_f2df

        _, transm_arr, list_wlmean_f_sel = load_filters_from_f2df(sel_df, wls_arr)
    else:  # elif "gogreen" in source.lower(): # the DESI case should be covered by any of these two functions, let's pick GOGREEN.
        from process_fors2.fetchData import load_filters_from_ggdf

        _, transm_arr, list_wlmean_f_sel = load_filters_from_ggdf(sel_df, wls_arr)
    list_name_f_sel = ["_".join(m.split("_")[1:]) for m in mags_list if "err" not in m.lower()]

    list_of_figs = []
    v_mags = vmap(vmap_calc_obs_mag, in_axes=(None, 0, None, None, None))
    vrews = vmap(vmap_calc_eqw, in_axes=(None, 0, None))
    v_spec = vmap(ssp_spectrum_fromparam, in_axes=(0, None, None))

    for _tag, row in sel_df.iterrows():
        f, (a_sfh, ax_spec, ax_rew) = plt.subplots(3, 1, figsize=(7, 10), constrained_layout=True)
        z_obs = row["redshift"]
        if "fors2" in source.lower():
            tag = _tag
        elif "gogreen" in source.lower():
            tag = f"{row['cluster']}_{row['specid']}"
        elif "desi" in source.lower():
            tag = f"{row['survey']}_{row['program']}"
            if row["BGS"]:
                tag += "_BGS"
            if row["ELG"]:
                tag += "_ELG"
            if row["LRG"]:
                tag += "_LRG"
            if row["QSO"]:
                tag += "_QSO"
            tag += f"_{row['specid']}"
        title_spec = f"{tag} z = {z_obs:.3f}"
        # spec_obs = get_fnu(gelatoh5, tag, zob=z_obs)
        # Xs = spec_obs["wl"]
        # Ys = spec_obs["fnu"]
        # EYs = spec_obs["fnuerr"]

        with h5py.File(gelatoh5, "r") as gel5:
            group = gel5.get(tag)
            wlo = jnp.array(group.get("wl_ang"))
            flamo = jnp.array(group.get("flam"))
            flamoerr = jnp.array(group.get("flam_err"))
            glamo = jnp.array(group.get("gelato_mod"))

        wlr, flamr = convert_flux_torestframe(wlo, flamo, z_obs)
        _, flamrerr = convert_flux_torestframe(wlo, flamoerr, z_obs)
        _, glamr = convert_flux_torestframe(wlo, glamo, z_obs)

        fnur = convertFlambdaToFnu(wlr, flamr)
        fnurerr = convertFlambdaToFnu(wlr, flamrerr)
        gnur = convertFlambdaToFnu(wlr, glamr)

        _, fnuo = convert_flux_toobsframe(wlr, fnur, z_obs)
        _, fnuoerr = convert_flux_toobsframe(wlr, fnurerr, z_obs)
        _, gnuo = convert_flux_toobsframe(wlr, gnur, z_obs)

        rchi2 = row["rChi2"]

        mags_arr = jnp.array(row[[c for c in mags_list if "err" not in c.lower()]].values, dtype=jnp.float64)
        magerrs_arr = jnp.array(row[[c for c in mags_list if "err" in c.lower()]].values, dtype=jnp.float64)

        rews_arr = jnp.array(row[[c for c in rews_list if "err" not in c.lower()]].values, dtype=jnp.float64)
        rewerrs_arr = jnp.array(row[[c for c in rews_list if "err" in c.lower()]].values, dtype=jnp.float64)

        # param_means_arr = jnp.tile(
        #    jnp.array(row[_DUMMY_P_ADQ.PARAM_NAMES_FLAT].values, dtype=jnp.float64),
        #    (n_bootstraps, 1)
        # )

        # param_errs_arr = jnp.tile(
        #    jnp.array(row[[f"{p}_ERR" for p in _DUMMY_P_ADQ.PARAM_NAMES_FLAT]].values, dtype=jnp.float64),
        #    (n_bootstraps, 1)
        # )

        # jkey = jax.random.key(141)
        # jkey, jsubkey = jax.random.split(jkey)
        # rnd_draws = jax.random.normal(jsubkey, shape=param_means_arr.shape)

        params_rnd = jnp.array(params_dict[_tag], dtype=jnp.float64)  # rnd_draws*param_errs_arr + param_means_arr

        # Plot SFH
        sfh_gal = vmap_mean_sfr(params_rnd)
        sfh_mean = jnp.mean(sfh_gal, axis=0)
        sfh_std = jnp.std(sfh_gal, axis=0)
        t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
        t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

        a_sfh.plot(T_ARR, sfh_mean, "-k", lw=2)
        a_sfh.fill_between(T_ARR, sfh_mean - sfh_std, sfh_mean + sfh_std, alpha=0.3, color="gray")
        a_sfh.axvline(t_obs, color="red")

        sfr_max = sfh_mean.max() * 1.1
        sfr_min = 0.0
        a_sfh.set_ylim(sfr_min, sfr_max)

        a_sfh.set_title("Fitted SFH")
        a_sfh.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
        a_sfh.set_ylabel(r"${\rm SFR\ [M_{\odot}/yr]}$")
        a_sfh.grid()

        # Plot Photometry
        x, y_nodust, y_dust = v_spec(params_rnd, z_obs, ssp_data)
        ynu_nodust = jnp.array([lsunPerHz_to_fnu(_y, z_obs) for _y in y_nodust])
        ynu_dust = jnp.array([lsunPerHz_to_fnu(_y, z_obs) for _y in y_dust])
        x = x[0, :]
        fnu_dsps_nodust, fnu_dsps = jnp.mean(ynu_nodust, axis=0), jnp.mean(ynu_dust, axis=0)
        fnuerr_dsps, fnuerr_dsps_nodust = jnp.std(ynu_nodust, axis=0), jnp.std(ynu_dust, axis=0)

        mags_predictions = v_mags(x, y_dust, wls_arr, transm_arr, z_obs)
        mags_means, mags_std = jnp.mean(mags_predictions, axis=0), jnp.std(mags_predictions, axis=0)

        ax_phot = ax_spec.twinx()
        ax_spec.set_yscale("log")
        ax_spec.set_xscale("log")

        # plot Fors2 data
        (l2,) = ax_spec.plot(wlo, fnuo, "b-", lw=0.2, label="Obs.\nspectrum")

        # plot SED model
        xplot, fnuobs = convert_flux_toobsframe(x, fnu_dsps, z_obs)
        _, fnumax = convert_flux_toobsframe(x, fnu_dsps + fnuerr_dsps, z_obs)
        _, fnumin = convert_flux_toobsframe(x, fnu_dsps - fnuerr_dsps, z_obs)
        (l0,) = ax_spec.plot(xplot, fnuobs, "-", color="green", lw=1, label="DSPS output\nwith dust")
        ax_spec.fill_between(xplot, fnumin, fnumax, color="green", alpha=0.3)

        _, fnuobs_nodust = convert_flux_toobsframe(x, fnu_dsps_nodust, z_obs)
        _, fnumax_nodust = convert_flux_toobsframe(x, fnu_dsps_nodust + fnuerr_dsps_nodust, z_obs)
        _, fnumin_nodust = convert_flux_toobsframe(x, fnu_dsps_nodust - fnuerr_dsps_nodust, z_obs)
        (l1,) = ax_spec.plot(xplot, fnuobs_nodust, "-", color="red", lw=1, label="DSPS output\nwithout dust")
        ax_spec.fill_between(xplot, fnumin_nodust, fnumax_nodust, color="red", alpha=0.3)

        # plot photometric data
        label = "Catalog\nphotometry"
        valid_phot = jnp.logical_and(jnp.isfinite(mags_arr), jnp.isfinite(magerrs_arr))
        l3 = ax_phot.errorbar(list_wlmean_f_sel[valid_phot], mags_arr[valid_phot], yerr=magerrs_arr[valid_phot], fmt=".", color="black", ecolor="black", markersize=20, label=label)
        l4 = ax_phot.errorbar(list_wlmean_f_sel[valid_phot], mags_means[valid_phot], mags_std[valid_phot], fmt="s", markersize=7, color="orange", ecolor="orange", label="Modeled\nphotometry")

        ax_spec.set_title(rf"DSPS fit (obs. frame) - $\chi^2=${row['fun_val']:.2f}")
        # ax.legend()  # (loc="upper left", bbox_to_anchor=(1.1, 1.0))

        ymax = max(fnu_dsps_nodust.max() / (1 + z_obs), fnuo.max())
        ymin = fnuo.min()
        ylim_max = ymax * 2.0
        ylim_min = ymin / 1.5

        filter_tags = [func_strip_name(n) for n, b in zip(list_name_f_sel, valid_phot, strict=True) if b]
        for idf, ftag in enumerate(filter_tags):
            ax_spec.text(list_wlmean_f_sel[valid_phot][idf], 2.0 * ymax - (idf % 2) * 0.5 * ymax, ftag, fontsize=10, fontweight="bold", horizontalalignment="center", verticalalignment="center")
            ax_spec.axvline(list_wlmean_f_sel[valid_phot][idf], linestyle=":")

        ax_spec.set_xlabel("$\\lambda\\ [\\AA]$")
        # ax_spec.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")
        ax_spec.set_ylabel("$F_\\nu\\ [\\mathrm{erg . s^{-1} . cm^{-2} . Hz^{-1}}]$")
        ax_phot.set_ylabel("$m_{AB}$")
        # ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

        ax_spec.set_xlim(jnp.min(list_wlmean_f_sel[valid_phot]) * 0.9, jnp.max(list_wlmean_f_sel[valid_phot]) * 1.1)
        ax_spec.set_ylim(ylim_min, ylim_max)

        m_min = min(mags_arr[valid_phot].min(), mags_means[valid_phot].min())
        m_max = max(mags_arr[valid_phot].max(), mags_means[valid_phot].max())
        ax_phot.set_ylim(m_max + 1, m_min - 1)

        ax_spec.grid()
        plt.legend(handles=[l0, l1, l2, l3, l4], loc="upper left", bbox_to_anchor=(1.1, 1.0))

        # Plot Equivalent widths + GELATO
        ax_rew.set_yscale("log")
        # ax_rew.set_xscale("log")

        (lf,) = ax_rew.plot(wlr, fnur, "b-", lw=0.2, label="Obs. spectrum")
        ax_rew.fill_between(wlr, fnur - fnurerr, fnur + fnurerr, color="b", alpha=0.2)

        (ld,) = ax_rew.plot(x, fnu_dsps, "-", color="green", lw=1, label="DSPS output\nwith dust")
        (lg,) = ax_rew.plot(wlr, gnur, color="maroon", lw=1, alpha=0.7, label="GELATO model")

        srwls = jnp.arange(1300.0, 8000.1, 0.1)

        v_interp = vmap(lambda _y: interp1d(srwls, x, _y, method="akima", extrap=False))  # noqa: B023
        surspec = v_interp(y_dust)

        mod_rews = vrews(srwls, surspec, li_wls)
        rews_means, rews_std = jnp.mean(mod_rews, axis=0), jnp.std(mod_rews, axis=0)
        ax_rews = ax_rew.twinx()

        valid_rew = jnp.logical_and(jnp.isfinite(rews_arr), jnp.isfinite(rewerrs_arr))

        label = "Restframe\nEq. Widths"
        lrg = ax_rews.errorbar(li_wls[valid_rew], rews_arr[valid_rew], yerr=rewerrs_arr[valid_rew], fmt=".", color="black", ecolor="black", markersize=20, label=label)
        lrd = ax_rews.errorbar(li_wls[valid_rew], rews_means[valid_rew], yerr=rews_std[valid_rew], fmt="s", markersize=7, color="orange", ecolor="orange", label="Modeled REWs")

        ymax = jnp.nanmax(fnur)
        ymin = jnp.nanmin(fnur)
        ylim_max = ymax * 1.2
        ylim_min = ymin / 1.2

        min_rew = jnp.nanmin(rews_means[valid_rew]) - 3
        max_rew = jnp.nanmax(rews_means[valid_rew]) + 3

        for ide, etag in enumerate(li_names[valid_rew]):
            _lnam = "_".join(etag.split("_")[:2])  # f"${li_wls[ide]:.2f}\ \AA$"
            ax_rews.text(
                li_wls[valid_rew][ide],
                min_rew * (1 - ide % 2) + max_rew * (ide % 2),
                _lnam,
                fontsize=8,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
                rotation="vertical",
            )
            ax_rews.axvline(li_wls[valid_rew][ide], linestyle=":")

        ax_rew.set_xlabel("$\\lambda\\ [\\AA]$")
        ax_rew.set_ylabel("$F_\\nu\\ [\\mathrm{erg . s^{-1} . cm^{-2} . Hz^{-1}}]$")
        ax_rews.set_ylabel(r"${\rm Restframe Eq. Width\ [\AA]}$")
        # ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

        ax_rew.set_xlim(min(wlr) - 200.0, max(wlr) + 200.0)
        ax_rew.set_ylim(ylim_min, ylim_max)
        ax_rews.set_ylim(min_rew, max_rew)
        # ax_rews.set_ylim(29, 18)

        ax_rews.grid()
        ax_rew.set_title(rf"GELATO fit (restframe) - $\chi^2=${rchi2:.2f}")
        f.suptitle(title_spec)
        plt.legend(handles=[lg, lrg, lrd], loc="upper left", bbox_to_anchor=(1.1, 1.0))

        list_of_figs.append(copy.deepcopy(f))
    pdfoutputfilename = f"BOOTSTRAP-{source}_dsps_and_gelato_plots.pdf" if outpdf is None else os.path.abspath(".".join([os.path.splitext(outpdf)[0], "pdf"]))
    _ = plot_figs_to_PDF(pdfoutputfilename, list_of_figs)


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
    _use_bounds = inputs["bounded_fit"]
    _weight_mag = inputs["weight_mag"]  # Only for combined fit : mags + rews
    _ssp_file = None if (inputs["ssp_file"].lower() == "default" or inputs["ssp_file"] == "" or inputs["ssp_file"] is None) else os.path.abspath(inputs["ssp_file"])

    _low = inputs["first_spec"]
    _high = None if inputs["last_spec"] < 0 else inputs["last_spec"]
    _src = inputs["data_origin"]
    _split = inputs["data_selection"]

    if _use_bounds:
        sel_df, fit_results_arr, low_bound, high_bound = fit_treemap(
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
            source=_src,
            selsplit=_split,
        )
        outdir = os.path.abspath(f"./DSPS_hdf5_TREEMAPfit_{_src}_{_fit_type}")
    else:
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
            source=_src,
            selsplit=_split,
        )
        outdir = os.path.abspath(f"./DSPS_hdf5_VMAPfit_{_src}_{_fit_type}")

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    filename_params = os.path.join(outdir, f"fitparams_{_fit_type}_{low_bound+1}_to_{high_bound}.h5")
    status = vmapFitsToHDF5(filename_params, sel_df, fit_results_arr)
    print(status)


def run_bs_fit(args):
    """run_bs_fit _summary_

    :param args: _description_
    :type args: _type_
    """
    from process_fors2.fetchData import json_to_inputs
    from process_fors2.fetchData.queries import FORS2DATALOC

    conf_json = args[3] if len(args) > 3 else os.path.join(FORS2DATALOC, "defaults.json")  # attention à la localisation du fichier !

    xmatchh5 = args[1]  # le premier argument de args est toujours `__main__.py`
    gelatoh5 = args[2]
    inputs = json_to_inputs(conf_json)["fitDSPS"]
    _fit_type = inputs["fit_type"]
    # _use_bounds = inputs["bounded_fit"]
    _weight_mag = inputs["weight_mag"]  # Only for combined fit : mags + rews
    _ssp_file = None if (inputs["ssp_file"].lower() == "default" or inputs["ssp_file"] == "" or inputs["ssp_file"] is None) else os.path.abspath(inputs["ssp_file"])

    # _low = inputs["first_spec"]
    # _high = None if inputs["last_spec"] < 0 else inputs["last_spec"]
    _src = inputs["data_origin"]
    _split = inputs["data_selection"]
    print(f"Fit on {_split}_data.")

    if inputs["bootstrap_id"] is None or len(inputs["bootstrap_id"]) == 0:  # noqa: SIM108
        inp_tags = None
    else:
        inp_tags = np.array(inputs["bootstrap_id"]) if isinstance(inputs["bootstrap_id"], list) else np.array([inputs["bootstrap_id"]])

    sel_df, fit_means, fit_stds, pars_dict = fit_bootstrap(
        xmatchh5,
        gelatoh5,
        fit_type=_fit_type,
        bs_tags=inp_tags,
        bs_classif=inputs["bootstrap_classif"],
        bs_type=inputs["bootstrap_type"],
        n_fits=inputs["number_bootstrap"],
        ssp_file=_ssp_file,
        weight_mag=_weight_mag,
        remove_visible=inputs["remove_visible"],
        remove_galex=inputs["remove_galex"],
        remove_galex_fuv=inputs["remove_fuv"],
        quiet=False,
        source=_src,
        selsplit=_split,
    )
    outdir = os.path.abspath(f"./DSPS_hdf5_BOOTSTRAP{inputs['bootstrap_type']}_{_src}_{_split}_{_fit_type}")

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    filename_params = os.path.join(outdir, f"fitparams_{_fit_type}_bs_{inputs['bootstrap_type']}.h5")
    status = bootstrapFitsToHDF5(filename_params, sel_df, fit_means, fit_stds, pars_dict)
    print(status)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
