"""Module providing SPS fit utilities with jaxopt
Please note that data files from DSPS must be donwloaded in data folder:
The file must be downloaded from
https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/
refer to https://dsps.readthedocs.io/en/latest/quickstart.html
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from diffmah.defaults import DiffmahParams
from diffstar import calc_sfh_singlegal  # sfh_singlegal
from diffstar.defaults import DiffstarUParams  # , DEFAULT_Q_PARAMS
from dsps import calc_obs_mag, load_ssp_templates
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from interpax import interp1d
from jax import jit, vmap
from jax.numpy import trapezoid as trapz

from .dsps_params import SSPParametersFit
from .met_weights_age_dep import calc_rest_sed_sfh_table_lognormal_mdf_agedep

jax.config.update("jax_enable_x64", True)


def _get_package_dir() -> str:
    """get the path of this fitters package"""
    dirname = os.path.dirname(__file__)
    return dirname


_DUMMY_P_ADQ = SSPParametersFit()


@jit
def mean_sfr(params, z_obs):
    """Model of the SFR

    :param params: Fitted parameter dictionnary
    :type params: float as a dictionnary

    :param z_obs: redshift of the observations
    :type z_obs: float

    :return: array of the star formation rate
    :rtype: float

    """
    today_gyr = 13.8

    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]  # DEFAULT_MAH_PARAMS[1]
    MAH_early_index = params["MAH_early_index"]  # DEFAULT_MAH_PARAMS[2]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO, MAH_logtc, MAH_early_index, MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]  # DEFAULT_MS_PARAMS[1]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]  # DEFAULT_MS_PARAMS[4]
    list_param_ms = [MS_lgmcrit, MS_lgy_at_mcrit, MS_indx_lo, MS_indx_hi, MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt, Q_lg_drop, Q_lg_rejuv]

    # compute SFR
    tup_param_sfh = DiffstarUParams(tuple(list_param_ms), tuple(list_param_q))
    tup_param_mah = DiffmahParams(*list_param_mah)

    tarr = np.linspace(0.1, today_gyr, 100)
    # sfh_gal = sfh_singlegal(tarr, list_param_mah , list_param_ms, list_param_q,\
    #                        ms_param_type="unbounded", q_param_type="unbounded"\
    #                       )

    sfh_gal = calc_sfh_singlegal(tup_param_sfh, tup_param_mah, tarr)

    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
    t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

    # clear sfh in future
    # sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)

    return t_obs, tarr, sfh_gal


@jit
def ssp_spectrum_fromparam(params, z_obs, ssp_file=None):
    """Return the SED of SSP DSPS with original wavelength range wihout and with dust

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float

    :param ssp_file: SSP library location
    :type z_obs: path or str

    :return: the wavelength and the spectrum with dust and no dust
    :rtype: float

    """
    if ssp_file is None:
        from process_fors2.fetchData import DEFAULTS_DICT

        fullfilename_ssp_data = DEFAULTS_DICT["DSPS HDF5"]
        ssp_data = load_ssp_templates(fn=fullfilename_ssp_data)
    else:
        fullfilename_ssp_data = os.path.abspath(ssp_file)
        ssp_data = load_ssp_templates(fn=fullfilename_ssp_data)

    # compute the SFR
    t_obs, gal_t_table, gal_sfr_table = mean_sfr(params, z_obs)

    # age-dependant metallicity
    gal_lgmet_young = 2.0  # log10(Z)
    gal_lgmet_old = -3.0  # params["LGMET_OLD"] # log10(Z)
    gal_lgmet_scatter = 0.2  # params["LGMETSCATTER"] # lognormal scatter in the metallicity distribution function

    # compute the SED_info object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf_agedep(
        gal_t_table, gal_sfr_table, gal_lgmet_young, gal_lgmet_old, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs
    )
    # dust attenuation parameters
    Av = params["AV"]
    uv_bump = params["UV_BUMP"]
    plaw_slope = params["PLAW_SLOPE"]
    # list_param_dust = [Av, uv_bump, plaw_slope]

    # compute dust attenuation
    wave_spec_micron = ssp_data.ssp_wave / 10000
    k = sbl18_k_lambda(wave_spec_micron, uv_bump, plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k, Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    return ssp_data.ssp_wave, sed_info.rest_sed, sed_attenuated


@partial(vmap, in_axes=(None, None, 0, 0, None))
def _calc_mag(ssp_wls, sed_fnu, filt_wls, filt_transm, z_obs):
    return calc_obs_mag(ssp_wls, sed_fnu, filt_wls, filt_transm, z_obs, *DEFAULT_COSMOLOGY)


@jit
def mean_mags(X, params, z_obs, ssp_file=None):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_file: SSP library location
    :type z_obs: path or str

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_file)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    mags_predictions = jax.tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters, list_transm_filters)

    mags_predictions = jnp.array(mags_predictions)

    return mags_predictions


'''
@jit
def mean_ugri_sedpy(X, params, z_obs):
    """
    Return the photometric magnitudes for the given filters transmission (SDSS u, g, r and i).

    :param X: Tuple of filters to be used (SDSS u, g, r and i)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """
    # Get the filters
    # sdss_filts = observate.load_filters(["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0"])
    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs)

    # Transpose as F_lambda and to observation frame
    flam_att = convertFnuToFlambda_noU(ssp_wave, sed_attenuated)  # Return to sed_attenuated because magnitudes in KiDS are corrected for galactic extinction (not dust)
    wls_o, flam_o = convert_flux_toobsframe(ssp_wave, flam_att, z_obs)

    # calculate magnitudes in observation frame
    # mags_predictions = observate.getSED(wls_o, flam_o, filterlist=sdss_filts)
    mags_predictions = jax.tree_map(lambda x, y: ab_mag(x, y, ssp_wave, sed_attenuated), list_wls_filters, list_transm_filters)

    mags_predictions = jnp.array(mags_predictions)

    return mags_predictions
'''


@jit
def mean_spectrum(wls, params, z_obs, ssp_file):
    """Return the Model of SSP spectrum including Dust at the wavelength wls

    :param wls: wavelengths of the spectrum in rest frame
    :type wls: float

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float

    :param ssp_file: SSP library location
    :type z_obs: path or str

    :return: the spectrum
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_file)

    # interpolate with interpax which is differentiable
    # Fobs = jnp.interp(wls, ssp_data.ssp_wave, sed_attenuated)
    Fobs = interp1d(wls, ssp_wave, sed_attenuated, method="cubic")

    return Fobs


@jit
def mean_lines(wls, params, z_obs, refmod, reflines, ssp_file):
    """
    Estimates the contribution of spectral lines to the flux density yielded by DSPS with parameters `params`.

    Parameters
    ----------
    wls : array
        Wavelengths in angstrom.
    params : dict
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`, turned into a dictionary.
    z_obs : int or float
        Redshift of the object.
    refmod : array
        GELATO model of the flux density (in Lsun/Hz) (includes the contribution of SSP and spetral lines).
    reflines : array
        Contribution of the spectral lines to the flux density (in Lsun/Hz).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    array
        Flux density in Lsun/Hz : should be near 0. outside of spectral lines, non-zero in lines' regions.
    """
    fnu_rest = mean_spectrum(wls, params, z_obs, ssp_file)  # in Lsun / Hz
    return fnu_rest - refmod + reflines


@partial(vmap, in_axes=(None, None, 0))
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
    nancont = jnp.where((sur_wls > lin - cont_wid) * (sur_wls < lin - line_wid) + (sur_wls > lin + line_wid) * (sur_wls < lin + cont_wid), sur_spec, jnp.nan)
    height = jnp.nanmean(nancont)
    vals = jnp.where((sur_wls > lin - line_wid) * (sur_wls < lin + line_wid), sur_spec / height - 1.0, 0.0)
    ew = trapz(vals, x=sur_wls)
    return ew


@jit
def lik_lines(p, wls, refmod, reflines, fnuerr, z_obs, ssp_file=None):
    r"""
    Negative log-likelihood ($\Chi^2$) of the SPS defined by the parameters `p` with respect to previously known spectral lines.

    Parameters
    ----------
    p : array
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`.
    wls : array
        Wavelengths in angstrom.
    refmod : array
        GELATO model of the flux density (in Lsun/Hz) (includes the contribution of SSP and spetral lines).
    reflines : array
        Contribution of the spectral lines to the flux density (in Lsun/Hz).
    fnuerr : array
        Estimated errors in the flux densities.
    z_obs : int or float
        Redshift of the object.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    float
        Value of the negative log-likelihood ($\Chi^2$).
    """
    params = {name: p[k] for k, name in enumerate(_DUMMY_P_ADQ.PARAM_NAMES_FLAT)}
    resid = trapz(mean_lines(wls, params, z_obs, refmod, reflines, ssp_file), x=wls) - trapz(reflines, x=wls)
    return jnp.sum((resid / fnuerr) ** 2)


@jit
def lik_rew(p, surwls, rews_wls, rews, rews_err, z_obs, ssp_file=None):
    r"""
    Negative log-likelihood ($\Chi^2$) of the SPS defined by the parameters `p` with respect to previously known spectral lines.

    Parameters
    ----------
    p : array
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`.
    surwls : array
        Wavelengths in angstrom - should be oversampled so that spectral lines can be sampled with a sufficiently high resolution (step of 0.1 angstrom is recommended)
    rews_wls : array
        Central wavelengths (in angstrom) of lines to be studied.
    rews : array
        Reference values of equivalent widths of the studied lines.
    rews_err : array
        Dispersions of the reference equivalent widths of the studied lines.
    z_obs : int or float
        Redshift of the object.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    float
        Value of the negative log-likelihood ($\Chi^2$).
    """
    params = {name: p[k] for k, name in enumerate(_DUMMY_P_ADQ.PARAM_NAMES_FLAT)}
    spec = mean_spectrum(surwls, params, z_obs, ssp_file)
    # surwls, _, spec = ssp_spectrum_fromparam(params, z_obs)
    rew_predictions = calc_eqw(surwls, spec, rews_wls)
    resid = rew_predictions - rews
    return jnp.sum(jnp.power(resid / rews_err, 2))


## BACKUP old doc when surwls was the second argument ##
"""
surwls : array
    Wavelengths in angstrom - should be oversampled so that spectral lines can be sampled with a sufficiently high resolution (step of 0.1 angstrom is recommended)
"""
## END BACKUP ##


@jit
def lik_spec(p, wls, F, sigma_obs, z_obs, ssp_file=None) -> float:
    """
    neg loglikelihood(parameters,x,y,sigmas) for the spectrum

    :param p: flat array of parameters to fit
    :param z_obs: redshift of the observations
    :type z_obs: float
    :param ssp_file: SSP library location
    :type z_obs: path or str

    :return: the chi2 value
    :rtype: float
    """

    params = {name: p[k] for k, name in enumerate(_DUMMY_P_ADQ.PARAM_NAMES_FLAT)}

    # rescaling parameter for spectra  are pre-calculated and applied to data
    # scaleF =  params["SCALE"]

    # residuals
    resid = mean_spectrum(wls, params, z_obs, ssp_file) - F  # *scaleF
    ev = (resid / sigma_obs) ** 2  # jnp.sum((resid/(sigma_obs*jnp.sqrt(scaleF)))** 2)

    return jnp.sum(ev)


@jit
def lik_spec_from_mag(p_tofit, p_fix, wls, F, sigma_obs, z_obs, ssp_file=None) -> float:
    """
    neg loglikelihood(parameters,x,y,sigmas) for the spectrum

    :param p: flat array of parameters to fit
    :param z_obs: redshift of the observations
    :type z_obs: float
    :param ssp_file: SSP library location
    :type z_obs: path or str


    :return: the chi2 value
    :rtype: float
    """
    pars = jnp.concatenate((p_tofit, p_fix), axis=None)  # As of now, parameters must be correctly ordered at fucntion call. All parameters to fit musst be before all fixed parameters.
    params = {_DUMMY_P_ADQ.PARAM_NAMES_FLAT[k]: val for k, val in enumerate(pars)}

    # rescaling parameter for spectra  are pre-calculated and applied to data
    # scaleF =  params["SCALE"]

    # residuals
    resid = mean_spectrum(wls, params, z_obs, ssp_file) - F  # *scaleF
    ev = (resid / sigma_obs) ** 2  # jnp.sum((resid/(sigma_obs*jnp.sqrt(scaleF)))** 2)

    return jnp.sum(ev)


@jit
def lik_normspec_from_mag(p_tofit, p_fix, wls, F, sigma_obs, z_obs, ssp_file=None) -> float:
    """
    neg loglikelihood(parameters,x,y,sigmas) for the spectrum

    :param p: flat array of parameters to fit
    :param z_obs: redshift of the observations
    :type z_obs: float
    :param ssp_file: SSP library location
    :type z_obs: path or str


    :return: the chi2 value
    :rtype: float
    """
    pars = jnp.concatenate((p_tofit, p_fix), axis=None)  # As of now, parameters must be correctly ordered at function call. In this case, all parameters to fit must be before all fixed parameters.
    params = {_DUMMY_P_ADQ.PARAM_NAMES_FLAT[k]: val for k, val in enumerate(pars)}

    # Normalize spectra to try and get rid of scaling parameter
    sps_spec = mean_spectrum(wls, params, z_obs, ssp_file)
    _norm_fors = trapz(F, x=wls)
    _norm_sps = trapz(sps_spec, x=wls)

    # residuals
    resid = sps_spec / _norm_sps - F / _norm_fors

    return jnp.sum((resid / (sigma_obs / jnp.sqrt(_norm_fors))) ** 2)


@jit
def lik_mag_partial(p_tofit, p_fix, xf, mags_measured, sigma_mag_obs, z_obs, ssp_file=None):
    """
    neg loglikelihood(parameters,x,y,sigmas) for the photometry
    """

    pars = jnp.concatenate((p_fix, p_tofit), axis=None)  # As of now, parameters must be correctly ordered at fucntion call. In this case, all parameters to fit must be after all fixed parameters.
    params = {_DUMMY_P_ADQ.PARAM_NAMES_FLAT[k]: val for k, val in enumerate(pars)}

    all_mags_predictions = mean_mags(xf, params, z_obs, ssp_file)
    resid = mags_measured - all_mags_predictions

    return jnp.sum((resid / sigma_mag_obs) ** 2)


@jit
def lik_mag(p, xf, mags_measured, sigma_mag_obs, z_obs, ssp_file=None):
    """
    neg loglikelihood(parameters,x,y,sigmas) for the photometry
    """

    params = {name: p[k] for k, name in enumerate(_DUMMY_P_ADQ.PARAM_NAMES_FLAT)}

    all_mags_predictions = mean_mags(xf, params, z_obs, ssp_file)
    resid = mags_measured - all_mags_predictions

    return jnp.sum((resid / sigma_mag_obs) ** 2)


'''
@jit
def lik_ugri_sedpy(p, xf, mags_measured, sigma_mag_obs, z_obs):
    """
    neg loglikelihood(parameters,x,y,sigmas) for the photometry
    """

    params = {name: p[k] for k, name in enumerate(_DUMMY_P_ADQ.PARAM_NAMES_FLAT)}

    all_mags_predictions = mean_ugri_sedpy(xf, params, z_obs)
    resid = mags_measured - all_mags_predictions

    return jnp.sum((resid / sigma_mag_obs) ** 2)
'''


@jit
def lik_comb(p, xc, datac, sigmac, z_obs, weight=0.5, ssp_file=None):
    """
    neg loglikelihood(parameters,xc,yc,sigmasc) combining the spectroscopy and the photometry

    Xc = [Xspec_data, Xf_sel]
    Yc = [Yspec_data, mags_measured ]
    EYc = [EYspec_data, data_selected_magserr]

    weight must be between 0 and 1
    """

    resid_spec = lik_spec(p, xc[0], datac[0], sigmac[0], z_obs, ssp_file)
    resid_phot = lik_mag(p, xc[1], datac[1], sigmac[1], z_obs, ssp_file)

    return weight * resid_phot + (1 - weight) * resid_spec


@jit
def lik_mag_rew(p, xf, mags_measured, sigma_mag_obs, surwls, rews_wls, rews, rews_err, z_obs, weight_mag=0.5, ssp_file=None):
    r"""
    neg loglikelihood(parameters,xc,yc,sigmasc) combining the lines rest equivalent widths and the photometry

    Parameters
    ----------
    p : array
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`.
    xf : 2-tuple of lists
        Filters in which the photometry is taken. First element of the 2-tuple is the list of arrays of wavelengths for each filter, second element is the list of arrays of corresponding transmission.
    mags_measured : array of float
        AB-magnitudes for reference (observations to fit).
    sigma_mag_obs : array of float
        Errors (std dev) on reference AB mags.
    surwls : array
        Wavelengths in angstrom - should be oversampled so that spectral lines can be sampled with a sufficiently high resolution (step of 0.1 angstrom is recommended)
    rews_wls : array
        Central wavelengths (in angstrom) of lines to be studied.
    rews : array
        Reference values of equivalent widths of the studied lines.
    rews_err : array
        Dispersions of the reference equivalent widths of the studied lines.
    z_obs : int or float
        Redshift of the object.
    weight_mag : float, optional
        Weight of the fit on photometry. 1-weight_mag is affected to the fit on rest equivalent widths. Must be between 0.0 and 1.0. The default is 0.5.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    float
        Value of the negative log-likelihood ($\Chi^2$).
    """

    resid_spec = lik_rew(p, surwls, rews_wls, rews, rews_err, z_obs, ssp_file)
    resid_phot = lik_mag(p, xf, mags_measured, sigma_mag_obs, z_obs, ssp_file)

    return weight_mag * resid_phot + (1 - weight_mag) * resid_spec


def get_infos_spec(res, model, wls, F, eF, z_obs):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param wls: _description_
    :type wls: _type_
    :param F: _description_
    :type F: _type_
    :param eF: _description_
    :type eF: _type_
    :return: _description_
    :rtype: _type_
    """
    params = res.params
    fun_min = model(params, wls, F, eF, z_obs)
    jacob_min = jax.jacfwd(model)(params, wls, F, eF, z_obs)
    # covariance matrix of parameters
    inv_hessian_min = jax.scipy.linalg.inv(jax.hessian(model)(params, wls, F, eF, z_obs))
    return params, fun_min, jacob_min, inv_hessian_min


def get_infos_mag(res, model, xf, mgs, mgse, z_obs):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param xf: _description_
    :type xf: _type_
    :param mgs: _description_
    :type mgs: _type_
    :param mgse: _description_
    :type mgse: _type_
    :return: _description_
    :rtype: _type_
    """
    params = res.params
    fun_min = model(params, xf, mgs, mgse, z_obs)
    jacob_min = jax.jacfwd(model)(params, xf, mgs, mgse, z_obs)
    # covariance matrix of parameters
    inv_hessian_min = jax.scipy.linalg.inv(jax.hessian(model)(params, xf, mgs, mgse, z_obs))
    return params, fun_min, jacob_min, inv_hessian_min


def get_infos_comb(res, model, xc, datac, sigmac, z_obs, weight):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param xc: _description_
    :type xc: _type_
    :param datac: _description_
    :type datac: _type_
    :param sigmac: _description_
    :type sigmac: _type_
    :param weight: _description_
    :type weight: _type_
    :return: _description_
    :rtype: _type_
    """
    params = res.params
    fun_min = model(params, xc, datac, sigmac, z_obs, weight=weight)
    jacob_min = jax.jacfwd(model)(params, xc, datac, sigmac, z_obs, weight=weight)
    # covariance matrix of parameters
    inv_hessian_min = jax.scipy.linalg.inv(jax.hessian(model)(params, xc, datac, sigmac, z_obs, weight=weight))
    return params, fun_min, jacob_min, inv_hessian_min
