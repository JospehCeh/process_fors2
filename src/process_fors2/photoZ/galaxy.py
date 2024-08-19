#!/bin/env python3

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, vmap

from process_fors2.photoZ import nz_prior_core, prior_alpt0, prior_ft, prior_kt, prior_ktf, prior_pcal, prior_zot

Observation = namedtuple("Observation", ["num", "ref_i_AB", "AB_colors", "AB_colerrs", "valid_filters", "valid_colors", "z_spec"])


def load_galaxy(photometry, ismag, id_i_band=3):
    """load_galaxy _summary_

    :param photometry: fluxes or magnitudes and corresponding errors as read from an ASCII input file
    :type photometry: list or array-like
    :param ismag: whether photometry is provided as AB-magnitudes or fluxes
    :type ismag: bool
    :param id_i_band: index of i-band in the photometry. The default is 3 for LSST u, g, r, i, z, y.
    :type id_i_band: int, optional
    :return: Tuple containing the i-band AB magnitude, the array of color indices for the observations (in AB mag units), the corresponding errors
    and the array of booleans indicating which filters were used for this observation.
    :rtype: tuple
    """
    assert len(photometry) % 2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
    _phot = jnp.array([photometry[2 * i] for i in range(len(photometry) // 2)])
    _phot_errs = jnp.array([photometry[2 * i + 1] for i in range(len(photometry) // 2)])

    if ismag:
        c_ab = _phot[:-1] - _phot[1:]
        c_ab_err = jnp.power(jnp.power(_phot_errs[:-1], 2) + jnp.power(_phot_errs[1:], 2), 0.5)
        i_ab = _phot[id_i_band]
        filters_to_use = _phot > 0.0
    else:
        c_ab = -2.5 * jnp.log10(_phot[:-1] / _phot[1:])
        c_ab_err = 2.5 / jnp.log(10) * jnp.power(jnp.power(_phot_errs[:-1] / _phot[:-1], 2) + jnp.power(_phot_errs[1:] / _phot[1:], 2), 0.5)
        i_ab = -2.5 * jnp.log10(_phot[id_i_band]) - 48.6
        filters_to_use = jnp.isfinite(_phot)
    colors_to_use = jnp.array([b1 and b2 for (b1, b2) in zip(filters_to_use[:-1], filters_to_use[1:], strict=True)])
    return i_ab, c_ab, c_ab_err, filters_to_use, colors_to_use


@jit
def chi_term(obs, ref, err):
    """chi_term _summary_

    :param obs: _description_
    :type obs: _type_
    :param ref: _description_
    :type ref: _type_
    :param err: _description_
    :type err: _type_
    :return: _description_
    :rtype: _type_
    """
    return jnp.power((obs - ref) / err, 2.0)


vmap_chi_term = vmap(chi_term, in_axes=(None, 0, None))  # vmap version to compute the chi value for all colors of a single template, i.e. for all redshifts values


@jit
def z_prior_val(i_mag, zp, nuvk):
    """z_prior_val _summary_

    :param i_mag: _description_
    :type i_mag: _type_
    :param zp: _description_
    :type zp: _type_
    :param nuvk: _description_
    :type nuvk: _type_
    :return: _description_
    :rtype: _type_
    """
    alpt0, zot, kt, pcal, ktf_m, ft_m = prior_alpt0(nuvk), prior_zot(nuvk), prior_kt(nuvk), prior_pcal(nuvk), prior_ktf(nuvk), prior_ft(nuvk)
    val_prior = nz_prior_core(zp, i_mag, alpt0, zot, kt, pcal, ktf_m, ft_m)
    return val_prior


vmap_nz_prior = vmap(z_prior_val, in_axes=(None, 0, None))  # vmap version to compute the prior value for a certain observation and a certain SED template at all redshifts


@jit
def val_neg_log_posterior(z_val, templ_cols, gal_cols, gel_colerrs, gal_iab, templ_nuvk):
    """val_neg_log_posterior _summary_

    :param z_val: _description_
    :type z_val: _type_
    :param templ_cols: _description_
    :type templ_cols: _type_
    :param gal_cols: _description_
    :type gal_cols: _type_
    :param gel_colerrs: _description_
    :type gel_colerrs: _type_
    :param gal_iab: _description_
    :type gal_iab: _type_
    :param templ_nuvk: _description_
    :type templ_nuvk: _type_
    :return: _description_
    :rtype: _type_
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    _prior = z_prior_val(gal_iab, z_val, templ_nuvk)
    return jnp.sum(_chi) / len(_chi) - 2 * jnp.log(_prior)


vmap_neg_log_posterior = vmap(val_neg_log_posterior, in_axes=(0, 0, None, None, None, None))


# @jit
def neg_log_posterior(sps_temp, obs_gal):
    """neg_log_posterior _summary_

    :param sps_temp: _description_
    :type sps_temp: _type_
    :param obs_gal: _description_
    :type obs_gal: _type_
    :return: _description_
    :rtype: _type_
    """
    _sel = obs_gal.valid_colors
    return vmap_neg_log_posterior(sps_temp.redshift, sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel], obs_gal.ref_i_AB, sps_temp.nuvk)


@jit
def val_neg_log_likelihood(templ_cols, gal_cols, gel_colerrs):
    """val_neg_log_likelihood _summary_

    :param templ_cols: _description_
    :type templ_cols: _type_
    :param gal_cols: _description_
    :type gal_cols: _type_
    :param gel_colerrs: _description_
    :type gel_colerrs: _type_
    :return: _description_
    :rtype: _type_
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    return jnp.sum(_chi) / len(_chi)


vmap_neg_log_likelihood = vmap(val_neg_log_likelihood, in_axes=(0, None, None))


# @jit
def neg_log_likelihood(sps_temp, obs_gal):
    """neg_log_likelihood _summary_

    :param sps_temp: _description_
    :type sps_temp: _type_
    :param obs_gal: _description_
    :type obs_gal: _type_
    :return: _description_
    :rtype: _type_
    """
    _sel = obs_gal.valid_colors
    return vmap_neg_log_likelihood(sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel])


## Old functions for reference:
"""
def noV_est_chi2(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid, opacities):
    _selOpa = (wl_grid<1300.)
    interp_opac = jnp.interp(wl_grid, wl_grid[_selOpa], opacities, left=1., right=1., period=None)
    temp_fab = template.noJit_make_scaled_template(base_temp_lums, filters, extinc_arr, zp, cosmo, gal_fab, gal_fab_err, wl_grid, interp_opac)
    _terms = chi_term(gal_fab, temp_fab, gal_fab_err)
    chi2 = jnp.sum(_terms)/len(_terms)
    return chi2

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, 0, None))
def est_chi2_prior_jaxcosmo(i_mag, gal_col_ab, gal_col_ab_err, zphot, base_temp_lums, extinc_arr, filters, j_cosmo, wl_grid, opacities, prior_band):
    dist_mod = cosmology.calc_distMod(j_cosmo, zphot)
    prior_zp = z_prior_val(i_mag, zphot, base_temp_lums, wl_grid)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, cosmology.calc_distMod(j_cosmo, zphot))
    _terms = chi_term(gal_fab,\
                      template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zphot, wl_grid, cosmology.calc_distMod(j_cosmo, zp), opacities),\
                      gal_fab_err)
    if len(filters)<7 : debug.print("chi in bands = {t}", t=_terms)
    #chi2 = jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid))
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid, prior_band))
"""
