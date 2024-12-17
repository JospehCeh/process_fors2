#!/usr/bin/env python3
#
#  Analysis.py
#
#  Copyright 2023  <joseph@wl-chevalier>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


import os

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from tqdm import tqdm

from process_fors2.fetchData import json_to_inputs
from process_fors2.stellarPopSynthesis import has_redshift

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid

from process_fors2.stellarPopSynthesis import SSPParametersFit

_DUMMY_PARS = SSPParametersFit()
PARS_DF = pd.DataFrame(index=_DUMMY_PARS.PARAM_NAMES_FLAT, columns=["INIT", "MIN", "MAX"], data=jnp.column_stack((_DUMMY_PARS.INIT_PARAMS, _DUMMY_PARS.PARAMS_MIN, _DUMMY_PARS.PARAMS_MAX)))
INIT_PARAMS = jnp.array(PARS_DF["INIT"])
PARAMS_MIN = jnp.array(PARS_DF["MIN"])
PARAMS_MAX = jnp.array(PARS_DF["MAX"])

"""
Reminder :
Cosmo = namedtuple('Cosmo', ['h0', 'om0', 'l0', 'omt'])
sedpyFilter = namedtuple('sedpyFilter', ['name', 'wavelengths', 'transmission'])
BaseTemplate = namedtuple('BaseTemplate', ['name', 'flux', 'z_sps'])
SPS_Templates = namedtuple('SPS_Templates', ['name', 'redshift', 'z_grid', 'i_mag', 'colors', 'nuvk'])
Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'z_spec'])
DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])
"""

# conf_json = 'EmuLP/COSMOS2020-with-FORS2-HSC_only-jax-CC-togglePriorTrue-opa.json' # attention à la localisation du fichier !


def load_data_for_run(inp_glob):
    """load_data_for_run Generates input data from the inputs configuration dictionary

    :param inp_glob: input configuration and settings
    :type inp_glob: dict
    :return: data for photo-z evaluation : redshift grid, templates dictionary and the arrays of processed observed data (input catalog) (i mags ; colors ; errors on colors ; spectro-z).
    :rtype: 6-tuple of jax.ndarray, dictionary, jax.ndarray, jax.ndarray, jax.ndarray, jax.ndarray
    """
    from interpax import interp1d

    from process_fors2.photoZ import DATALOC, NIR_filt, NUV_filt, get_2lists, load_filt, sedpyFilter
    from process_fors2.stellarPopSynthesis import load_ssp

    _ssp_file = (
        None
        if (inp_glob["fitDSPS"]["ssp_file"].lower() == "default" or inp_glob["fitDSPS"]["ssp_file"] == "" or inp_glob["fitDSPS"]["ssp_file"] is None)
        else os.path.abspath(inp_glob["fitDSPS"]["ssp_file"])
    )
    ssp_data = load_ssp(_ssp_file)

    inputs = inp_glob["photoZ"]
    z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + inputs["Z_GRID"]["z_step"], inputs["Z_GRID"]["z_step"])
    wl_grid = jnp.arange(inputs["WL_GRID"]["lambda_min"], inputs["WL_GRID"]["lambda_max"] + inputs["WL_GRID"]["lambda_step"], inputs["WL_GRID"]["lambda_step"])

    filters_dict = inputs["Filters"]
    for _f in filters_dict:
        filters_dict[_f]["path"] = os.path.abspath(os.path.join(DATALOC, filters_dict[_f]["path"]))
    print("Loading filters :")
    filters_arr = tuple(sedpyFilter(*load_filt(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict)) + (NUV_filt, NIR_filt)

    filters_names = [_f["name"] for _, _f in filters_dict.items()]
    wls, trans = get_2lists(filters_arr)
    transm_arr = jnp.array([interp1d(wl_grid, wl, tr, method="linear", extrap=0.0) for wl, tr in zip(wls, trans, strict=True)])

    print("Building templates :")
    sps_temp_h5 = os.path.abspath(inputs["Templates"]["input"])
    templ_df = pd.read_hdf(sps_temp_h5)
    pars_arr = jnp.array(templ_df[_DUMMY_PARS.PARAM_NAMES_FLAT])

    """
    Xfilt = get_2lists(filters_arr)
    # sps_temp_pkl = os.path.abspath(inputs["Templates"])
    # sps_par_dict = read_params(sps_temp_pkl)
    if inputs["Templates"]["overwrite"] or not os.path.isfile(os.path.abspath(inputs["Templates"]["output"])):
        sps_temp_h5 = os.path.abspath(inputs["Templates"]["input"])
        sps_par_dict = readDSPSHDF5(sps_temp_h5)
        if "sps" in inputs["Mode"].lower():
            templ_dict = jax.tree_util.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        else:
            templ_dict = jax.tree_util.tree_map(lambda dico: make_legacy_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        _ = templatesToHDF5(inputs["Templates"]["output"], templ_dict)
    else:
        templ_dict = readTemplatesHDF5(inputs["Templates"]["output"])
    """

    print("Loading observations :")
    data_path = os.path.abspath(inputs["Dataset"]["path"])
    data_ismag = inputs["Dataset"]["type"].lower() == "m"

    if inputs["Dataset"]["is_ascii"]:
        from process_fors2.fetchData import catalog_ASCIItoHDF5

        h5catpath = catalog_ASCIItoHDF5(data_path, data_ismag, filt_names=filters_names)
    else:
        h5catpath = data_path

    clrh5file = f"pz_inputs_iclrs_{os.path.basename(h5catpath)}" if inputs["i_colors"] else f"pz_inputs_{os.path.basename(h5catpath)}"

    if inputs["Dataset"]["overwrite"] or not (os.path.isfile(clrh5file)):
        from process_fors2.fetchData import readCatalogHDF5

        ab_mags, ab_mags_errs, z_specs = readCatalogHDF5(h5catpath, filt_names=filters_names)

        from .galaxy import vmap_mags_to_i_and_colors, vmap_mags_to_i_and_icolors

        i_mag_ab, ab_colors, ab_cols_errs = (
            vmap_mags_to_i_and_icolors(ab_mags, ab_mags_errs, inputs["i_band_num"]) if inputs["i_colors"] else vmap_mags_to_i_and_colors(ab_mags, ab_mags_errs, inputs["i_band_num"])
        )

        from process_fors2.fetchData import pzInputsToHDF5

        _colrs_h5out = pzInputsToHDF5(clrh5file, ab_colors, ab_cols_errs, z_specs, i_mag_ab, filt_names=filters_names, i_colors=inputs["i_colors"], iband_num=inputs["i_band_num"])
    else:
        from process_fors2.fetchData import readPZinputsHDF5

        i_mag_ab, ab_colors, ab_cols_errs, z_specs = readPZinputsHDF5(clrh5file, filt_names=filters_names, i_colors=inputs["i_colors"], iband_num=inputs["i_band_num"])

    """old-fashioned way, deprecated, kept for reference only
    N_FILT = len(filters_arr) - 2
    data_file_arr = np.loadtxt(data_path)
    obs_arr = []

    for i in tqdm(range(data_file_arr.shape[0])):
        try:
            assert (len(data_file_arr[i, :]) == 1 + 2 * N_FILT) or (
                len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1
            ), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            # print(int(data_file_arr[i, 0]))
            if len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), data_file_arr[i, 2 * N_FILT + 1])
            else:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), jnp.nan)
            # print(observ.num)
            obs_arr.extend([observ])
        except AssertionError:
            pass
    """

    return z_grid, wl_grid, transm_arr, pars_arr, i_mag_ab, ab_colors, ab_cols_errs, z_specs, ssp_data


@jax.jit
def _cdf(z, pdz):
    cdf = jnp.array([trapezoid(pdz[:i], x=z[:i]) for i in range(len(z))])
    return cdf


@jax.jit
def _median(z, pdz):
    cdz = _cdf(z, pdz)
    medz = z[jnp.nonzero(cdz >= 0.5, size=1)][0]
    return medz


vmap_median = jax.vmap(_median, in_axes=(None, 1))


@jax.jit
def _mean(z, pdz):
    return trapezoid(z * pdz, x=z)


vmap_mean = jax.vmap(_mean, in_axes=(None, 1))


def extract_pdz(pdf_arr, zs, z_grid):
    """extract_pdz Computes and returns the marginilized Probability Density function of redshifts and associated statistics for all observations.
    Each item of the `pdf_arr` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_arr: Output of photo-z estimation as a dictonary of JAX arrays.
    :type pdf_arr: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and associated summarized statistics
    :rtype: dict
    """
    # pdf_dict = pdf_res[0]
    # zs = pdf_res[1]
    # pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    pdz_arr = jnp.nansum(pdf_arr, axis=0)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "z_spec": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def extract_pdz_pars_z_anu(pdf_arr, zs, z_grid, anu_grid):
    """extract_pdz_pars_z_anu Computes and returns the marginilized Probability Density function of redshifts and associated statistics for all observations.
    Each item of the `pdf_arr` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_arr: Output of photo-z estimation as a dictonary of JAX arrays.
    :type pdf_arr: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :param anu_grid: Grid of dust law parameter $A_\nu$ values on which the likelihood was computed
    :type anu_grid: jax array
    :return: Marginalized Probability Density function of redshift values and associated summarized statistics
    :rtype: dict
    """
    print(f"DEBUG : {len(zs)} obs. ; {len(z_grid)} z vals ; {len(anu_grid)} Av vals ; {pdf_arr.shape[0]} templates")
    _n2 = trapezoid(trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0), x=anu_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    print(f"DEBUG : Shape of PDF : {pdf_arr.shape}")
    arr_anu_sel = jnp.nanmax(pdf_arr, axis=2)
    print(f"DEBUG : Shape of Av-selected posterior values : {arr_anu_sel.shape}")
    pdz_arr = jnp.nansum(arr_anu_sel, axis=0)
    # marg_anu = trapezoid(pdz_arr, x=anu_grid, axis=1)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "z_spec": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def extract_pdz_fromchi2(chi2_dict, zs, z_grid):
    r"""extract_pdz_fromchi2 Similar to extract_pdz except takes $\chi^2$ values as inputs (*i.e.* negative log-likelihood).
    Computes and returns the marginilized Probability Density function of redshifts

    :param chi2_dict: Output of photo-z estimation as a dictonary of JAX arrays.
    :type chi2_dict: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and elementary associated stats
    :rtype: dict
    """
    chi2_arr = jnp.array([chi2_templ for _, chi2_templ in chi2_dict.items()])
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    chi2_arr = chi2_arr * _n1
    exp_arr = jnp.power(jnp.exp(-0.5 * chi2_arr), 1 / _n1)
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = trapezoid(jnp.nansum(exp_arr, axis=0), x=z_grid)
    exp_arr = exp_arr / _n2
    pdz_arr = jnp.nansum(exp_arr, axis=0)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "z_spec": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def extract_pdz_allseds(pdf_dict, zs, z_grid):
    """extract_pdz_allseds Computes and returns the marginilized Probability Density function of redshifts for a single observation ;
    The conditional probability density is also computed for each galaxy template.
    Each item of the `pdf_dict` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_dict: Output of photo-z estimation as a dictonary of JAX arrays.
    :type pdf_dict: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and conditional PDF for each template.
    :rtype: dict
    """
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    _n2 = trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    pdz_arr = jnp.nansum(pdf_arr, axis=0)
    templ_wgts = trapezoid(pdf_arr, x=z_grid, axis=1)
    sed_evid_z = jnp.nansum(pdf_arr, axis=2)
    sed_evid_marg = jnp.nansum(templ_wgts, axis=1)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {
        "z_grid": z_grid,
        "PDZ": pdz_arr,
        "p(z, sed)": pdf_arr,
        "z_spec": zs,
        "z_ML": z_MLs,
        "z_mean": z_means,
        "z_med": z_meds,
        "SED weights per galaxy": templ_wgts,
        "SED evidence along z": sed_evid_z,
        "Marginalised SED evidence": sed_evid_marg,
    }
    return pdz_dict


def run_from_inputs(inputs):
    """run_from_inputs Run the photometric redshifts estimation with the given input settings.

    :param inputs: Input settings for the photoZ run. Can be loaded from a `JSON` file using `process_fors2.fetchData.json_to_inputs`.
    :type inputs: dict
    :return: Photo-z estimation results. These are not written to disk within this function.
    :rtype: list (tree-like)
    """
    from process_fors2.photoZ import (
        likelihood_pars_z_anu,
        likelihood_pars_z_anu_iclrs,
        load_data_for_run,
        posterior_pars_z_anu,
        posterior_pars_z_anu_iclrs,
        vmap_z_nllik,
        vmap_z_nllik_iclrs,
        vmap_z_prior_pars_zanu,
    )
    from process_fors2.stellarPopSynthesis import istuple

    z_grid, wl_grid, transm_arr, templ_parsarr, observed_imags, observed_colors, observed_noise, observed_zs, sspdata = load_data_for_run(inputs)

    """Old, deprecated way, kept here for reference and safety
    observed_colors = jnp.array([obs.AB_colors for obs in obs_arr])
    observed_noise = jnp.array([obs.AB_colerrs for obs in obs_arr])
    observed_zs = jnp.array([obs.z_spec for obs in obs_arr])
    observed_imags = jnp.array([obs.ref_i_AB for obs in obs_arr])
    """

    """Dust and Opacity are normally included in DSPS calculations
    ebvs_in_use = jnp.array([d.EBV for d in dust_arr])
    laws_in_use = jnp.array([0 if d.name == "Calzetti" else 1 for d in dust_arr])

    _old_dir = os.getcwd()
    _path = os.path.abspath(__file__)
    _dname = os.path.dirname(_path)
    os.chdir(_dname)
    opa_path = os.path.abspath(inputs['Opacity'])
    #ebv_prior_file = inputs['E(B-V) prior file']
    #ebv_prior_df = pd.read_pickle(ebv_prior_file)
    #cols_to_stack = tuple(ebv_prior_df[col].values for col in ebv_prior_df.columns)
    #ebv_prior_arr = jnp.column_stack(cols_to_stack)
    os.chdir(_old_dir)

    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = extinction.load_opacity(opa_path, wls_opa)
    extrap_ones = jnp.ones((len(z_grid), len(wl_grid)-len(wls_opa)))
    """

    print("Photometric redshift estimation (please be patient, this may take a couple of hours on large datasets) :")

    """
    def has_sps_template(cont):
        return isinstance(cont, SPS_Templates)

    def estim_zp(observs_cols, observs_errs, observs_i):
        # c = observ.AB_colors[observ.valid_colors]
        # c_err = observ.AB_colerrs[observ.valid_colors]
        if inputs["photoZ"]["prior"]:  # and observ.valid_filters[inputs["photoZ"]["i_band_num"]]:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: posterior(sps_templ.colors, observs_cols, observs_errs, observs_i, sps_templ.z_grid, sps_templ.nuvk), templ_pars_list, is_leaf=istuple)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(
                    lambda sps_templ: posterior_fluxRatio(sps_templ.colors, observs_cols, observs_errs, observs_i, sps_templ.z_grid, sps_templ.nuvk), templ_pars_list, is_leaf=istuple
                )
            )
        else:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: likelihood(sps_templ.colors, observs_cols, observs_errs), templ_pars_list, is_leaf=istuple)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(lambda sps_templ: likelihood_fluxRatio(sps_templ.colors, observs_cols, observs_errs), templ_pars_list, is_leaf=istuple)
            )
        # z_phot_loc = jnp.nanargmin(chi2_arr)
        return probz_dict  # , observ.z_spec  # chi2_arr, z_phot_loc
    """

    anu_arr = jnp.arange(PARS_DF.loc["AV", "MIN"], PARS_DF.loc["AV", "MAX"] + 0.5, 0.5)

    if inputs["photoZ"]["Templates"]["as_array"]:
        if inputs["photoZ"]["prior"]:
            if inputs["photoZ"]["i_colors"]:
                probz_arr = posterior_pars_z_anu_iclrs(templ_parsarr, z_grid, anu_arr, observed_colors, observed_noise, observed_imags, wl_grid, transm_arr, sspdata, inputs["photoZ"]["i_band_num"])
            else:
                probz_arr = posterior_pars_z_anu(templ_parsarr, z_grid, anu_arr, observed_colors, observed_noise, observed_imags, wl_grid, transm_arr, sspdata)
        else:
            if inputs["photoZ"]["i_colors"]:
                probz_arr = likelihood_pars_z_anu_iclrs(templ_parsarr, z_grid, anu_arr, observed_colors, observed_noise, wl_grid, transm_arr[:-2, :], sspdata, inputs["photoZ"]["i_band_num"])
            else:
                probz_arr = likelihood_pars_z_anu(templ_parsarr, z_grid, anu_arr, observed_colors, observed_noise, wl_grid, transm_arr[:-2, :], sspdata)
    else:
        templ_pars_list = [tuple(_fp) for _fp in templ_parsarr]
        if inputs["photoZ"]["i_colors"]:
            probz_arr = jax.tree_util.tree_map(
                lambda parstupl: vmap_z_nllik_iclrs(jnp.array(parstupl), z_grid, anu_arr, observed_colors, observed_noise, wl_grid, transm_arr[:-2, :], sspdata, inputs["photoZ"]["i_band_num"]),
                templ_pars_list,
                is_leaf=istuple,
            )
        else:
            probz_arr = jax.tree_util.tree_map(
                lambda parstupl: vmap_z_nllik(jnp.array(parstupl), z_grid, anu_arr, observed_colors, observed_noise, wl_grid, transm_arr[:-2, :], sspdata),
                templ_pars_list,
                is_leaf=istuple,
            )
        probz_arr = jnp.array(probz_arr)
        _n1 = 100.0 / jnp.nanmax(probz_arr)
        probz_arr = _n1 * probz_arr
        probz_arr = jnp.power(jnp.exp(-0.5 * probz_arr), 1 / _n1)

        if inputs["photoZ"]["prior"]:
            prior_arr = jax.tree_util.tree_map(
                lambda parstupl: vmap_z_prior_pars_zanu(jnp.array(parstupl), z_grid, anu_arr, observed_imags, wl_grid, transm_arr[-2:, :], sspdata), templ_pars_list, is_leaf=istuple
            )
            prior_arr = jnp.array(prior_arr)
            probz_arr *= prior_arr

    """
    def is_obs(elt):
        return isinstance(elt, Observation)
    tree_of_results_dict = jax.tree_util.tree_map(lambda elt: extract_pdz(estim_zp(elt), z_grid), obs_arr, is_leaf=is_obs)
    """

    results_dict = extract_pdz(probz_arr, observed_zs, z_grid)  # extract_pdz_pars_z_anu(probz_arr, observed_zs, z_grid, anu_arr)
    print("All done !")

    return results_dict


def load_data_for_analysis(conf_json):
    """load_data_for_analysis DEPRECATED - Similar to `load_data_for_run` but for analysis purposes. Inherited from `EmuLP` and not maintained since : might work, might not.

    :param conf_json: `JSON` configuration file location
    :type conf_json: str or path-like
    :return: Data useful for analysis of photometric redshifts estimation
    :rtype: tuple of array-like and tree-like (dict) data
    """
    inputs = json_to_inputs(conf_json)["photoZ"]
    from process_fors2.photoZ import DATALOC, NIR_filt, NUV_filt, Observation, get_2lists, load_filt, load_galaxy, make_jcosmo, make_sps_templates, read_params, sedpyFilter

    cosmo = make_jcosmo(inputs["Cosmology"]["h0"])
    # cosmo = Cosmo(
    #    inputs['Cosmology']['h0'],
    #    inputs['Cosmology']['om0'],
    #    inputs['Cosmology']['l0'],
    #    inputs['Cosmology']['om0'] + inputs['Cosmology']['l0']
    #    )

    z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + inputs["Z_GRID"]["z_step"], inputs["Z_GRID"]["z_step"])

    fine_z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + min(0.01, inputs["Z_GRID"]["z_step"]), min(0.01, inputs["Z_GRID"]["z_step"]))

    wl_grid = jnp.arange(inputs["WL_GRID"]["lambda_min"], inputs["WL_GRID"]["lambda_max"] + inputs["WL_GRID"]["lambda_step"], inputs["WL_GRID"]["lambda_step"])

    print("Loading filters :")
    filters_dict = inputs["Filters"]
    for _f in filters_dict:
        filters_dict[_f]["path"] = os.path.abspath(os.path.join(DATALOC, filters_dict[_f]["path"]))
    filters_arr = tuple(sedpyFilter(*load_filt(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict)) + (NUV_filt, NIR_filt)
    N_FILT = len(filters_arr) - 2

    named_filters = tuple(sedpyFilter(*load_filt(filters_dict[ident]["name"], filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict))

    print("Building templates :")
    Xfilt = get_2lists(filters_arr)
    sps_temp_pkl = os.path.abspath(inputs["Templates"])
    sps_par_dict = read_params(sps_temp_pkl)
    templ_dict = jax.tree_util.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, wl_grid, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)

    """Old-fashioned way
    templates_dict = inputs['Templates']
    for _t in templates_dict:
        templates_dict[_t]['path'] = os.path.abspath(templates_dict[_t]['path'])
    baseTemp_arr = tuple(
        template.BaseTemplate(
            *template.make_base_template(
                templates_dict[ident]["name"],
                templates_dict[ident]["path"],
                wl_grid
                )
        for ident in tqdm(templates_dict)
        )
    """

    """Dust is included in DSPS templates
    print("Generating dust attenuations laws :")
    extlaws_dict = inputs['Extinctions']
    for _e in extlaws_dict:
        extlaws_dict[_e]['path'] = os.path.abspath(extlaws_dict[_e]['path'])
    ebv_vals = jnp.array(inputs['e_BV'])
    dust_arr = []
    for ident in tqdm(extlaws_dict):
        dust_arr.extend(
            [
                extinction.DustLaw(
                    extlaws_dict[ident]['name'],
                    ebv,
                    extinction.load_extinc(
                        extlaws_dict[ident]['path'],
                        ebv,
                        wl_grid
                        )
                    )
                for ebv in tqdm(ebv_vals)
                ]
            )

    print("Loading IGM attenuations :")
    opa_path = os.path.abspath(inputs['Opacity'])
    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = extinction.load_opacity(opa_path, wls_opa)
    """

    print("Loading observations :")
    data_path = os.path.abspath(inputs["Dataset"]["path"])
    data_ismag = inputs["Dataset"]["type"].lower() == "m"

    data_file_arr = np.loadtxt(data_path)
    obs_arr = []

    for i in tqdm(range(data_file_arr.shape[0])):
        try:
            assert (len(data_file_arr[i, :]) == 1 + 2 * N_FILT) or (
                len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1
            ), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            # print(int(data_file_arr[i, 0]))
            if len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), data_file_arr[i, 2 * N_FILT + 1])
            else:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), jnp.nan)
            # print(observ.num)
            obs_arr.extend([observ])
        except AssertionError:
            pass
    return cosmo, z_grid, fine_z_grid, wl_grid, filters_arr[:-2], named_filters, templ_dict, obs_arr


"""Functions not in use at the moment : mostly marginalisations of any combination of bust law, E(B-V), SED template, redshift.
def results_in_dataframe(conf_json, observations, filters, filt_nums=(1,2,3,4)):
    inputs = json_to_inputs(conf_json)
    df_res = pd.read_pickle(f"{inputs['run name']}_results_summary.pkl")
    for j,obs in enumerate(tqdm(observations)):
        if j in df_res.index:
            assert obs.num == df_res.loc[j,'Id'], "Incorrect match Obs. number <-> Gal. ID"
            for i,filt in enumerate(filters):
                    df_res.loc[j, f"MagAB({filt.name})"] = -2.5*jnp.log10(obs.AB_fluxes[i])-48.6
                    df_res.loc[j, f"err_MagAB({filt.name})"] = 1.086*obs.AB_f_errors[i]/obs.AB_fluxes[i]
    df_res['Bias'] = df_res['Photometric redshift']-df_res['True redshift']
    #df_res['std'] = df_res['bias']/(1.+df_res['True redshift'])
    df_res['Outlier'] = np.abs(df_res['Bias']/(1.+df_res['True redshift']))>0.15
    df_res['U-B'] = df_res[f"MagAB({filters[filt_nums[0]].name})"]-df_res[f"MagAB({filters[filt_nums[1]].name})"]
    df_res['R-I'] = df_res[f"MagAB({filters[filt_nums[2]].name})"]-df_res[f"MagAB({filters[filt_nums[3]].name})"]
    #df_res['redness'] = df_res['U-B']/df_res['R-I']
    outl_rate = 100.0*len(df_res[df_res['Outlier']])/len(df_res)
    NMAD = 1.4821 * np.median(np.abs(df_res['Bias']/(1.+df_res['True redshift'])))

    print(f'Outlier rate = {outl_rate:.4f}% ; NMAD = {NMAD:.5f}')
    return df_res, outl_rate, NMAD

def enrich_dataframe(res_df, observations, filters, filt_nums=(1,2,3,4)):
    results_df = res_df.copy()
    for j,obs in enumerate(tqdm(observations)):
        if j in results_df.index:
            assert obs.num == results_df.loc[j,'Id'], "Incorrect match Obs. number <-> Gal. ID"
            for i,filt in enumerate(filters):
                results_df.loc[j, f"MagAB({filt.name})"] = -2.5*jnp.log10(obs.AB_fluxes[i])-48.6
                results_df.loc[j, f"err_MagAB({filt.name})"] = 1.086*obs.AB_f_errors[i]/obs.AB_fluxes[i]
    results_df['Bias'] = results_df['Photometric redshift']-results_df['True redshift']
    #results_df['std'] = results_df['bias']/(1.+results_df['True redshift'])
    results_df['Outlier'] = np.abs(results_df['Bias']/(1.+results_df['True redshift']))>0.15
    results_df['U-B'] = results_df[f"MagAB({filters[filt_nums[0]].name})"]-results_df[f"MagAB({filters[filt_nums[1]].name})"]
    results_df['R-I'] = results_df[f"MagAB({filters[filt_nums[2]].name})"]-results_df[f"MagAB({filters[filt_nums[3]].name})"]
    #results_df['redness'] = results_df['U-B']/df_res['R-I']
    outl_rate = 100.0*len(results_df[results_df['Outlier']])/len(results_df)
    NMAD = 1.4821 * np.median(np.abs(results_df['Bias']/(1.+results_df['True redshift'])))

    print(f'Outlier rate = {outl_rate:.4f}% ; NMAD = {NMAD:.5f}')
    return results_df.copy(), outl_rate, NMAD

@partial(jit, static_argnums=(1,2))
def probability_distrib(chi2_array, n_baseTemp, n_extLaws, EBVs, zgrid):
    # Compute the probability values
    probs_array = jnp.exp(-0.5*chi2_array)

    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=0)

        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=zgrid, axis=0)

        sub_ints_ebv_z.append(_int_z)

    ## Over laws
    _int_laws = jnp.trapz(jnp.array(sub_ints_ebv_z), x=jnp.arange(1, 1+len(sub_ints_ebv_z)), axis=0)

    return probs_array / _int_laws, _int_laws

@partial(jit, static_argnums=(1))
def probability_distrib_noDust(chi2_array, n_baseTemp, zgrid):
    # Compute the probability values
    probs_array = jnp.exp(-0.5*chi2_array)

    # Integrate successively:
    ## Over z
    _int_z = jnp.trapz(probs_array, x=zgrid, axis=2)
    #debug.print("Int over z = {n}", n=_int_z)

    ## Over models
    _int_mods = jnp.trapz(_int_z, x=jnp.arange(1, 1+n_baseTemp), axis=0)
    #debug.print("Norm = {n}", n=_int_mods)

    return probs_array / _int_mods[0], _int_mods[0]

@partial(jit, static_argnums=(1,2))
def probability_distrib_oneEBV(chi2_array, n_baseTemp, n_extLaws, zgrid):
    # Compute the probability values
    probs_array = jnp.exp(-0.5*chi2_array)

    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Intergrate over z
        _int_z = jnp.trapz(sub_arr, x=zgrid, axis=0)
        sub_ints_ebv_z.append(_int_z)

    ## Over laws
    _int_laws = jnp.trapz(jnp.array(sub_ints_ebv_z), x=jnp.arange(1, 1+len(sub_ints_ebv_z)), axis=0)

    return probs_array / _int_laws, _int_laws

@partial(jit, static_argnums=(1))
def probability_distrib_oneLaw(chi2_array, n_baseTemp, EBVs, zgrid):
    # Compute the probability values
    probs_array = jnp.exp(-0.5*chi2_array)

    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    ## Integration over E(B-V)
    _int_ebv = jnp.trapz(_int_mods, x=EBVs, axis=0)

    ## Intergrate over z
    _int_z = jnp.trapz(_int_ebv, x=zgrid, axis=0)

    return probs_array / _int_z, _int_z

@partial(jit, static_argnums=(1))
def prob_mod(probs_array, n_extLaws, EBVs, zgrid):
    # Integrate successively:
    _sub_ints = jnp.split(probs_array, n_extLaws, axis=1)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=1)
        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=zgrid, axis=1)
        sub_ints_ebv_z.append(_int_z)

    ## Over laws
    sub_ints_ebv_z_arr = jnp.array(sub_ints_ebv_z)
    _int_laws = jnp.trapz(sub_ints_ebv_z_arr, x=jnp.arange(1, 1+len(sub_ints_ebv_z)), axis=0)
    return _int_laws

@partial(jit, static_argnums=(1,2))
def prob_ebv(probs_array, n_baseTemp, n_extLaws, zgrid):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
    sub_ints_z = []
    for sub_arr in _sub_ints:
        ## Over z
        _int_z = jnp.trapz(sub_arr, x=zgrid, axis=1)
        sub_ints_z.append(_int_z)

    ## Over laws
    sub_ints_z_arr = jnp.array(sub_ints_z)
    _int_laws = jnp.trapz(sub_ints_z_arr, x=jnp.arange(1, 1+len(sub_ints_z)), axis=0)
    return _int_laws

@partial(jit, static_argnums=(1,2))
def prob_z(probs_array, n_baseTemp, n_extLaws, EBVs):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
    sub_ints_ebv = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=0)
        sub_ints_ebv.append(_int_ebv)

    ## Over laws
    sub_ints_ebv_arr = jnp.array(sub_ints_ebv)
    _int_laws = jnp.trapz(sub_ints_ebv_arr, x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)
    return _int_laws

@partial(jit, static_argnums=(1,2))
def prob_law(probs_array, n_baseTemp, n_extLaws, EBVs, zgrid):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=0)
        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=zgrid, axis=0)
        sub_ints_ebv_z.append(_int_z)
    sub_ints_z_arr = jnp.array(sub_ints_ebv_z)
    return sub_ints_z_arr

@partial(jit, static_argnums=(1,3))
def evidence(probs_array, n_extLaws, zgrid, split_laws=False):
    # it is really just returning the array integrated over z
    if split_laws:
        # returned dimension will be nb of laws, nb of base templates, nb of E(B-V)
        _sub_ints = jnp.split(probs_array, n_extLaws, axis=1)
        sub_ints_z = []
        for sub_arr in _sub_ints:
            ## Over z
            _int_z = jnp.trapz(sub_arr, x=zgrid, axis=2)
            sub_ints_z.append(_int_z)
            res = jnp.array(sub_ints_z)
    else:
        # returned dimension will be nb of base templates, nb of laws * nb of E(B-V)
        res = jnp.trapz(probs_array, x=zgrid, axis=2)
    return res

@partial(jit, static_argnums=(2,3,6,7))
def probs_at_fixed_z(probs_array, fixed_z, n_baseTemp, n_extLaws, EBVs, zgrid, renormalize=True, prenormalize=False):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array = jnp.zeros((probs_array.shape[0], probs_array.shape[1]))

    # Interpolate pdf at fixed_z
    for i in range(probs_array.shape[0]):
        for j in range(probs_array.shape[1]):
            _probs = probs_array[i,j,:]
            if prenormalize:
                _prenorm = jnp.trapz(_probs, x=zgrid, axis=0)
                _probs = _probs / _prenorm
            #f_interp = j_spline(z_grid, _probs, k=2)
            #_interp_pdf = f_interp(fixed_z)
            _interp_pdf = jnp.interp(fixed_z, zgrid, _probs)
            interpolated_array = interpolated_array.at[i,j].set(_interp_pdf)

    norm = 1.0
    if renormalize:
        # Integrate successively:
        ## Over models
        _int_mods = jnp.trapz(interpolated_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

        _sub_ints = jnp.split(_int_mods, n_extLaws, axis=0)
        sub_ints_ebv = []
        for sub_arr in _sub_ints:
            ## Integration over E(B-V)
            _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=0)
            sub_ints_ebv.append(_int_ebv)
        ## Over laws
        norm = jnp.trapz(jnp.array(sub_ints_ebv), x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)

    # return values array the same size as the number of based templates
    return interpolated_array / norm, norm

@partial(jit, static_argnums=(2,5,6))
def probs_at_fixed_z_oneLaw(probs_array, fixed_z, n_baseTemp, EBVs, zgrid, renormalize=True, prenormalize=False):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array = jnp.zeros((probs_array.shape[0], probs_array.shape[1]))

    # Interpolate pdf at fixed_z
    for i in range(probs_array.shape[0]):
        for j in range(probs_array.shape[1]):
            _probs = probs_array[i,j,:]
            if prenormalize:
                _prenorm = jnp.trapz(_probs, x=zgrid, axis=0)
                _probs = _probs / _prenorm
            #f_interp = j_spline(z_grid, _probs, k=2)
            #_interp_pdf = f_interp(fixed_z)
            _interp_pdf = jnp.interp(fixed_z, zgrid, _probs)
            interpolated_array = interpolated_array.at[i,j].set(_interp_pdf)

    norm = 1.0
    if renormalize:
        # Integrate successively:
        ## Over models
        _int_mods = jnp.trapz(interpolated_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(_int_mods, x=EBVs, axis=0)

    # return values array the same size as the number of based templates
    return interpolated_array / _int_ebv, _int_ebv

@partial(jit, static_argnums=(2,3,5,6))
def probs_at_fixed_z_oneEBV(probs_array, fixed_z, n_baseTemp, n_extLaws, zgrid, renormalize=True, prenormalize=False):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array = jnp.zeros((probs_array.shape[0], probs_array.shape[1]))

    # Interpolate pdf at fixed_z
    for i in range(probs_array.shape[0]):
        for j in range(probs_array.shape[1]):
            _probs = probs_array[i,j,:]
            if prenormalize:
                _prenorm = jnp.trapz(_probs, x=zgrid, axis=0)
                _probs = _probs / _prenorm
            #f_interp = j_spline(z_grid, _probs, k=2)
            #_interp_pdf = f_interp(fixed_z)
            _interp_pdf = jnp.interp(fixed_z, zgrid, _probs)
            interpolated_array = interpolated_array.at[i,j].set(_interp_pdf)

    norm = 1.0
    if renormalize:
        # Integrate successively:
        ## Over models
        _int_mods = jnp.trapz(interpolated_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

        ## Over laws
        norm = jnp.trapz(jnp.array(_int_mods), x=jnp.arange(1, 1+_int_mods.shape[0]), axis=0)

    # return values array the same size as the number of based templates
    return interpolated_array / norm, norm

@partial(jit, static_argnums=(2,4,5))
def probs_at_fixed_z_noDust(probs_array, fixed_z, n_baseTemp, zgrid, renormalize=True, prenormalize=False):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array = jnp.zeros((probs_array.shape[0], probs_array.shape[1]))

    # Interpolate pdf at fixed_z
    for i in range(probs_array.shape[0]):
        for j in range(probs_array.shape[1]):
            _probs = probs_array[i,j,:]
            if prenormalize:
                _prenorm = jnp.trapz(_probs, x=zgrid, axis=0)
                _probs = _probs / _prenorm
            #f_interp = j_spline(z_grid, _probs, k=2)
            #_interp_pdf = f_interp(fixed_z)
            _interp_pdf = jnp.interp(fixed_z, zgrid, _probs)
            interpolated_array = interpolated_array.at[i,j].set(_interp_pdf)

    norm = 1.0
    if renormalize:
        # Integrate successively:
        ## Over models
        _int_mods = jnp.trapz(interpolated_array, x=jnp.arange(1, 1+n_baseTemp), axis=0)

    # return values array the same size as the number of based templates
    return interpolated_array / _int_mods, _int_mods

@partial(jit, static_argnums=(2,3))
def p_template_at_fixed_z(probs_array, fixed_z, n_baseTemp, n_extLaws, EBVs, zgrid):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array, _norm = probs_at_fixed_z(probs_array, fixed_z, n_baseTemp, n_extLaws, EBVs, zgrid, renormalize=True)

    # Split over dust extinction laws
    _sub_ints = jnp.split(interpolated_array, n_extLaws, axis=1)
    sub_ints_ebv = []
    for sub_arr in _sub_ints:
        ## Marginalize over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=EBVs, axis=1)
        sub_ints_ebv.append(_int_ebv)

    # Marginalize over extinction law
    sub_ints_ebv_arr = jnp.array(sub_ints_ebv)
    int_laws = jnp.trapz(sub_ints_ebv_arr, x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)

    # return values array the same size as the number of based templates
    return int_laws
"""

# TBD - Working on arrays to try and use vmap for this too
"""
def evidences_in_df(df_res, results_dict, z_grid, baseTemp_arr, dust_arr):
    seds_zs = []
    laws_zs = []
    ebvs_zs = []
    odds_zs = []
    z_means = []
    z_stds = []
    z_mods = []

    #keys = [key for key in results_dict.keys()]
    for kk in tqdm(results_dict) :
        chi2arr = results_dict[kk]
        probsarr, norm = probability_distrib(chi2arr)
        while abs(1-norm)>1.0e-5 :
            chi2arr = chi2arr + 2*jnp.log(norm)
            probsarr, norm = probability_distrib(chi2arr)

        evs_nosplit = evidence(probsarr, split_laws=False)
        id_t = jnp.nonzero([t.name == df_res.loc[kk, "Template SED"] for t in baseTemp_arr])[0][0]
        id_dust = jnp.nonzero([(d.name == df_res.loc[kk, "Extinction law"]) and (d.EBV == df_res.loc[kk, "E(B-V)"]) for d in dust_arr])[0][0]

        sorted_evs_flat = jnp.argsort(evs_nosplit, axis=None)
        sorted_evs = [ jnp.unravel_index(idx, evs_nosplit.shape) for idx in sorted_evs_flat ]
        sorted_evs.reverse()
        n_temp, n_dust = sorted_evs[0]

        pz_at_ev = probsarr[n_temp, n_dust, :] / jnp.trapz(probsarr[n_temp, n_dust, :], x=z_grid)
        #cum_distr = np.cumsum(pz_at_ev)
        z_mean = jnp.trapz(z_grid*pz_at_ev, x=z_grid)
        z_std = jnp.trapz(pz_at_ev*jnp.power(z_grid-z_mean, 2), x=z_grid)
        #_selmed = cum_distr > 0.5
        #z_med = z_grid[_selmed][0]
        try:
            z_mod = z_grid[jnp.nanargmax(pz_at_ev)]
        except ValueError:
            z_mod = jnp.nan
        seds_zs.append(baseTemp_arr[n_temp].name)
        laws_zs.append(dust_arr[n_dust].name)
        ebvs_zs.append(dust_arr[n_dust].EBV)
        odds_zs.append(float(evs_nosplit[n_temp, n_dust] / evs_nosplit[id_t, id_dust]))
        z_means.append(z_mean)
        z_stds.append(z_std)
        z_mods.append(z_mod)

    df_res["Highest evidence SED"] = seds_zs
    df_res["Highest evidence dust law"] = laws_zs
    df_res["Highest evidence E(B-V)"] = ebvs_zs
    df_res["Highest evidence odd ratio"] = odds_zs
    df_res["Highest evidence z_phot (mode)"] = z_mods
    df_res["Highest evidence z_phot (mean)"] = z_means
    df_res["Highest evidence sigma(z)"] = z_stds


def investigate_at_z_spec(df_res, results_dict, baseTemp_arr, dust_arr):
    seds_zs = []
    laws_zs = []
    ebvs_zs = []
    odds_zs = []

    for kk in tqdm(results_dict) :
        chi2arr = results_dict[kk]
        probsarr, norm = probability_distrib(chi2arr)
        while abs(1-norm)>1.0e-5 :
            chi2arr = chi2arr + 2*jnp.log(norm)
            probsarr, norm = probability_distrib(chi2arr)

        evs_nosplit = evidence(probsarr, split_laws=False)
        id_t = jnp.nonzero([t.name == df_res.loc[kk, "Template SED"] for t in baseTemp_arr])[0][0]
        id_dust = jnp.nonzero([(d.name == df_res.loc[kk, "Extinction law"]) and (d.EBV == df_res.loc[kk, "E(B-V)"]) for d in dust_arr])[0][0]

        p_zfix_nosplit, _n = probs_at_fixed_z(probsarr, df_res.loc[kk, 'True redshift'], renormalize=True, prenormalize=False)
        sorted_pzfix_flat = jnp.argsort(p_zfix_nosplit, axis=None)
        sorted_pzfix = [ jnp.unravel_index(idx, p_zfix_nosplit.shape) for idx in sorted_pzfix_flat ]
        sorted_pzfix.reverse()
        n_temp, n_dust = sorted_pzfix[0]

        seds_zs.append(baseTemp_arr[n_temp].name)
        laws_zs.append(dust_arr[n_dust].name)
        ebvs_zs.append(dust_arr[n_dust].EBV)
        odds_zs.append(float(evs_nosplit[n_temp, n_dust] / evs_nosplit[id_t, id_dust]))

    df_res["Best SED at z_spec"] = seds_zs
    df_res["Best dust law at z_spec"] = laws_zs
    df_res["E(B-V) at z_spec"] = ebvs_zs
    df_res["Odd ratio"] = odds_zs
"""
