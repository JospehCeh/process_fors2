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
# Most functions are inside the package. This notebook inherits largely from `Fors2ToStellarPopSynthesis/docs/notebooks/fitters/FitFors2ManySpecLoop.ipynb` in the `fors2tostellarpopsynthesis` package.

# ## Imports and general settings
import copy
import os
import pickle
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from tqdm import tqdm

from process_fors2.analysis import convert_flux_torestframe, get_fnu, get_fnu_clean, get_gelmod
from process_fors2.fetchData import gelato_xmatch_todict

from .dsps_params import SSPParametersFit

jax.config.update("jax_enable_x64", True)

plt.style.use("default")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.labelsize"] = "x-large"
plt.rcParams["axes.titlesize"] = "x-large"
plt.rcParams["xtick.labelsize"] = "x-large"
plt.rcParams["ytick.labelsize"] = "x-large"
plt.rcParams["legend.fontsize"] = 12

kernel = kernels.RBF(0.5, (8000, 20000.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)


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


FLAG_REMOVE_GALEX = True
FLAG_REMOVE_GALEX_FUV = False
FLAG_REMOVE_VISIBLE = False

p = SSPParametersFit()
init_params = p.INIT_PARAMS
params_min = p.PARAMS_MIN
params_max = p.PARAMS_MAX


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


def fit_mags(data_dict, ssp_file=None):
    """
    Function to fit SPS on magnitudes with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    # data_dict = dict_fors2_for_fit[tag]
    # fit with all magnitudes
    from process_fors2.stellarPopSynthesis import lik_mag

    lbfgsb_mag = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=5000)
    res_m = lbfgsb_mag.run(
        init_params, bounds=(params_min, params_max), xf=data_dict["filters"], mags_measured=data_dict["mags"], sigma_mag_obs=data_dict["mags_err"], z_obs=data_dict["redshift"], ssp_file=ssp_file
    )

    """
    params_m, fun_min_m, jacob_min_m, inv_hessian_min_m = get_infos_mag(res_m,\
                                                                        lik_mag,\
                                                                        xf = data_dict["filters"],\
                                                                        mgs = data_dict["mags"],\
                                                                        mgse = data_dict["mags_err"],\
                                                                        z_obs=data_dict["redshift"])
    """

    # Convert fitted parameters into a dictionnary
    params_m = res_m.params

    # plot SFR
    # f, a = plt.subplots(1, 2)
    # plot_SFH(dict_params_m, data_dict["redshift"], subtit = data_dict["title"], ax=a[0])
    # plot_fit_ssp_spectrophotometry(dict_params_m,
    #                               data_dict["wavelengths"], data_dict["fnu"], data_dict["fnu_err"],
    #                               data_dict["filters"], data_dict["wl_mean_filters"], data_dict["mags"], data_dict["mags_err"],
    #                               data_dict["redshift"], data_dict["title"], ax=a[1])

    # save to dictionary
    dict_out = OrderedDict()
    # dict_out["fors2name"] = tag
    # dict_out["zobs"] = data_dict["redshift"]
    # dict_out["funcmin_m"] = fun_min_m

    # convert into a dictionnary
    dict_out.update({"fit_params": params_m, "zobs": data_dict["redshift"]})
    return dict_out


def fit_spec(data_dict, ssp_file=None):
    """
    Function to fit SPS on spectrum with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    from process_fors2.stellarPopSynthesis import lik_spec

    lbfgsb_spec = jaxopt.ScipyBoundedMinimize(fun=lik_spec, method="L-BFGS-B", maxiter=5000)
    res_s = lbfgsb_spec.run(
        init_params, bounds=(params_min, params_max), wls=data_dict["wavelengths"], F=data_dict["fnu"], sigma_obs=data_dict["fnu_err"], z_obs=data_dict["redshift"], ssp_file=ssp_file
    )

    # Convert fitted parameters into a dictionnary
    params_s = res_s.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_s, "zobs": data_dict["redshift"]})
    return dict_out


def fit_gelmod(data_dict, ssp_file=None):
    """
    Function to fit SPS on spectrum with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    from process_fors2.stellarPopSynthesis import lik_spec

    lbfgsb_spec = jaxopt.ScipyBoundedMinimize(fun=lik_spec, method="L-BFGS-B", maxiter=5000)
    res_s = lbfgsb_spec.run(
        init_params, bounds=(params_min, params_max), wls=data_dict["wavelengths"], F=data_dict["gelato_mod"], sigma_obs=data_dict["fnu_err"], z_obs=data_dict["redshift"], ssp_file=ssp_file
    )

    # Convert fitted parameters into a dictionnary
    params_s = res_s.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_s, "zobs": data_dict["redshift"]})
    return dict_out


def fit_rew(data_dict, ssp_file=None):
    """
    Function to fit SPS on rest equivalent widths with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    from process_fors2.stellarPopSynthesis import lik_rew

    lbfgsb_rew = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=5000)
    surechwls = jnp.arange(min(data_dict["wavelengths"]), max(data_dict["wavelengths"]) + 0.1, 0.1)
    # Removed the argument surwls from the REW likelihood to try and fix crashes.
    res_ew = lbfgsb_rew.run(
        init_params,
        bounds=(params_min, params_max),
        surwls=surechwls,
        rews_wls=data_dict["rews_wls"],
        rews=data_dict["rews"],
        rews_err=data_dict["rews_err"],
        z_obs=data_dict["redshift"],
        ssp_file=ssp_file,
    )

    # Convert fitted parameters into a dictionnary
    params_rew = res_ew.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_rew, "zobs": data_dict["redshift"]})
    return dict_out


def fit_mags_and_rew(data_dict, weight_mag=0.5, ssp_file=None):
    """
    Function to fit SPS on both observed magnitudes and rest equivalent widths with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    weight_mag : float, optional
        Weight of the fit on photometry. 1-weight_mag is affected to the fit on rest equivalent widths. Must be between 0.0 and 1.0. The default is 0.5.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    from process_fors2.stellarPopSynthesis import lik_mag_rew

    lbfgsb_comb = jaxopt.ScipyBoundedMinimize(fun=lik_mag_rew, method="L-BFGS-B", maxiter=5000)
    surechwls = jnp.arange(min(data_dict["wavelengths"]), max(data_dict["wavelengths"]) + 0.1, 0.1)
    # Removed the argument surwls from the REW likelihood to try and fix crashes.
    res_comb = lbfgsb_comb.run(
        init_params,
        bounds=(params_min, params_max),
        xf=data_dict["filters"],
        mags_measured=data_dict["mags"],
        sigma_mag_obs=data_dict["mags_err"],
        surwls=surechwls,
        rews_wls=data_dict["rews_wls"],
        rews=data_dict["rews"],
        rews_err=data_dict["rews_err"],
        z_obs=data_dict["redshift"],
        weight_mag=weight_mag,
        ssp_file=ssp_file,
    )

    # Convert fitted parameters into a dictionnary
    params_comb = res_comb.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_comb, "zobs": data_dict["redshift"]})
    return dict_out


def fit_lines(data_dict, ssp_file=None):
    """
    Function to fit SPS on spectral bands with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    from process_fors2.stellarPopSynthesis import lik_lines

    lbfgsb_lin = jaxopt.ScipyBoundedMinimize(fun=lik_lines, method="L-BFGS-B", maxiter=5000)
    res_li = lbfgsb_lin.run(
        init_params,
        bounds=(params_min, params_max),
        wls=data_dict["wavelengths"],
        refmod=data_dict["gelato_mod"],
        reflines=data_dict["gelato_lines"],
        fnuerr=data_dict["fnu_err"],
        z_obs=data_dict["redshift"],
        ssp_file=ssp_file,
    )

    # Convert fitted parameters into a dictionnary
    params_li = res_li.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_li, "zobs": data_dict["redshift"]})
    return dict_out


def filter_tags(attrs_dict, remove_visible=False, remove_galex=False, remove_galex_fuv=True):
    """
    Function to filter galaxies to fit according to their available photometry.

    Parameters
    ----------
    attrs_dict : dict
        Dictionary of attributes - must contain KiDS and GALEX photometry keywords in its keys.
    remove_visible : bool, optional
        Whether to remove galaxies with photometry in the visible range of the EM spectrum. The default is `False`.
    remove_galex : bool, optional
        Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum. The default is `False`.
    remove_galex_fuv : bool, optional
        Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum. The default is `True`.

    Returns
    -------
    list
        List of applicable tags after filtering, to be used as keys in the original `attrs_dict` dictionary for instance.
    """
    # ## Select applicable spectra
    filtered_tags = []
    for tag, fors2_attr in attrs_dict.items():
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


def prepare_data_dict(gelatoh5, attrs_dict, selected_tags, useclean=False, remove_visible=False, remove_galex=False, remove_galex_fuv=True):
    """
    Function to prepare data for the SPS fitting procedure (only DSPS available atm).

    Parameters
    ----------
    gelatoh5 : path or str
        Path to the HDF5 file gathering outputs from GELATO run.
    attrs_dict : dict
        Dictionary of attributes - must contain KiDS and GALEX photometry keywords in its keys.
    selected_tags : list
        List of tags to be considered - must correspond to keys of `attrs_dict`.
    useclean : bool, optional
        Whether to use the raw spectrum for each galaxy, or a smoothed version to remove features such as emission/absorption lines. The default is `False`.
    remove_visible : bool, optional
        Whether to remove galaxies with photometry in the visible range of the EM spectrum. The default is `False`.
    remove_galex : bool, optional
        Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum. The default is `False`.
    remove_galex_fuv : bool, optional
        Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum. The default is `True`.

    Returns
    -------
    dict
        Dictionary of dictionaries, of the form `{tag: {key: val, ..}, ..}` with `key` and `val` match data that will be used for the SPS fitting procedure of galaxy `tag`.
    """
    from process_fors2.stellarPopSynthesis import FilterInfo

    ps = FilterInfo()
    # ps.plot_transmissions()
    # ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
    dict_fors2_for_fit = {}
    for tag in tqdm(selected_tags):
        dict_tag = {}
        # extract most basic info
        fors2_attr = attrs_dict[tag]
        selected_spectrum_number = fors2_attr["num"]
        z_obs = fors2_attr["redshift"]
        title_spec = f"{tag} z = {z_obs:.2f}"

        dict_tag["spec ID"] = selected_spectrum_number
        dict_tag["redshift"] = z_obs
        dict_tag["title"] = title_spec

        # retrieve magnitude data
        data_mags = np.array(
            [
                fors2_attr["fuv_mag"],
                fors2_attr["nuv_mag"],
                fors2_attr["MAG_GAAP_u"],
                fors2_attr["MAG_GAAP_g"],
                fors2_attr["MAG_GAAP_r"],
                fors2_attr["MAG_GAAP_i"],
                fors2_attr["MAG_GAAP_Z"],
                fors2_attr["MAG_GAAP_Y"],
                fors2_attr["MAG_GAAP_J"],
                fors2_attr["MAG_GAAP_H"],
                fors2_attr["MAG_GAAP_Ks"],
            ]
        )
        data_magserr = np.array(
            [
                fors2_attr["fuv_magerr"],
                fors2_attr["nuv_magerr"],
                fors2_attr["MAGERR_GAAP_u"],
                fors2_attr["MAGERR_GAAP_g"],
                fors2_attr["MAGERR_GAAP_r"],
                fors2_attr["MAGERR_GAAP_i"],
                fors2_attr["MAGERR_GAAP_Z"],
                fors2_attr["MAGERR_GAAP_Y"],
                fors2_attr["MAGERR_GAAP_J"],
                fors2_attr["MAGERR_GAAP_H"],
                fors2_attr["MAGERR_GAAP_Ks"],
            ]
        )

        # ugri_mags_c = np.array([fors2_attr["MAG_GAAP_u"], fors2_attr["MAG_GAAP_g"], fors2_attr["MAG_GAAP_r"], fors2_attr["MAG_GAAP_i"]])
        # ugri_magserr_c = np.array([fors2_attr["MAGERR_GAAP_u"], fors2_attr["MAGERR_GAAP_g"], fors2_attr["MAGERR_GAAP_r"], fors2_attr["MAGERR_GAAP_i"]])

        # get the Fors2 spectrum
        if useclean:
            spec_obs = get_fnu_clean(gelatoh5, tag, zob=z_obs, nsigs=6)
            Xs = spec_obs["wl_cl"]
            Ys = spec_obs["fnu_cl"]
            EYs = spec_obs["bg"]
        else:
            spec_obs = get_fnu(gelatoh5, tag, zob=z_obs)
            Xs = spec_obs["wl"]
            Ys = spec_obs["fnu"]
            EYs = spec_obs["fnuerr"]
        # EYs_med = spec_obs['bg_med']
        # get the Gelato model
        gel_obs = get_gelmod(gelatoh5, tag, zob=z_obs)
        gemod = np.interp(Xs, gel_obs["wl"], gel_obs["mod"])
        geline = np.interp(Xs, gel_obs["wl"], gel_obs["line"])
        gessp = np.interp(Xs, gel_obs["wl"], gel_obs["ssp"])

        # convert to restframe
        Xspec_data, Yspec_data = convert_flux_torestframe(Xs, Ys, z_obs)
        EYspec_data = EYs  # * (1+z_obs)
        # EYspec_data_med = EYs_med #* (1+z_obs)

        _, gmod_data = convert_flux_torestframe(Xs, gemod, z_obs)
        _, glin_data = convert_flux_torestframe(Xs, geline, z_obs)
        _, gssp_data = convert_flux_torestframe(Xs, gessp, z_obs)

        dict_tag["wavelengths"] = Xspec_data
        dict_tag["fnu"] = Yspec_data
        dict_tag["fnu_err"] = EYspec_data

        dict_tag["gelato_mod"] = gmod_data
        dict_tag["gelato_lines"] = glin_data
        dict_tag["gelato_ssp"] = gssp_data

        # smooth the error over the spectrum
        # fit_res = gpr.fit(Xspec_data[:, None], EYspec_data)
        # EYspec_data_sm = gpr.predict(Xspec_data[:, None], return_std=False)

        # need to increase error to decrease chi2 error
        # EYspec_data_sm *= 2

        # Choose filters with mags without Nan
        NoNaN_mags = np.intersect1d(np.argwhere(~np.isnan(data_mags)).flatten(), np.argwhere(~np.isnan(data_magserr)).flatten())

        # selected indexes for filters
        index_selected_filters = NoNaN_mags

        if remove_galex:
            galex_indexes = np.array([0, 1])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)
        elif remove_galex_fuv:
            galex_indexes = np.array([0])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)

        if remove_visible:
            visible_indexes = np.array([2, 3, 4, 5, 6, 7])
            index_selected_filters = np.setdiff1d(NoNaN_mags, visible_indexes)

        # Select filters
        XF = ps.get_2lists()
        list_wls_f_sel = []
        list_trans_f_sel = []
        list_name_f_sel = []
        list_wlmean_f_sel = []

        for index in index_selected_filters:
            list_wls_f_sel.append(XF[0][index])
            list_trans_f_sel.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_f_sel.append(the_wlmean)
            list_name_f_sel.append(ps.filters_namelist[index])

        list_wlmean_f_sel = jnp.array(list_wlmean_f_sel)
        Xf_sel = (list_wls_f_sel, list_trans_f_sel)

        """
        NoNan_ugri = np.intersect1d(NoNaN_mags, np.array([2, 3, 4, 5]))
        list_wls_ugri = []
        list_trans_ugri = []
        list_name_ugri = []
        list_wlmean_ugri = []

        for index in NoNan_ugri:
            list_wls_ugri.append(XF[0][index])
            list_trans_ugri.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_ugri.append(the_wlmean)
            list_name_ugri.append(ps.filters_namelist[index])

        list_wlmean_ugri = jnp.array(list_wlmean_ugri)
        Xf_ugri = (list_wls_ugri, list_trans_ugri)
        # print(NoNan_ugri, list_name_ugri)
        """

        # get the magnitudes and magnitude errors
        data_selected_mags = jnp.array(data_mags[index_selected_filters])
        data_selected_magserr = jnp.array(data_magserr[index_selected_filters])
        # data_selected_ugri_corr = jnp.array(ugri_mags_c)
        # data_selected_ugri_correrr = jnp.array(ugri_magserr_c)

        dict_tag["filters"] = Xf_sel
        dict_tag["wl_mean_filters"] = list_wlmean_f_sel
        dict_tag["mags"] = data_selected_mags
        dict_tag["mags_err"] = data_selected_magserr
        # dict_tag["ugri_filters"] = Xf_ugri
        # dict_tag["wl_mean_ugri"] = list_wlmean_ugri
        # dict_tag["ugri_corr"] = data_selected_ugri_corr
        # dict_tag["ugri_corr_err"] = data_selected_ugri_correrr

        lines_list = np.unique(sorted([fl.split("_REW")[:-1] for fl in fors2_attr if "REW" in fl]))
        lines_wls = jnp.array([float(li.split("_")[-1]) for li in lines_list])
        # lines_flux = jnp.array([fors2_attr[f"{attr}_Flux"] for attr in lines_list])
        # lines_fluxerr = jnp.array([fors2_attr[f"{attr}_Flux_err"] for attr in lines_list])
        # lines_ramp = jnp.array([fors2_attr[f"{attr}_RAmp"] for attr in lines_list])
        # lines_ramperr = jnp.array([fors2_attr[f"{attr}_RAmp_err"] for attr in lines_list])
        lines_rew = jnp.array([fors2_attr[f"{attr}_REW"] for attr in lines_list])
        lines_rewerr = jnp.array([fors2_attr[f"{attr}_REW_err"] for attr in lines_list])

        selew = jnp.logical_and(jnp.isfinite(lines_rew), lines_rewerr > 1.0e-4)

        dict_tag["rews_wls"] = lines_wls[selew]
        dict_tag["rews"] = lines_rew[selew]
        dict_tag["rews_err"] = lines_rewerr[selew]

        dict_fors2_for_fit[tag] = dict_tag
    return dict_fors2_for_fit


def prepare_bootstrap_dict(gelatoh5, attrs_dict, selected_tag, n_fits=10, bs_type="mags", remove_visible=False, remove_galex=False, remove_galex_fuv=True):
    """
    Function to prepare data for the SPS fitting procedure (only DSPS available atm).

    Parameters
    ----------
    gelatoh5 : path or str
        Path to the HDF5 file gathering outputs from GELATO run.
    attrs_dict : dict
        Dictionary of attributes - must contain KiDS and GALEX photometry keywords in its keys.
    selected_tag : str
        Identifier (tag) of the galaxy to perform several fits on. For FORS2 data, it is of the shape `'SPECnnn'` where `nnn` is an integer.
    n_fits : int, optional
        Number of bootstrap samples to draw. The default is 10.
    bs_type : str, optional
        Data from which to draw random samples from a distribution (mean+std dev). Can be any combination of 'mags' for magnitudes and 'rews' for Restframe Equivalent Widths. The default is 'mags'.
    remove_visible : bool, optional
        Whether to remove galaxies with photometry in the visible range of the EM spectrum. The default is `False`.
    remove_galex : bool, optional
        Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum. The default is `False`.
    remove_galex_fuv : bool, optional
        Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum. The default is `True`.

    Returns
    -------
    dict
        Dictionary of dictionaries, of the form `{tag_N: {key: val, ..}, ..}` with `key` and `val` match data that will be used for the SPS fitting procedure of galaxy `tag` and bootstrap draw `N`.
    """
    from process_fors2.stellarPopSynthesis import FilterInfo

    ps = FilterInfo()
    # ps.plot_transmissions()
    # ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
    dict_fors2_for_fit = {}
    for n_bs in range(n_fits):
        tag = f"{selected_tag}_{n_bs}"
        dict_tag = {}
        # extract most basic info
        fors2_attr = attrs_dict[selected_tag]
        selected_spectrum_number = fors2_attr["num"]
        z_obs = fors2_attr["redshift"]
        title_spec = f"{tag} z = {z_obs:.2f}"

        dict_tag["spec ID"] = selected_spectrum_number
        dict_tag["redshift"] = z_obs
        dict_tag["title"] = title_spec

        # retrieve magnitude data
        data_mags = np.array(
            [
                fors2_attr["fuv_mag"],
                fors2_attr["nuv_mag"],
                fors2_attr["MAG_GAAP_u"],
                fors2_attr["MAG_GAAP_g"],
                fors2_attr["MAG_GAAP_r"],
                fors2_attr["MAG_GAAP_i"],
                fors2_attr["MAG_GAAP_Z"],
                fors2_attr["MAG_GAAP_Y"],
                fors2_attr["MAG_GAAP_J"],
                fors2_attr["MAG_GAAP_H"],
                fors2_attr["MAG_GAAP_Ks"],
            ]
        )
        data_magserr = np.array(
            [
                fors2_attr["fuv_magerr"],
                fors2_attr["nuv_magerr"],
                fors2_attr["MAGERR_GAAP_u"],
                fors2_attr["MAGERR_GAAP_g"],
                fors2_attr["MAGERR_GAAP_r"],
                fors2_attr["MAGERR_GAAP_i"],
                fors2_attr["MAGERR_GAAP_Z"],
                fors2_attr["MAGERR_GAAP_Y"],
                fors2_attr["MAGERR_GAAP_J"],
                fors2_attr["MAGERR_GAAP_H"],
                fors2_attr["MAGERR_GAAP_Ks"],
            ]
        )

        # get the Fors2 spectrum
        spec_obs = get_fnu(gelatoh5, selected_tag, zob=z_obs)
        Xs = spec_obs["wl"]
        Ys = spec_obs["fnu"]
        EYs = spec_obs["fnuerr"]
        # EYs_med = spec_obs['bg_med']

        # get the Gelato model
        gel_obs = get_gelmod(gelatoh5, selected_tag, zob=z_obs)
        gemod = np.interp(Xs, gel_obs["wl"], gel_obs["mod"])
        geline = np.interp(Xs, gel_obs["wl"], gel_obs["line"])
        gessp = np.interp(Xs, gel_obs["wl"], gel_obs["ssp"])

        # convert to restframe
        Xspec_data, Yspec_data = convert_flux_torestframe(Xs, Ys, z_obs)
        EYspec_data = EYs  # * (1+z_obs)
        # EYspec_data_med = EYs_med #* (1+z_obs)

        _, gmod_data = convert_flux_torestframe(Xs, gemod, z_obs)
        _, glin_data = convert_flux_torestframe(Xs, geline, z_obs)
        _, gssp_data = convert_flux_torestframe(Xs, gessp, z_obs)

        dict_tag["wavelengths"] = Xspec_data
        dict_tag["fnu"] = Yspec_data
        dict_tag["fnu_err"] = EYspec_data

        dict_tag["gelato_mod"] = gmod_data
        dict_tag["gelato_lines"] = glin_data
        dict_tag["gelato_ssp"] = gssp_data

        # Choose filters with mags without Nan
        NoNaN_mags = np.intersect1d(np.argwhere(~np.isnan(data_mags)).flatten(), np.argwhere(~np.isnan(data_magserr)).flatten())

        # selected indexes for filters
        index_selected_filters = NoNaN_mags

        if remove_galex:
            galex_indexes = np.array([0, 1])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)
        elif remove_galex_fuv:
            galex_indexes = np.array([0])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)

        if remove_visible:
            visible_indexes = np.array([2, 3, 4, 5, 6, 7])
            index_selected_filters = np.setdiff1d(NoNaN_mags, visible_indexes)

        # Select filters
        XF = ps.get_2lists()
        list_wls_f_sel = []
        list_trans_f_sel = []
        list_name_f_sel = []
        list_wlmean_f_sel = []

        for index in index_selected_filters:
            list_wls_f_sel.append(XF[0][index])
            list_trans_f_sel.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_f_sel.append(the_wlmean)
            list_name_f_sel.append(ps.filters_namelist[index])

        list_wlmean_f_sel = jnp.array(list_wlmean_f_sel)
        Xf_sel = (list_wls_f_sel, list_trans_f_sel)

        # get the magnitudes and magnitude errors
        data_selected_mags = jnp.array(data_mags[index_selected_filters])
        data_selected_magserr = jnp.array(data_magserr[index_selected_filters])

        dict_tag["filters"] = Xf_sel
        dict_tag["wl_mean_filters"] = list_wlmean_f_sel
        dict_tag["mags"] = np.random.normal(loc=data_selected_mags, scale=data_selected_magserr) if "mag" in bs_type.lower() else data_selected_mags
        dict_tag["mags_err"] = data_selected_magserr

        lines_list = np.unique(sorted([fl.split("_REW")[:-1] for fl in fors2_attr if "REW" in fl]))
        lines_wls = jnp.array([float(li.split("_")[-1]) for li in lines_list])
        lines_rew = jnp.array([fors2_attr[f"{attr}_REW"] for attr in lines_list])
        lines_rewerr = jnp.array([fors2_attr[f"{attr}_REW_err"] for attr in lines_list])

        selew = jnp.logical_and(jnp.isfinite(lines_rew), lines_rewerr > 1.0e-4)

        dict_tag["rews_wls"] = lines_wls[selew]
        dict_tag["rews"] = np.random.normal(loc=lines_rew[selew], scale=lines_rewerr[selew]) if "rew" in bs_type.lower() else lines_rew[selew]
        dict_tag["rews_err"] = lines_rewerr[selew]

        dict_fors2_for_fit[tag] = dict_tag
    return dict_fors2_for_fit


def fit_loop(
    xmatch_h5, gelato_h5, fit_type="mags", use_clean=False, low_bound=0, high_bound=None, ssp_file=None, weight_mag=0.5, remove_visible=False, remove_galex=False, remove_galex_fuv=True, quiet=False
):
    """
    Function to fit a stellar population onto observations of galaxies.

    Parameters
    ----------
    xmatch_h5 : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    gelato_h5 : path or str
        Path to the HDF5 file gathering outputs from GELATO run.
    fit_type : str, optional
        Data to fit the SPS on. Must be one of :
            - 'mags' to fit on KiDS+VIKING+GALEX photometry
            - 'spec' to fit on spectral density of flux
            - 'lines' to fit on spectral emission/absorption lines (_i.e._ the spectral density of flux after removal of the continuum as detected by GELATO)
            - 'gelato' to fit on the model output from GELATO (incl. SSP, lines and power law continuum) directly instead of the raw spectrum (*e.g.* FORS2)
            - 'rews' to fit on Restframe Equivalent Widths of spectral emission/absorption lines as detected and computed by GELATO.
            - 'mags+rews' to fit on both magnitudes and Restframe Equivalent Widths. The weight associated to each likelihood can be controlled with the optional parameter `weight_mag`.
        The default is 'mags'.
    use_clean : bool, optional
        If fitting on spectra, whether to use the raw spectrum for each galaxy, or a smoothed version to remove features such as emission/absorption lines. The default is `False`.
    low_bound : int, optional
        If fitting a slice of the original data : the index of the first element (natural count : starts at 1, ends at nb of elements). The default is 0.
    high_bound : int or None, optional
        If fitting a slice of the original data : the index of the last element (natural count : starts at 1, ends at nb of elements).
        If None, all galaxies are fitted satrting with `low_bound`. The default is None.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.
    weight_mag : float, optional
        Weight of the fit on photometry. 1-weight_mag is affected to the fit on rest equivalent widths. Must be between 0.0 and 1.0. The default is 0.5.
    remove_visible : bool, optional
        Whether to remove galaxies with photometry in the visible range of the EM spectrum. The default is `False`.
    remove_galex : bool, optional
        Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum. The default is `False`.
    remove_galex_fuv : bool, optional
        Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum. The default is `True`.
    quiet : bool, optional
        Whether to silence some prints (for convenience while running in loops for instance). The default is False.

    Returns
    -------
    dict
        Dictionary of dictionaries, of the form `{tag: {key: val, ..}, ..}` where `key` and `val` match data that were used for the SPS fitting procedure of galaxy `tag`.
    dict
        Dictionary of dictionaries, of the form `{tag: {key: val, ..}, ..}` where `key` and `val` match data that were produced by the fitting procedure and can be used to synthetise an SED with DSPS.
    int
        The effective lower boundary used during the fit.
    int
        The effective higher boundary used during the fit.
    """
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs = gelato_xmatch_todict(gelatoh5, xmatchh5)

    # ## Select applicable spectra
    filtered_tags = filter_tags(merged_attrs, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)

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

    # ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
    dict_fors2_for_fit = prepare_data_dict(gelatoh5, merged_attrs, selected_tags, useclean=use_clean, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv)

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "line" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on spectral lines... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_lines(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "spec" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed spectra... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_spec(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "gelato" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on GELATO models... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_gelmod(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "mag" in fit_type.lower() and "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_mags_and_rew(dico, weight_mag, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "rew" in fit_type.lower():
        if not quiet:
            print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_rew(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
    else:
        if not quiet:
            print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
        fit_results_dict = jax.tree_map(lambda dico: fit_mags(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)

    return dict_fors2_for_fit, fit_results_dict, low_bound, high_bound


def fit_bootstrap(
    xmatch_h5, gelato_h5, specID, fit_type="mags", n_fits=10, bs_type="mags", ssp_file=None, weight_mag=0.5, remove_visible=False, remove_galex=False, remove_galex_fuv=True, quiet=False
):
    """
    Function to fit a stellar population onto observations of galaxies.

    Parameters
    ----------
    xmatch_h5 : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    gelato_h5 : path or str
        Path to the HDF5 file gathering outputs from GELATO run.
    specID : str
        Identifier (tag) of the galaxy to perform several fits on. For FORS2 data, it is of the shape `'SPECnnn'` where `nnn` is an integer.
    fit_type : str, optional
        Data to fit the SPS on. Must be one of :
            - 'mags' to fit on KiDS+VIKING+GALEX photometry
            - 'rews' to fit on Restframe Equivalent Widths of spectral emission/absorption lines as detected and computed by GELATO.
            - 'mags+rews' to fit on both magnitudes and Restframe Equivalent Widths. The weight associated to each likelihood can be controlled with the optional parameter `weight_mag`.
        The default is 'mags'.
    n_fits : int, optional
        Number of bootstrap samples to draw. The default is 10.
    bs_type : str, optional
        Data from which to draw random samples from a distribution (mean+std dev). Can be any combination of 'mags' for magnitudes and 'rews' for Restframe Equivalent Widths. The default is 'mags'.
    ssp_file : path or str, optional
        SSP library location. If None, loads the defaults file from `process_fors2.fetchData`. The default is None.
    weight_mag : float, optional
        Weight of the fit on photometry. 1-weight_mag is affected to the fit on rest equivalent widths. Must be between 0.0 and 1.0. The default is 0.5.
    remove_visible : bool, optional
        Whether to remove galaxies with photometry in the visible range of the EM spectrum. The default is `False`.
    remove_galex : bool, optional
        Whether to remove galaxies with photometry in the ultraviolet (near and far) range of the EM spectrum. The default is `False`.
    remove_galex_fuv : bool, optional
        Whether to remove galaxies with photometry in the far ultraviolet range (only) of the EM spectrum. The default is `True`.
    quiet : bool, optional
        Whether to silence some prints (for convenience while running in loops for instance). The default is False.

    Returns
    -------
    dict
        Dictionary of dictionaries, of the form `{tag: {key: val, ..}, ..}` where `key` and `val` match data that were used for the SPS fitting procedure of galaxy `tag`.
    dict
        Dictionary of dictionaries, of the form `{tag: {key: val, ..}, ..}` where `key` and `val` match data that were produced by the fitting procedure and can be used to synthetise an SED with DSPS.
    """
    xmatchh5 = os.path.abspath(xmatch_h5)
    gelatoh5 = os.path.abspath(gelato_h5)
    merged_attrs = gelato_xmatch_todict(gelatoh5, xmatchh5)

    # ## Select applicable spectra
    # filtered_tags = filter_tags(merged_attrs, remove_galex=FLAG_REMOVE_GALEX, remove_galex_fuv=FLAG_REMOVE_GALEX_FUV, remove_visible=FLAG_REMOVE_VISIBLE)

    try:
        if not quiet:
            print(f"Performing {n_fits} fits of galaxy {specID} with bootstrapped {fit_type}.")

        # ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
        dict_fors2_for_fit = prepare_bootstrap_dict(
            gelatoh5, merged_attrs, specID, n_fits=n_fits, bs_type=bs_type, remove_visible=remove_visible, remove_galex=remove_galex, remove_galex_fuv=remove_galex_fuv
        )

        # fit loop
        # for tag in tqdm(dict_fors2_for_fit):
        if "mag" in fit_type.lower() and "rew" in fit_type.lower():
            if not quiet:
                print("Fitting SPS on observed magnitudes and restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
            fit_results_dict = jax.tree_map(lambda dico: fit_mags_and_rew(dico, weight_mag, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
        elif "rew" in fit_type.lower():
            if not quiet:
                print("Fitting SPS on restframe equivalent widths... it may take (more than) a few minutes, please be patient.")
            fit_results_dict = jax.tree_map(lambda dico: fit_rew(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)
        else:
            if not quiet:
                print("Fitting SPS on observed magnitudes... it may take (more than) a few minutes, please be patient.")
            fit_results_dict = jax.tree_map(lambda dico: fit_mags(dico, ssp_file), dict_fors2_for_fit, is_leaf=has_redshift)

        return dict_fors2_for_fit, fit_results_dict
    except IndexError:
        print(f"Specified tag ({specID}) not found in database.")
        return None


def make_fit_plots(dict_for_fit, results_dict, outdir, fitname=None, start=None, end=None):
    """
    Function to make plots of the fitting procedure outputs and gather them in a PDF file.

    Parameters
    ----------
    dict_for_fit : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    results_dict : dict
        Path to the HDF5 file gathering outputs from GELATO run.
    outdir : path or str
        Path to the directory where to write the PDF file.
    fitname : str, optional
        Name of the fitting procedure (_e.g._ the `fit_type` in `fit_loop`), to be included in the PDF title. If None, it will be sort of guessed from `outdir`. The default is None.
    start : int or None, optional
        The effective start index of the fitted slice, to be included in the PDF title. If None, will be set to 1. The default is None.
    end : int or None, optional
        The effective end index of the fitted slice, to be included in the PDF title. If None, will be set to the length of the list of figures. The default is None.

    Returns
    -------
    None
    """
    from process_fors2.stellarPopSynthesis import paramslist_to_dict, plot_fit_ssp_spectrophotometry, plot_SFH

    plt.style.use("default")
    # parameters for fit
    list_of_figs = []
    outdir = os.path.abspath(outdir)

    for tag, fit_dict in results_dict.items():
        dict_params_fit = paramslist_to_dict(fit_dict["fit_params"], p.PARAM_NAMES_FLAT)
        data_dict = dict_for_fit[tag]

        # plot SFR
        f, a = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
        plot_SFH(dict_params_fit, data_dict["redshift"], subtit=data_dict["title"], ax=a[0])
        plot_fit_ssp_spectrophotometry(
            dict_params_fit,
            data_dict["wavelengths"],
            data_dict["fnu"],
            data_dict["fnu_err"],
            data_dict["filters"],
            data_dict["wl_mean_filters"],
            data_dict["mags"],
            data_dict["mags_err"],
            data_dict["redshift"],
            data_dict["title"],
            ax=a[1],
        )

        # save figures and parameters
        list_of_figs.append(copy.deepcopy(f))
    if fitname is None:
        fitname = outdir.split("_")[-1]
    if fitname[-1] == "/":
        fitname = fitname[:-1]

    f.suptitle(f"Fit method : {fitname}")
    if start is None:
        start = 0
    if end is None:
        end = len(list_of_figs)

    pdfoutputfilename = os.path.join(outdir, f"fitparams_{fitname}_{start+1}_to_{end}.pdf")
    plot_figs_to_PDF(pdfoutputfilename, list_of_figs)


def make_bootstrap_plot(dict_for_fit, results_dict, outdir, fitname=None):
    """
    Function to make plots of the bootstrap-fitting procedure outputs and gather them in a PDF file.

    Parameters
    ----------
    dict_for_fit : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    results_dict : dict
        Path to the HDF5 file gathering outputs from GELATO run.
    outdir : path or str
        Path to the directory where to write the PDF file.
    fitname : str, optional
        Name of the fitting procedure (_e.g._ the `fit_type` in `fit_loop`), to be included in the PDF title. If None, it will be sort of guessed from `outdir`. The default is None.

    Returns
    -------
    None
    """
    from process_fors2.stellarPopSynthesis import plot_bootstrap_ssp_spectrophotometry, plot_SFH_bootstrap

    plt.style.use("default")
    # parameters for fit
    # list_of_figs = []
    outdir = os.path.abspath(outdir)

    f, a = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    plot_SFH_bootstrap(dict_for_fit, results_dict, p.PARAM_NAMES_FLAT, ax=a[0])
    plot_bootstrap_ssp_spectrophotometry(dict_for_fit, results_dict, p.PARAM_NAMES_FLAT, ax=a[1])

    # save figures and parameters
    # list_of_figs.append(copy.deepcopy(f))
    if fitname is None:
        fitname = outdir.split("_")[-1]
    if fitname[-1] == "/":
        fitname = fitname[:-1]

    f.suptitle(f"Fit method : {fitname}")
    keylist = list(results_dict.keys())
    specn = keylist[0].split("_")[0]
    f.savefig(os.path.join(outdir, f"bootstrap_plot_{specn}_{fitname}.png"), pad_inches="layout")
    # pdfoutputfilename = os.path.join(outdir, f"bootstrap_plot_{specn}_{fitname}.pdf")
    # plot_figs_to_PDF(pdfoutputfilename, list_of_figs)


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
    _ssp_file = None if inputs["ssp_file"].lower() == "default" else os.path.abspath(inputs["ssp_file"])

    if inputs["bootstrap"]:
        dict_fors2_for_fit, fit_results_dict = fit_bootstrap(
            xmatchh5,
            gelatoh5,
            inputs["bootstrap_id"],
            n_fits=inputs["number_bootstrap"],
            bs_type=inputs["bootstrap_type"],
            fit_type=_fit_type,
            ssp_file=_ssp_file,
            weight_mag=_weight_mag,
            remove_visible=inputs["remove_visible"],
            remove_galex=inputs["remove_galex"],
            remove_galex_fuv=inputs["remove_fuv"],
        )

        fitname = _fit_type

        outdir = os.path.abspath("./DSPS_pickles_bootstraps")
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        make_bootstrap_plot(dict_fors2_for_fit, fit_results_dict, outdir, fitname=fitname)

        filename_params = os.path.join(outdir, f"bootstrap_params_{inputs['bootstrap_id']}_{fitname}.pickle")
        with open(filename_params, "wb") as outf:
            pickle.dump(fit_results_dict, outf)
    else:
        _useclean = inputs["use_clean"]  # Only for fit on spectra
        _low = inputs["first_spec"]
        _high = None if inputs["last_spec"] < 0 else inputs["last_spec"]
        dict_fors2_for_fit, fit_results_dict, low_bound, high_bound = fit_loop(
            xmatchh5,
            gelatoh5,
            fit_type=_fit_type,
            use_clean=("spec" in _fit_type.lower() and _useclean),
            low_bound=_low,
            high_bound=_high,
            ssp_file=_ssp_file,
            weight_mag=_weight_mag,
            remove_visible=inputs["remove_visible"],
            remove_galex=inputs["remove_galex"],
            remove_galex_fuv=inputs["remove_fuv"],
        )

        fitname = _fit_type
        if "spec" in _fit_type.lower():
            fitname = f"{fitname}_clean" if _useclean else f"{fitname}_raw"

        outdir = os.path.abspath(f"./DSPS_pickles_fit_{fitname}")
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        make_fit_plots(dict_fors2_for_fit, fit_results_dict, outdir, fitname=fitname, start=low_bound, end=high_bound)

        filename_params = os.path.join(outdir, f"fitparams_{fitname}_{low_bound+1}_to_{high_bound}.pickle")
        with open(filename_params, "wb") as outf:
            pickle.dump(fit_results_dict, outf)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))


#######################################
## BACKUP FOR MORE FITTING PROCESSES ##
#######################################
"""
#load starlight spectrum
dict_sl = sl.getspectrum_fromgroup(tag)

# rescale starlight spectrum
w_sl, fnu_sl, _ = rescale_starlight_inrangefors2(dict_sl["wl"],\
                                                 dict_sl["fnu"],\
                                                 Xspec_data_rest,\
                                                 Yspec_data_rest)

# plot all final data + starlight
plot_fit_ssp_spectrophotometry_sl(dict_params_mm,\
                                  Xspec_data_rest,\
                                  Yspec_data_rest,\
                                  EYspec_data_rest,\
                                  data_dict["filters"],\
                                  xphot_rest,\
                                  yphot_rest,\
                                  eyphot_rest,\
                                  w_sl,\
                                  fnu_sl,\
                                  data_dict["redshift"],\
                                  data_dict["title"],\
                                  ax=axs[1])
"""
"""
# combining spectro and photometry
Xc = [Xspec_data_rest, Xf_sel]
Yc = [Yspec_data_rest, data_selected_mags]
EYc = [EYspec_data_rest, data_selected_magserr]
weight_spec = 0.5
Ns = len(Yspec_data_rest)
Nm = len(data_selected_mags)
Nc = Ns+Nm

# do the combined fit
lbfgsb = jaxopt.ScipyBoundedMinimize(fun = lik_comb, method = "L-BFGS-B")
res_c = lbfgsb.run(init_params,\
                   bounds = (params_min, params_max),\
                   xc = Xc,\
                   datac = Yc,\
                   sigmac = EYc,\
                   z_obs = z_obs,\
                   weight = weight_spec)
params_c, fun_min_c, jacob_min_c, inv_hessian_min_c = get_infos_comb(res_c,\
                                                                     lik_comb,\
                                                                     xc = Xc,\
                                                                     datac = Yc,\
                                                                     sigmac = EYc,\
                                                                     z_obs = z_obs,\
                                                                     weight = weight_spec)
params_cm, fun_min_cm, jacob_min_cm, inv_hessian_min_cm  = get_infos_mag(res_c,\
                                                                         lik_mag,\
                                                                         xf = Xf_sel,\
                                                                         mgs = data_selected_mags,\
                                                                         mgse = data_selected_magserr,\
                                                                         z_obs = z_obs)
params_cs, fun_min_cs, jacob_min_cs, inv_hessian_min_cs = get_infos_spec(res_c,\
                                                                         lik_spec,\
                                                                         wls = Xspec_data_rest,\
                                                                         F = Yspec_data_rest,\
                                                                         eF = EYspec_data_rest,\
                                                                         z_obs = z_obs)
print("params_c:", params_c, "\nfun@min:", fun_min_c, "\njacob@min:", jacob_min_c) #,"\n invH@min:",inv_hessian_min_c)
print("params_cm:", params_cm, "\nfun@min:", fun_min_cm, "\njacob@min:", jacob_min_cm)
print("params_cs:", params_cs, "\nfun@min:", fun_min_cs, "\njacob@min:", jacob_min_cs)

#save to dictionary
dict_out = OrderedDict()
dict_out["fors2name"] = tag
dict_out["zobs"] = z_obs
dict_out["Nc"] = Nc
dict_out["Ns"] = Ns
dict_out["Nm"] = Nm
dict_out["funcmin_c"] = fun_min_c
dict_out["funcmin_m"] = fun_min_cm
dict_out["funcmin_s"] = fun_min_cs

# convert into a dictionnary
dict_params_c = paramslist_to_dict(params_c, p.PARAM_NAMES_FLAT)
dict_out.update(dict_params_c)

# plot the combined fit
plot_fit_ssp_spectrophotometry(dict_params_c,\
                               Xspec_data_rest,\
                               Yspec_data_rest,\
                               EYspec_data_rest,\
                               data_dict["filters"],\
                               xphot_rest,\
                               yphot_rest,\
                               eyphot_rest,\
                               z_obs = z_obs,\
                               subtit = title_spec )
"""
