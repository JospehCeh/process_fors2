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
from process_fors2.stellarPopSynthesis import FilterInfo, SSPParametersFit, lik_lines, lik_mag, lik_rew, lik_spec, paramslist_to_dict, plot_fit_ssp_spectrophotometry, plot_SFH

jax.config.update("jax_enable_x64", True)

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


def fit_mags(data_dict):
    """
    Function to fit SPS on magnitudes with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    # data_dict = dict_fors2_for_fit[tag]
    # fit with all magnitudes
    lbfgsb_mag = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B", maxiter=5000)
    res_m = lbfgsb_mag.run(init_params, bounds=(params_min, params_max), xf=data_dict["filters"], mags_measured=data_dict["mags"], sigma_mag_obs=data_dict["mags_err"], z_obs=data_dict["redshift"])

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


def fit_spec(data_dict):
    """
    Function to fit SPS on spectrum with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    lbfgsb_spec = jaxopt.ScipyBoundedMinimize(fun=lik_spec, method="L-BFGS-B", maxiter=5000)
    res_s = lbfgsb_spec.run(init_params, bounds=(params_min, params_max), wls=data_dict["wavelengths"], F=data_dict["fnu"], sigma_obs=data_dict["fnu_err"], z_obs=data_dict["redshift"])

    # Convert fitted parameters into a dictionnary
    params_s = res_s.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_s, "zobs": data_dict["redshift"]})
    return dict_out


def fit_rew(data_dict):
    """
    Function to fit SPS on rest equivalent widths with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    lbfgsb_rew = jaxopt.ScipyBoundedMinimize(fun=lik_rew, method="L-BFGS-B", maxiter=5000)
    # surechwls = jnp.arange(min(data_dict["wavelengths"]), max(data_dict["wavelengths"]) + 0.1, 0.1)
    # Removed the argument surwls from the REW likelihood to try and fix crashes.
    res_ew = lbfgsb_rew.run(init_params, bounds=(params_min, params_max), rews_wls=data_dict["rews_wls"], rews=data_dict["rews"], rews_err=data_dict["rews_err"], z_obs=data_dict["redshift"])

    # Convert fitted parameters into a dictionnary
    params_rew = res_ew.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_rew, "zobs": data_dict["redshift"]})
    return dict_out


def fit_lines(data_dict):
    """
    Function to fit SPS on spectral bands with DSPS.

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with properties (filters, photometry and redshift) of an individual galaxy - *i.e.* a leaf of the global dictionary (tree).

    Returns
    -------
    dictionary
        Dictionary containing all fitted SPS parameters, from which one can synthesize the SFH and the correponding SED with DSPS.
    """
    lbfgsb_lin = jaxopt.ScipyBoundedMinimize(fun=lik_lines, method="L-BFGS-B", maxiter=5000)
    res_li = lbfgsb_lin.run(
        init_params,
        bounds=(params_min, params_max),
        wls=data_dict["wavelengths"],
        refmod=data_dict["gelato_mod"],
        reflines=data_dict["gelato_lines"],
        fnuerr=data_dict["fnu_err"],
        z_obs=data_dict["redshift"],
    )

    # Convert fitted parameters into a dictionnary
    params_li = res_li.params
    # save to dictionary
    dict_out = OrderedDict()

    # convert into a dictionnary
    dict_out.update({"fit_params": params_li, "zobs": data_dict["redshift"]})
    return dict_out


def main(args):
    """
    Function that goes through the whole fitting process, callable from outside.

    Parameters
    ----------
    args : list, tuple or array
        Arguments to be passed to the function as command line arguments.
        Mandatory arguments are 1- path to the HDF5 file of cross-matched data, 2- path to the HDF5 file of GELATO outputs, 3- the type of fit ('mags', 'spec', 'rews' or 'lines') and
        4- whether to clean the spectrum before the fit ('raw' or 'clean' - only affects fitting on spectra).
        Optional arguments are 5- (resp. 6-) the index of the first (resp. last) galaxy to fit (starts at one). If not specified, all galaxies will be fitted. This may cause crashes.

    Returns
    -------
    int
        0 if exited correctly.
    """

    ps = FilterInfo()
    ps.plot_transmissions()

    xmatchh5 = os.path.abspath(args[1])
    gelatoh5 = os.path.abspath(args[2])
    merged_attrs = gelato_xmatch_todict(gelatoh5, xmatchh5)

    # ## Select applicable spectra
    filtered_tags = []
    for tag, fors2_attr in merged_attrs.items():
        bool_viz = FLAG_REMOVE_VISIBLE or (
            not (FLAG_REMOVE_VISIBLE)
            and np.isfinite(fors2_attr["MAG_GAAP_u"])
            and np.isfinite(fors2_attr["MAG_GAAP_g"])
            and np.isfinite(fors2_attr["MAG_GAAP_r"])
            and np.isfinite(fors2_attr["MAG_GAAP_i"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_u"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_g"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_r"])
            and np.isfinite(fors2_attr["MAGERR_GAAP_i"])
        )

        bool_fuv = (FLAG_REMOVE_GALEX or FLAG_REMOVE_GALEX_FUV) or (not (FLAG_REMOVE_GALEX or FLAG_REMOVE_GALEX_FUV) and np.isfinite(fors2_attr["fuv_mag"]) and np.isfinite(fors2_attr["fuv_magerr"]))

        bool_nuv = FLAG_REMOVE_GALEX or (not (FLAG_REMOVE_GALEX) and np.isfinite(fors2_attr["nuv_mag"]) and np.isfinite(fors2_attr["nuv_magerr"]))

        if bool_viz and bool_fuv and bool_nuv:
            filtered_tags.append(tag)
    print(f"Number of galaxies in the sample : {len(filtered_tags)}.")

    if len(args) < 7:
        low_bound, high_bound = 0, len(filtered_tags)
    else:
        low_bound, high_bound = int(args[5]), int(args[6])

    use_clean = True
    if "spec" in args[3].lower() and "raw" in args[4].lower():
        use_clean = False

    low_bound = max(0, low_bound - 1)
    high_bound = min(high_bound, len(filtered_tags))
    low_bound = min(low_bound, high_bound - 1)

    selected_tags = filtered_tags[low_bound:high_bound]

    print(f"Number of galaxies to be fitted : {len(selected_tags)}.")
    start_tag, end_tag = selected_tags[0], selected_tags[-1]

    # ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
    dict_fors2_for_fit = {}
    for tag in tqdm(selected_tags):
        dict_tag = {}
        # extract most basic info
        fors2_attr = merged_attrs[tag]
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

        ugri_mags_c = np.array([fors2_attr["MAG_GAAP_u"], fors2_attr["MAG_GAAP_g"], fors2_attr["MAG_GAAP_r"], fors2_attr["MAG_GAAP_i"]])
        ugri_magserr_c = np.array([fors2_attr["MAGERR_GAAP_u"], fors2_attr["MAGERR_GAAP_g"], fors2_attr["MAGERR_GAAP_r"], fors2_attr["MAGERR_GAAP_i"]])

        # get the Fors2 spectrum
        if use_clean:
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

        if FLAG_REMOVE_GALEX:
            galex_indexes = np.array([0, 1])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)
        elif FLAG_REMOVE_GALEX_FUV:
            galex_indexes = np.array([0])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)

        if FLAG_REMOVE_VISIBLE:
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

        # get the magnitudes and magnitude errors
        data_selected_mags = jnp.array(data_mags[index_selected_filters])
        data_selected_magserr = jnp.array(data_magserr[index_selected_filters])
        data_selected_ugri_corr = jnp.array(ugri_mags_c)
        data_selected_ugri_correrr = jnp.array(ugri_magserr_c)

        dict_tag["filters"] = Xf_sel
        dict_tag["wl_mean_filters"] = list_wlmean_f_sel
        dict_tag["mags"] = data_selected_mags
        dict_tag["mags_err"] = data_selected_magserr
        dict_tag["ugri_filters"] = Xf_ugri
        dict_tag["wl_mean_ugri"] = list_wlmean_ugri
        dict_tag["ugri_corr"] = data_selected_ugri_corr
        dict_tag["ugri_corr_err"] = data_selected_ugri_correrr

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

    # parameters for fit
    list_of_figs = []

    # fit loop
    # for tag in tqdm(dict_fors2_for_fit):
    if "line" in args[3].lower():
        fit_results_dict = jax.tree_map(lambda dico: fit_lines(dico), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "rew" in args[3].lower():
        fit_results_dict = jax.tree_map(lambda dico: fit_rew(dico), dict_fors2_for_fit, is_leaf=has_redshift)
    elif "spec" in args[3].lower():
        fit_results_dict = jax.tree_map(lambda dico: fit_spec(dico), dict_fors2_for_fit, is_leaf=has_redshift)
    else:
        fit_results_dict = jax.tree_map(lambda dico: fit_mags(dico), dict_fors2_for_fit, is_leaf=has_redshift)

    for tag, fit_dict in fit_results_dict.items():
        dict_params_fit = paramslist_to_dict(fit_dict["fit_params"], p.PARAM_NAMES_FLAT)
        data_dict = dict_fors2_for_fit[tag]

        # plot SFR
        f, a = plt.subplots(1, 2)
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
    fitname = args[3]
    if "spec" in args[3].lower():
        fitname += f"_{args[4]}"
    outdir = os.path.abspath(f"./DSPS_pickles_fit_{fitname}")
    pdfoutputfilename = os.path.join(outdir, f"fitparams_{fitname}_{low_bound+1}-{start_tag}_to_{high_bound}-{end_tag}.pdf")

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filename_params = os.path.join(outdir, f"fitparams_{fitname}_{low_bound+1}-{start_tag}_to_{high_bound}-{end_tag}.pickle")
    with open(filename_params, "wb") as outf:
        pickle.dump(fit_results_dict, outf)
    plot_figs_to_PDF(pdfoutputfilename, list_of_figs)
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
