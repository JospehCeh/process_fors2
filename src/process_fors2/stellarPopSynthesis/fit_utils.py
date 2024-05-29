"""Module to provide help for fitting and diagnostics
It contains fluxes rescaling function and plots

"""
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=trailing-newlines
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
# pylint: disable=missing-final-newline
# pylint: disable=too-many-lines
# pylint: disable=redefined-outer-name


import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# import numpy as np
from interpax import interp1d
from jax import vmap

from .fit_filters import FilterInfo
from .fitter_dsps import mean_mags, mean_sfr, ssp_spectrum_fromparam

jax.config.update("jax_enable_x64", True)

# build the list of filter tag to plot
ps = FilterInfo()
print(ps.filters_indexlist)
print(ps.filters_surveylist)
print(ps.filters_namelist)

index_selected_filters = np.arange(0, 11)
list_name_f_sel = []
list_wlmean_f_sel = []

for index in index_selected_filters:
    the_filt = ps.filters_transmissionlist[index]
    the_wlmean = the_filt.wave_mean
    list_wlmean_f_sel.append(the_wlmean)
    list_name_f_sel.append(ps.filters_namelist[index])
list_wlmean_f_sel = jnp.array(list_wlmean_f_sel)
list_name_f_sel = np.array(list_name_f_sel)


def func_strip_name(x):
    """
    Strip string of filters name for shorter name plotting
    :param x: name
    :type x: string
    """
    return x.split("_")[-1]


def calc_ratio(wl, spec, w_blue=(3750.0, 3950.0), w_red=(4050.0, 4250)):
    """Calculate the ratio of spectra in the wavelength range.


    :param wl: wavelength
    :type wl: array of float
    :param spec: spectrum in fnu
    :type spec: array of float
    :param w_blue: wavelength range in blue part of the spectrum
    :type w_blue: array of two floats, default to calculate D4000.
    :param w_red: wavelength range in red part of the spectrum
    :type w_red: array of two floats, default to calculate D4000.
    :return: the flux ratio red/blue
    :rtype: float
    """

    indexes_red = np.where(np.logical_and(wl >= w_red[0], wl <= w_red[1]))[0]
    indexes_blue = np.where(np.logical_and(wl >= w_blue[0], wl <= w_blue[1]))[0]
    int_spec_blue = np.trapz(spec[indexes_blue], wl[indexes_blue])
    int_spec_red = np.trapz(spec[indexes_red], wl[indexes_red])
    specratio = int_spec_red / int_spec_blue
    return specratio


def rescale_starlight_inrangefors2(wls, fluxes, Xspec_data_rest, Yspec_data_rest):
    """
    Rescale Starlight model (in rest frame) on rescaled Fors2 data (in rest frame).

    :param wls: StarLight SED wavelength
    :type wls: jax array of floats
    :param fluxes: StarLight SED fluxes
    :type fluxes: jax array of floats
    :param Xspec_data_rest: Fors2 spectrum wavelength in rest-frame
    :type Xspec_data_rest: jax array of floats
    :param Yspec_data_rest: Fors2 spectrum fluxes in rest-frame
    :type Yspec_data_rest: jax array of floats
    :return: StarLight spectrum rescaled to rescaled Fors2 data
    :rtype: jax array of floats and the scaling factor

    """

    # boundaries of the Fors2 spectrum in rest-frame
    xmin = Xspec_data_rest.min()
    xmax = Xspec_data_rest.max()
    selected_indexes = jnp.where(jnp.logical_and(wls > xmin, wls < xmax))[0]

    # select StarLight spectrum part match Fors2 wavelength range
    xsl = wls[selected_indexes]
    ysl = fluxes[selected_indexes]

    # calculate the scaling factor for StarLight
    ysl_pred = interp1d(xsl, Xspec_data_rest, Yspec_data_rest)
    scaling_factor = jnp.mean(ysl_pred / ysl)

    # return rescaled starlight spectrum
    return wls, fluxes * scaling_factor, scaling_factor


def plot_params_kde(samples, hdi_probs=(0.393, 0.865, 0.989), patName=None, fname=None, pcut=None, var_names=None, point_estimate="median"):
    """Plot contour ellipse from samples

    :param samples: _description_
    :type samples: _type_
    :param hdi_probs: _description_, defaults to [0.393, 0.865, 0.989]
    :type hdi_probs: list, optional
    :param patName: _description_, defaults to None
    :type patName: _type_, optional
    :param fname: _description_, defaults to None
    :type fname: _type_, optional
    :param pcut: _description_, defaults to None
    :type pcut: _type_, optional
    :param var_names: _description_, defaults to None
    :type var_names: _type_, optional
    :param point_estimate: _description_, defaults to "median"
    :type point_estimate: str, optional
    """

    if pcut is not None:
        low = pcut[0]
        up = pcut[1]
        # keep only data in the [low, up] percentiles ex. 0.5, 99.5
        samples = {name: value[(value > np.percentile(value, low)) & (value < np.percentile(value, up))] for name, value in samples.items()}
        len_min = np.min([len(value) for name, value in samples.items()])
        len_max = np.max([len(value) for name, value in samples.items()])
        if (len_max - len_min) > 0.01 * len_max:
            print(f"Warning: pcut leads to min/max spls size = {len_min}/{len_max}")
        samples = {name: value[:len_min] for name, value in samples.items()}

    axs = az.plot_pair(
        samples,
        var_names=var_names,
        figsize=(10, 10),
        kind="kde",
        #        marginal_kwargs={"plot_kwargs": {"lw": 3, "c": "b"}},
        kde_kwargs={
            #                "hdi_probs": [0.68, 0.9],  # Plot 68% and 90% HDI contours
            "hdi_probs": hdi_probs,  # 1, 2 and 3 sigma contours
            "contour_kwargs": {"colors": ("r", "green", "blue"), "linewidths": 3},
            "contourf_kwargs": {"alpha": 0},
        },
        point_estimate_kwargs={"lw": 3, "c": "b"},
        marginals=True,
        textsize=20,
        point_estimate=point_estimate,
    )

    plt.tight_layout()

    if patName is not None:
        patName_patch = mpatches.Patch(color="b", label=patName)
        axs[0, 0].legend(handles=[patName_patch], fontsize=40, bbox_to_anchor=(1, 0.7))
    if fname is not None:
        plt.savefig(fname)
        plt.close()


def rescale_photometry(params, wls, mags, errmags, z_obs):
    """
    Rescale photometric data points from observation frame onto SED fnu scale in restframe
    for plotting on the same axis the phtometry and the SED
    :param params: fitted parameters
    :type params: dictionnary of parameters
    :param wls: central wavelength of photometric observation filters
    :type wls: jax array of floats
    :param mags: selected magnitudes used for the fit (usually AB magnitudes)
    :type mags: jax array of floats
    :param errmags: errors on magnitudes (usually AB magnitudes)
    :type errmags: jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :return: wavelengths, rescaled fluxes and errors and scaling factor of phtometric data
    :rtype: 3 jax arrays of floats and a float
    """
    # transform magnitudes into fluxes
    fluxes = vmap(lambda x: jnp.power(10.0, -0.4 * x), in_axes=0)(mags)
    efluxes = vmap(lambda x, y: jnp.power(10.0, -0.4 * x) * y)(mags, errmags)

    # transform from observation frame to restframe
    wls_rest = wls / (1.0 + z_obs)
    fluxes *= 1.0 + z_obs
    efluxes *= 1.0 + z_obs

    # calculate the SED model from the parameters
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    # calculate the scaling factor for photometric points
    flux_pred = interp1d(wls_rest, x, y_dust)
    scaling_factor = jnp.mean(flux_pred / fluxes)

    return wls_rest, fluxes * scaling_factor, efluxes * scaling_factor, scaling_factor


def rescale_spectroscopy(params, wls, fluxes, efluxes, z_obs):
    """
    Calculate the rescaling factor of fors2 spectroscopy on restframe SSP spectrum plot.
    Comparison is done in restframe.

    :param params:fitted parameters
    :type params:dictionnary of parameters
    :param wls: wavelength of spectroscopic Fors2  observations
    :type wls: jax array of floats
    :param fluxes: relative fluxes of Fors2
    :type fluxes: jax array of floats
    :param efluxes: errors on Fors2 fluxes
    :type efluxes:jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :return: wavelengths, rescaled fluxes and errors and scaling factor of spectroscopic data
    :rtype: 3 jax arrays of floats and a float

    """
    # convert the spectrum in rest frame
    wls_rest = wls  # /(1.+z_obs) spectres déjà dans le restframe lors de l'extraction de FORS2
    # fluxes *= (1.+z_obs)
    # efluxes *= (1.+z_obs)

    # compute the model from params (params from photometry)
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    # calculate the scaling factor
    flux_pred = interp1d(wls_rest, x, y_dust)
    scaling_factor = jnp.mean(flux_pred / fluxes)

    # return rescaled fluxes in rest-frame
    return wls_rest, fluxes * scaling_factor, efluxes * scaling_factor, scaling_factor


def plot_fit_ssp_photometry(params, X, wls, mags, errmags, z_obs, subtit, ax=None):
    """
    Plot SSP model fitted and photometric data points. Photometric data point are rescaled
    from observation frame to rest-frame.

    :param params:fitted parameters
    :type params:dictionnary of parameters
    :param wls:central wavelength of photometric observation filters
    :type wls: jax array of floats
    :param mags: selected magnitudes used for the fit (usually AB magnitudes)
    :type mags: jax array of floats
    :param errmags: errors on magnitudes (usually AB magnitudes)
    :type errmags:jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtitle: str
    :return: plot the figure
    :rtype: None
    """
    # Compute the SED from the fitted model
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)
    mean_mag = mean_mags(X, params, z_obs)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax_phot = ax.twinx()
    ax.set_yscale("log")
    ax.set_xscale("log")

    # plot model with/without dust
    ax.plot(x * (1 + z_obs), y_dust, "-", color="green", lw=1, label="fitted DSPS model with dust")
    ax.plot(x * (1 + z_obs), y_nodust, "-", color="red", lw=1, label="fitted DSPS model no dust")

    # rescale photometric data-points from observation frame to rest frame
    # xphot, yphot, eyphot, factor = rescale_photometry_ageDepMet_Q(params, wls, mags, errmags, z_obs)

    # plot photometric data-points
    label = "Photometry for " + subtit
    # ax.errorbar(xphot , yphot, yerr=eyphot, marker='o', color="black",ecolor="black",markersize=12,lw=2,label=label)
    ax_phot.errorbar(wls, mags, yerr=errmags, marker="o", color="black", ecolor="black", markersize=9, lw=2, label=label)
    ax_phot.scatter(wls, mean_mag, marker="s", c="cyan", s=81, lw=2, label="Modeled photometry")

    title = "SED $L_\\nu$ with and without dust and rescaled photometry (obs. frame)"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlabel("$\\lambda (\\AA)$")
    ax.set_ylabel("$L_\\nu(\\lambda)$")
    ax_phot.set_ylabel("$m_{AB}$")
    ax_phot.legend()

    ymax = y_nodust.max()
    ylim_max = ymax * 3
    ylim_min = ymax / 2e4

    filter_tags = [func_strip_name(n) for n in list_name_f_sel]
    for idx, tag in enumerate(filter_tags):
        ax.text(wls[idx], 2.0 * ymax, tag, fontsize=12, fontweight="bold", horizontalalignment="center", verticalalignment="center")
        ax.axvline(wls[idx], linestyle=":")

    ax.set_xlim(1e3, 1e6)
    ax.set_ylim(ylim_min, ylim_max)
    ax_phot.set_ylim(27, 18)

    ax.grid()
    plt.show(block=False)


def plot_fit_ssp_spectroscopy(params, Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, z_obs, subtit, ax=None):
    """
    Plot SSP model fitted and Fors2 spectroscopic data points. Spectroscopic data points are rescaled
    from observation frame to rest-frame.

    :param params: fitted parameters on rescaled spectroscopic data
    :type params: dictionnary of parameters
    :param Xspec_data_rest: wavelength of spectroscopic observation in restframe
    :type Xspec_data_rest: jax array of floats
    :param Yspec_data_rest: rescaled fluxes of spectroscopic data
    :type Yspec_data_rest: jax array of floats
    :param EYspec_data_rest: errors on rescaled fluxes of spectroscopic data
    :type EYspec_data_rest:jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtitle: str
    :return: plot the figure
    :rtype: None

    """

    # calculate the SED
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_yscale("log")
    ax.set_xscale("log")

    # plot the fitted model
    ax.plot(x, y_dust, "-", color="green", lw=1, label="fitted DSPS model with dust")
    ax.plot(x, y_nodust, "-", color="red", lw=1, label="fitted DSPS model No dust")

    # plot spectroscopic data (rest-frame)
    # xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest*params["SCALE"], EYspec_data_rest*params["SCALE"]
    xspec_r, yspec, _ = Xspec_data_rest, Yspec_data_rest, EYspec_data_rest
    lab = "Fors spectrum " + subtit
    ax.plot(xspec_r, yspec, "b-", lw=3, label=lab)

    title = "SED $L_\\nu$ with and without dust and rescaled spectroscopy(rest frame)"
    ax.set_title(title)
    ax.legend(loc="lower right")

    ax.set_xlabel("$\\lambda (\\AA)$")
    ax.set_ylabel("$L_\\nu(\\lambda)$")

    ymax = y_nodust.max()
    ylim_max = ymax * 3
    ylim_min = ymax / 2e4
    ax.set_xlim(1.5e3, 5e4)
    ax.set_ylim(ylim_min, ylim_max)

    ax.grid()
    plt.show(block=False)


def plot_fit_ssp_spectrophotometry(params, Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, X, Xphot_data_rest, Yphot_data_rest, EYphot_data_rest, z_obs, subtit, ax=None):
    """
    Plot SSP model fitted with combined spectro and photometric data.
    Both data are rescaled and set to rest-frame.

    :param params: fitted parameters on rescaled spectroscopic data
    :type params: dictionnary of parameters
    :param Xspec_data_rest: wavelength of spectroscopic observation in restframe
    :type Xspec_data_rest: jax array of floats
    :param Yspec_data_rest: rescaled fluxes of spectroscopic data
    :type Yspec_data_rest: jax array of floats
    :param EYspec_data_rest: errors on rescaled fluxes of spectroscopic data
    :type EYspec_data_rest:jax array of floats
    :param Xphot_data_rest: filter central wavelenth in rest frame
    :type Xphot_data_rest: jax array of floats
    :param Yphot_data_rest: photometric flux in rest frame
    :type Yphot_data_rest: jax array of floats
    :param EYphot_data_rest: error on photometric flux in rest frame
    :type EYphot_data_rest: jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtitle: str
    :return: plot the figure
    :rtype: None

    """
    # calculate the SED model from fitted parameters
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax_phot = ax.twinx()
    ax.set_yscale("log")
    ax.set_xscale("log")

    # plot SED model
    ax.plot(x * (1 + z_obs), y_dust, "-", color="green", lw=1, label="DSPS output\nwith dust")
    ax.plot(x * (1 + z_obs), y_nodust, "-", color="red", lw=1, label="DSPS output\nwithout dust")

    # xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest*params["SCALE"], EYspec_data_rest*params["SCALE"]
    xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest, EYspec_data_rest

    # plot Fors2 data
    label = "Fors2 spectrum\n" + subtit
    ax.plot(xspec_r * (1 + z_obs), yspec, "b-", lw=0.5)
    ax.fill_between(xspec_r * (1 + z_obs), yspec - eyspec, yspec + eyspec, color="b", alpha=0.4, label=label)

    # plot photometric data
    label = "Photometry for\n" + subtit

    # need to rescale photometric data with dusty model
    # .............................................................................
    # calculate the scaling factor for photometric points
    # fluxphot_pred = interp1d(Xphot_data_rest, x, y_dust)
    # scaling_factor = jnp.mean(fluxphot_pred / Yphot_data_rest)
    # xphot , yphot, eyphot =  Xphot_data_rest,Yphot_data_rest*scaling_factor,EYphot_data_rest*scaling_factor
    xphot, yphot, eyphot = Xphot_data_rest, Yphot_data_rest, EYphot_data_rest

    # ...............................................................................

    # xphot , yphot, eyphot = Xphot_data_rest,Yphot_data_rest,EYphot_data_rest
    ax_phot.errorbar(xphot, yphot, yerr=eyphot, marker="o", color="black", ecolor="black", markersize=9, lw=2, label=label)
    mean_mag = mean_mags(X, params, z_obs)
    ax_phot.scatter(xphot, mean_mag, marker="s", c="cyan", s=81, lw=2, label="Modeled\nphotometry")

    title = "DSPS fitting results (obs. frame)"
    ax.set_title(title)
    ax.legend()  # (loc="upper left", bbox_to_anchor=(1.1, 1.0))

    ymax = y_nodust.max()
    ylim_max = ymax * 3.0
    ylim_min = ymax / 3e4

    filt_sel = [wlmen in xphot for wlmen in list_wlmean_f_sel]
    filter_tags = [func_strip_name(n) for n in list_name_f_sel[filt_sel]]
    for idx, tag in enumerate(filter_tags):
        ax.text(xphot[idx], 2.0 * ymax - (idx % 2) * 0.5 * ymax, tag, fontsize=10, fontweight="bold", horizontalalignment="center", verticalalignment="center")
        ax.axvline(xphot[idx], linestyle=":")

    ax.set_xlabel("$\\lambda\\ [\\AA]$")
    ax.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")
    ax_phot.set_ylabel("$m_{AB}$")
    ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

    ax.set_xlim(1.5e3, 5e4)
    ax.set_ylim(ylim_min, ylim_max)
    ax_phot.set_ylim(27, 16)

    ax.grid()
    plt.show(block=False)


def plot_input_spectrophotometry(params, Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, X, Xphot_data_rest, Yphot_data_rest, EYphot_data_rest, z_obs, subtit, plot_phot=True, ax=None):
    """
    Plot input model (combined spectro and photometric data).

    :param params: fitted parameters on rescaled spectroscopic data
    :type params: dictionnary of parameters
    :param Xspec_data_rest: wavelength of spectroscopic observation in restframe
    :type Xspec_data_rest: jax array of floats
    :param Yspec_data_rest: rescaled fluxes of spectroscopic data
    :type Yspec_data_rest: jax array of floats
    :param EYspec_data_rest: errors on rescaled fluxes of spectroscopic data
    :type EYspec_data_rest:jax array of floats
    :param Xphot_data_rest: filter central wavelenth in rest frame
    :type Xphot_data_rest: jax array of floats
    :param Yphot_data_rest: photometric flux in rest frame
    :type Yphot_data_rest: jax array of floats
    :param EYphot_data_rest: error on photometric flux in rest frame
    :type EYphot_data_rest: jax array of floats
    :param z_obs: redshift of observed galaxy object
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtit: str
    :plot_phot: whether or not to add the photometry to the plot
    :return: plot the figure
    :rtype: None

    """
    # calculate the SED model from fitted parameters
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    ymax = y_nodust.max()
    ylim_max = ymax * 3.0
    ylim_min = ymax / 3e4

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_yscale("log")
    ax.set_xscale("log")

    xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest, EYspec_data_rest

    # plot Fors2 data
    label = "Fors2 spectrum\n" + subtit
    ax.plot(xspec_r * (1 + z_obs), yspec, "b-", lw=0.5)
    ax.fill_between(xspec_r * (1 + z_obs), yspec - eyspec, yspec + eyspec, color="b", alpha=0.4, label=label)

    # plot photometric data
    if plot_phot:
        ax_phot = ax.twinx()
        label = "Photometry for\n" + subtit
        xphot, yphot, eyphot = Xphot_data_rest, Yphot_data_rest, EYphot_data_rest
        ax_phot.errorbar(xphot, yphot, yerr=eyphot, marker="o", color="black", ecolor="black", markersize=9, lw=2, label=label)
        filt_sel = [wlmen in xphot for wlmen in list_wlmean_f_sel]
        filter_tags = [func_strip_name(n) for n in list_name_f_sel[filt_sel]]
        for idx, tag in enumerate(filter_tags):
            ax.text(xphot[idx], 2.0 * ymax - (idx % 2) * 0.5 * ymax, tag, fontsize=10, fontweight="bold", horizontalalignment="center", verticalalignment="center")
            ax.axvline(xphot[idx], linestyle=":")
        ax_phot.set_ylabel("$m_{AB}$")
        ax_phot.legend(loc="lower right")  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))
        ax_phot.set_ylim(27, 16)

    title = "DSPS inputs (obs. frame)"
    ax.set_title(title)
    ax.legend()  # (loc="upper left", bbox_to_anchor=(1.1, 1.0))

    ax.set_xlabel("$\\lambda\\ [\\AA]$")
    ax.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")

    ax.set_xlim(1.5e3, 5e4)
    ax.set_ylim(ylim_min, ylim_max)

    ax.grid()
    plt.show(block=False)


def plot_fit_ssp_spectrophotometry_sl(params, Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, X, Xphot_data_rest, Yphot_data_rest, EYphot_data_rest, w_sl, fnu_sl, z_obs, subtit, ax=None):
    """
    Plot SSP models DSPS + StarLight fitted with combined spectro and photometric data.
    Both data are rescaled and set to rest-frame.

    :param params: fitted parameters on rescaled spectroscopic data
    :type params: dictionnary of parameters
    :param Xspec_data_rest: wavelength of spectroscopic observation in restframe
    :type Xspec_data_rest: jax array of floats
    :param Yspec_data_rest: rescaled fluxes of spectroscopic data
    :type Yspec_data_rest: jax array of floats
    :param EYspec_data_rest: errors on rescaled fluxes of spectroscopic data
    :type EYspec_data_rest:jax array of floats
    :param Xphot_data_rest: filter central wavelenth in rest frame
    :type Xphot_data_rest: jax array of floats
    :param Yphot_data_rest: photometric flux in rest frame
    :type Yphot_data_rest: jax array of floats
    :param EYphot_data_rest: error on photometric flux in rest frame
    :type EYphot_data_rest: jax array of floats
    :param z_obs: redshift of observed galaxy object
    :param w_sl: wavelength for StarLight model in restframe
    :type w_sl: jax array of floats
    :param fnu_sl: flux of Starlight model in restframe
    :type fnu_sl: jax array of floats
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtitle: str
    :return: plot the figure
    :rtype: None

    """
    # compute the model from the parameters
    x, y_nodust, y_dust = ssp_spectrum_fromparam(params, z_obs)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax_phot = ax.twinx()
    ax.set_yscale("log")
    ax.set_xscale("log")

    # plot models
    ax.plot(x * (1 + z_obs), y_dust, "-", color="green", lw=1, label="DSPS output\nwith dust")
    ax.plot(x * (1 + z_obs), y_nodust, "-", color="red", lw=1, label="DSPS output\nwithout dust")
    ax.plot(w_sl * (1 + z_obs), fnu_sl, "-", color="grey", lw=2, label="StarLight model")

    # plot Fors2 spectrum in restframe
    # xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest*params["SCALE"], EYspec_data_rest*params["SCALE"]
    xspec_r, yspec, eyspec = Xspec_data_rest, Yspec_data_rest, EYspec_data_rest
    label = "Fors2 spectrum\n" + subtit
    ax.plot(xspec_r * (1 + z_obs), yspec, "b-", lw=0.5)
    ax.fill_between(xspec_r * (1 + z_obs), yspec - eyspec, yspec + eyspec, color="b", alpha=0.4, label=label)

    # plot Photometric data in restframe
    label = "Photometry for\n" + subtit
    # xphot , yphot, eyphot = Xphot_data_rest,Yphot_data_rest,EYphot_data_rest
    # .............................................................................
    # calculate the scaling factor for photometric points
    # fluxphot_pred = interp1d(Xphot_data_rest, x, y_dust)
    # scaling_factor = jnp.mean(fluxphot_pred / Yphot_data_rest)
    # xphot, yphot, eyphot =  Xphot_data_rest, Yphot_data_rest*scaling_factor, EYphot_data_rest*scaling_factor
    xphot, yphot, eyphot = Xphot_data_rest, Yphot_data_rest, EYphot_data_rest

    mean_mag = mean_mags(X, params, z_obs)

    # ...............................................................................

    ax_phot.errorbar(xphot, yphot, yerr=eyphot, marker="o", color="black", ecolor="black", markersize=9, lw=2, label=label)
    ax_phot.scatter(xphot, mean_mag, marker="s", c="cyan", s=81, lw=2, label="Modeled\nphotometry")

    title = "DSPS fitting results (obs. frame)"
    ax.set_title(title)
    ax.legend(loc="lower right")

    ymax = y_nodust.max()
    ylim_max = ymax * 3.0
    ylim_min = ymax / 2e4

    filter_tags = [func_strip_name(n) for n in list_name_f_sel]
    for idx, tag in enumerate(filter_tags):
        ax.text(xphot[idx], 2.0 * ymax, tag, fontsize=12, fontweight="bold", horizontalalignment="center", verticalalignment="center")
        ax.axvline(xphot[idx], linestyle=":")

    ax.set_xlim(1e3, 1e6)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel("$\\lambda\\ [\\AA]$")
    ax.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")
    ax_phot.set_ylabel("$m_{AB}$")
    ax_phot.legend()
    ax_phot.set_ylim(27, 18)

    ax.grid()
    plt.show(block=False)


def plot_SFH(params, z_obs, subtit, ax=None):
    """
    Plot Star Formation History

    :param params: fitted parameters on rescaled spectroscopic data
    :type params: dictionnary of parameters
    :type z_obs: float
    :param ax: matplotlib axis to plot the figure, default None
    :type ax: matplotlib axis
    :param subtit: info on the photometric data on the corresponding spectrum
    :type subtitle: str
    :return: plot the figure
    :rtype: None
    """

    t_obs, tarr, sfh_gal = mean_sfr(params, z_obs)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # plot star formation history
    ax.plot(tarr, sfh_gal, "-k", lw=2)
    ax.axvline(t_obs, color="red")

    sfr_max = sfh_gal.max() * 1.1
    sfr_min = 0.0
    ax.set_ylim(sfr_min, sfr_max)

    ax.set_title("Fitted Star Formation History (SFH) for " + subtit)
    ax.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
    ax.set_ylabel(r"${\rm SFR\ [M_{\odot}/yr]}$")
    ax.grid()
    # ax.legend()
    plt.show(block=False)


def plot_SFH_bootstrap(dict_for_fit, results_dict, params_names, ax=None):
    """
    Plot Star Formation History

    Parameters
    ----------
    dict_for_fit : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    results_dict : dict
        Path to the HDF5 file gathering outputs from GELATO run.
    params_names : list of str
        Name of the fitted parameters for dictionary creation.
    ax : matplotlib axes, optional
        Axes to plot the results on. If None, a new axes objetc is created. The default is None.

    Returns
    -------
    None
    """
    from process_fors2.stellarPopSynthesis import paramslist_to_dict

    if ax is None:
        _, ax = plt.subplots(1, 1)

    sfh_list = []

    for tag, fit_dict in results_dict.items():
        dict_params_fit = paramslist_to_dict(fit_dict["fit_params"], params_names)
        data_dict = dict_for_fit[tag]
        t_obs, tarr, sfh_gal = mean_sfr(dict_params_fit, data_dict["redshift"])
        sfh_list.append(sfh_gal)

    sfh_list = np.array(sfh_list)

    mean_sfh = np.mean(sfh_list, axis=0)
    std_sfh = np.std(sfh_list, axis=0)

    # plot star formation history
    ax.plot(tarr, mean_sfh, "-k", lw=1)
    ax.fill_between(tarr, mean_sfh + std_sfh, mean_sfh - std_sfh, color="gray", alpha=0.5)
    ax.axvline(t_obs, color="red")

    sfr_max = (mean_sfh + std_sfh).max() * 1.1
    sfr_min = 0.0
    ax.set_ylim(sfr_min, sfr_max)

    ax.set_title(f"Fitted Star Formation History (SFH) for {data_dict['title'].split('_')[0]}")
    ax.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
    ax.set_ylabel(r"${\rm SFR\ [M_{\odot}/yr]}$")
    ax.grid()
    # ax.legend()
    plt.show(block=False)


def plot_bootstrap_ssp_spectrophotometry(dict_for_fit, results_dict, params_names, ax=None):
    """
    Plot SSP model fitted with combined spectro and photometric data.
    Both data are set to observation frame.

    Parameters
    ----------
    dict_for_fit : path or str
        Path to the HDF5 file gathering outputs from the cross-match between spectra and photometry - as used as an input for GALETO for instance.
    results_dict : dict
        Path to the HDF5 file gathering outputs from GELATO run.
    params_names : list of str
        Name of the fitted parameters for dictionary creation.
    ax : matplotlib axes, optional
        Axes to plot the results on. If None, a new axes objetc is created. The default is None.

    Returns
    -------
    None
    """
    from process_fors2.stellarPopSynthesis import paramslist_to_dict

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ref_mags = []
    calc_mags = []
    calc_ydust = []
    calc_ynodust = []

    for loc, (tag, fit_dict) in enumerate(results_dict.items()):
        dict_params_fit = paramslist_to_dict(fit_dict["fit_params"], params_names)
        data_dict = dict_for_fit[tag]
        if loc == 0:
            z_obs = data_dict["redshift"]
            subtit = data_dict["title"].split("_")[0]
            xspec_r, yspec, eyspec, xphot = data_dict["wavelengths"], data_dict["fnu"], data_dict["fnu_err"], data_dict["wl_mean_filters"]

        # calculate the SED model from fitted parameters
        x, y_nodust, y_dust = ssp_spectrum_fromparam(dict_params_fit, data_dict["redshift"])
        mean_mag = mean_mags(data_dict["filters"], dict_params_fit, data_dict["redshift"])
        calc_ynodust.append(y_nodust)
        calc_ydust.append(y_dust)
        calc_mags.append(mean_mag)
        ref_mags.append(data_dict["mags"])

    calc_ynodust = np.array(calc_ynodust)
    calc_ydust = np.array(calc_ydust)
    calc_mags = np.array(calc_mags)
    ref_mags = np.array(ref_mags)

    mean_ydust = np.mean(calc_ydust, axis=0)
    mean_ynodust = np.mean(calc_ynodust, axis=0)
    mean_calcmags = np.mean(calc_mags, axis=0)
    mean_refmags = np.mean(ref_mags, axis=0)

    std_ydust = np.std(calc_ydust, axis=0)
    std_ynodust = np.std(calc_ynodust, axis=0)
    std_calcmags = np.std(calc_mags, axis=0)
    std_refmags = np.std(ref_mags, axis=0)

    ax_phot = ax.twinx()
    ax.set_yscale("log")
    ax.set_xscale("log")

    # plot SED model
    ax.plot(x * (1 + z_obs), mean_ydust, "-", color="green", lw=1, label="DSPS output\nwith dust")
    ax.plot(x * (1 + z_obs), mean_ynodust, "-", color="red", lw=1, label="DSPS output\nwithout dust")
    ax.fill_between(x * (1 + z_obs), mean_ydust + std_ydust, mean_ydust - std_ydust, color="green", alpha=0.4, lw=1)
    ax.fill_between(x * (1 + z_obs), mean_ynodust + std_ynodust, mean_ynodust - std_ynodust, color="red", alpha=0.4, lw=1)

    # plot Fors2 data
    label = "Fors2 spectrum\n" + subtit
    ax.plot(xspec_r * (1 + z_obs), yspec, "b-", lw=0.5)
    ax.fill_between(xspec_r * (1 + z_obs), yspec - eyspec, yspec + eyspec, color="b", alpha=0.4, label=label)

    # plot photometric data
    label = "Photometry for\n" + subtit
    ax_phot.errorbar(xphot, mean_refmags, yerr=std_refmags, marker="o", color="black", ecolor="black", markersize=6, lw=2, label=label)
    ax_phot.errorbar(xphot, mean_calcmags, yerr=std_calcmags, marker="s", c="cyan", ecolor="cyan", markersize=6, lw=2, label="Modeled\nphotometry")

    title = "DSPS fitting results (obs. frame)"
    ax.set_title(title)
    ax.legend()  # (loc="upper left", bbox_to_anchor=(1.1, 1.0))

    ymax = (mean_ynodust + std_ynodust).max()
    ylim_max = ymax * 3.0
    ylim_min = ymax / 3e4

    filt_sel = [wlmen in xphot for wlmen in list_wlmean_f_sel]
    filter_tags = [func_strip_name(n) for n in list_name_f_sel[filt_sel]]
    for idx, tag in enumerate(filter_tags):
        ax.text(xphot[idx], 2.0 * ymax - (idx % 2) * 0.5 * ymax, tag, fontsize=10, fontweight="bold", horizontalalignment="center", verticalalignment="center")
        ax.axvline(xphot[idx], linestyle=":")

    ax.set_xlabel("$\\lambda\\ [\\AA]$")
    ax.set_ylabel("$L_\\nu(\\lambda)\\ [\\mathrm{L_{\\odot} . Hz^{-1}}]$")
    ax_phot.set_ylabel("$m_{AB}$")
    ax_phot.legend()  # (loc="lower left", bbox_to_anchor=(1.1, 0.0))

    ax.set_xlim(1.5e3, 5e4)
    ax.set_ylim(ylim_min, ylim_max)
    ax_phot.set_ylim(27, 16)

    ax.grid()
    plt.show(block=False)
