import jaxopt
import pandas as pd
from jax import jit, vmap
from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax.tree_util import tree_map

from process_fors2.stellarPopSynthesis import SSPParametersFit

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


@jit
def mean_icolors(params, wls, filt_trans_arr, z_obs, ssp_data, iband_num):
    """mean_icolors _summary_

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
    from process_fors2.stellarPopSynthesis import mean_mags

    mags = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    imag = mags.at[iband_num].get()
    return mags - imag


vmap_mean_icolors = vmap(mean_icolors, in_axes=(0, None, None, 0, None, None))


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
    from process_fors2.stellarPopSynthesis import mean_mags, red_chi2

    z_obs, anu = z_anu
    params = fixed_pars.at[13].set(anu)
    all_mags_predictions = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_mags_predictions, mags_measured, sigma_mag_obs)
    return redchi2


@jit
def lik_colr_z_anu(z_anu, fixed_pars, wls, filt_trans_arr, clrs_measured, sigma_clr_obs, ssp_data):
    """lik_colr_z_anu _summary_

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
    from process_fors2.stellarPopSynthesis import red_chi2

    z_obs, anu = z_anu
    params = fixed_pars.at[13].set(anu)
    all_clrs_predictions = mean_icolors(params, wls, filt_trans_arr, z_obs, ssp_data)
    redchi2 = red_chi2(all_clrs_predictions, clrs_measured, sigma_clr_obs)
    return redchi2


def vmap_mags_zp_anu(fixed_pars, fwls, filts_transm, omags, omagerrs, ssp_data):
    """vmap_mags_zp_anu _summary_

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
        res_m = minimize(lik_mag_z_anu, jnp.array([0.5, INIT_PARAMS[13]]), (fixed_pars, fwls, filts_transm, _omags, _oerrs, ssp_data), method="BFGS")
        return res_m.x

    vsolve = vmap(solve, in_axes=(0, 0))
    return vsolve(omags, omagerrs)


def vmap_colrs_zp_anu(fixed_pars, fwls, filts_transm, ocolrs, ocolrerrs, ssp_data):
    """vmap_colrs_zp_anu _summary_

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
        res_m = minimize(lik_colr_z_anu, jnp.array([0.5, INIT_PARAMS[13]]), (fixed_pars, fwls, filts_transm, _ocolrs, _oerrs, ssp_data), method="BFGS")
        return res_m.x

    vsolve = vmap(solve, in_axes=(0, 0))
    return vsolve(ocolrs, ocolrerrs)  # params_m


def treemap_mags_zp_anu(fixed_pars, zmax, fwls, filts_transm, omags, omagerrs, ssp_data):
    """treemap_mags_zp_anu _summary_

    :param fixed_pars: _description_
    :type fixed_pars: _type_
    :param zmax: _description_
    :type zmax: _type_
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
    :return: _description_
    :rtype: _type_
    """
    from process_fors2.stellarPopSynthesis import istuple

    lbfgsb_magzanu = jaxopt.ScipyBoundedMinimize(fun=lik_mag_z_anu, method="L-BFGS-B", maxiter=1000)

    def solve(arg_tupl):
        _omags, _oerrs = arg_tupl
        pars, stat = lbfgsb_magzanu.run(
            jnp.array([0.5, INIT_PARAMS[13]]), (jnp.array([0.0, PARAMS_MIN[13]]), jnp.array([zmax, PARAMS_MAX[13]])), fixed_pars, fwls, filts_transm, _omags, _oerrs, ssp_data
        )
        return pars

    _arglist = [tuple((ma, mer)) for ma, mer in zip(omags, omagerrs, strict=True)]
    fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)

    return jnp.array(fit_results_tree)


def treemap_colrs_zp_anu(fixed_pars, zmax, fwls, filts_transm, ocolrs, ocolrerrs, ssp_data):
    """treemap_colrs_zp_anu _summary_

    :param fixed_pars: _description_
    :type fixed_pars: _type_
    :param zmax: _description_
    :type zmax: _type_
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
    :return: _description_
    :rtype: _type_
    """
    from process_fors2.stellarPopSynthesis import istuple

    lbfgsb_clrzanu = jaxopt.ScipyBoundedMinimize(fun=lik_colr_z_anu, method="L-BFGS-B", maxiter=1000)

    def solve(arg_tupl):
        _oclrs, _oerrs = arg_tupl
        pars, stat = lbfgsb_clrzanu.run(
            jnp.array([0.5, INIT_PARAMS[13]]), (jnp.array([0.0, PARAMS_MIN[13]]), jnp.array([zmax, PARAMS_MAX[13]])), fixed_pars, fwls, filts_transm, _oclrs, _oerrs, ssp_data
        )
        return pars

    _arglist = [tuple((co, cer)) for co, cer in zip(ocolrs, ocolrerrs, strict=True)]
    fit_results_tree = tree_map(lambda otupl: solve(otupl), _arglist, is_leaf=istuple)

    return jnp.array(fit_results_tree)
