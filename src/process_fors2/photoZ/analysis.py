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
from jax import numpy as jnp
from tqdm import tqdm

from process_fors2.fetchData import json_to_inputs
from process_fors2.stellarPopSynthesis import has_redshift

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
    """load_data_for_run _summary_

    :param inputs: _description_
    :type inputs: _type_
    :return: _description_
    :rtype: _type_
    """
    from process_fors2.fetchData import readDSPSHDF5, readTemplatesHDF5, templatesToHDF5
    from process_fors2.photoZ import DATALOC, NIR_filt, NUV_filt, Observation, get_2lists, load_filt, load_galaxy, make_legacy_templates, make_sps_templates, sedpyFilter
    from process_fors2.stellarPopSynthesis import load_ssp

    _ssp_file = (
        None
        if (inp_glob["fitDSPS"]["ssp_file"].lower() == "default" or inp_glob["fitDSPS"]["ssp_file"] == "" or inp_glob["fitDSPS"]["ssp_file"] is None)
        else os.path.abspath(inp_glob["fitDSPS"]["ssp_file"])
    )
    ssp_data = load_ssp(_ssp_file)

    inputs = inp_glob["photoZ"]
    z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + inputs["Z_GRID"]["z_step"], inputs["Z_GRID"]["z_step"])

    # wl_grid = jnp.arange(inputs["WL_GRID"]["lambda_min"], inputs["WL_GRID"]["lambda_max"] + inputs["WL_GRID"]["lambda_step"], inputs["WL_GRID"]["lambda_step"])

    filters_dict = inputs["Filters"]
    for _f in filters_dict:
        filters_dict[_f]["path"] = os.path.abspath(os.path.join(DATALOC, filters_dict[_f]["path"]))
    print("Loading filters :")
    filters_arr = tuple(sedpyFilter(*load_filt(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict)) + (NUV_filt, NIR_filt)
    N_FILT = len(filters_arr) - 2
    # print(f"DEBUG: filters = {filters_arr}")

    print("Building templates :")
    Xfilt = get_2lists(filters_arr)
    # sps_temp_pkl = os.path.abspath(inputs["Templates"])
    # sps_par_dict = read_params(sps_temp_pkl)
    if inputs["Templates"]["overwrite"] or not os.path.isfile(os.path.abspath(inputs["Templates"]["output"])):
        sps_temp_h5 = os.path.abspath(inputs["Templates"]["input"])
        sps_par_dict = readDSPSHDF5(sps_temp_h5)
        if "sps" in inputs["Mode"].lower():
            templ_dict = jax.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        else:
            templ_dict = jax.tree_map(lambda dico: make_legacy_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        _ = templatesToHDF5(inputs["Templates"]["output"], templ_dict)
    else:
        templ_dict = readTemplatesHDF5(inputs["Templates"]["output"])

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
    return z_grid, templ_dict, obs_arr


def extract_pdz(pdf_res, z_grid):
    """pdf_res _summary_

    :param chi2_res: _description_
    :type chi2_res: _type_
    :param z_grid: _description_
    :type z_grid: _type_
    :return: _description_
    :rtype: _type_
    """
    pdf_dict = pdf_res[0]
    zs = pdf_res[1]
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.sum(pdf_arr, axis=0), x=z_grid)
    pdf_arr = pdf_arr / _n2
    pdz_dict = {}
    for key, val in pdf_dict.items():
        joint_pdz = val / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        z_ml = z_grid[jnp.nanargmax(joint_pdz)]
        z_avg = jnp.trapezoid(z_grid * joint_pdz / evidence, x=z_grid)
        pdz_dict.update({key: {"evidence_SED": evidence, "z_ML_SED": z_ml, "z_mean_SED": z_avg}})
    pdz = jnp.sum(pdf_arr, axis=0)
    z_ML = z_grid[jnp.nanargmax(pdz)]
    z_MEAN = jnp.trapezoid(z_grid * pdz, x=z_grid)
    pdz_dict.update({"PDZ": jnp.column_stack((z_grid, pdz)), "z_spec": zs, "z_ML": z_ML, "z_mean": z_MEAN})
    return pdz_dict


def extract_pdz_fromchi2(chi2_res, z_grid):
    """extract_pdz_fromchi2 _summary_

    :param chi2_res: _description_
    :type chi2_res: _type_
    :param z_grid: _description_
    :type z_grid: _type_
    :return: _description_
    :rtype: _type_
    """
    chi2_dict = chi2_res[0]
    zs = chi2_res[1]
    chi2_arr = jnp.array([chi2_templ for _, chi2_templ in chi2_dict.items()])
    _n1 = 100.0  # jnp.max(chi2_arr)
    chi2_arr = chi2_arr - _n1  # 10 * chi2_arr / _n1
    exp_arr = jnp.exp(-0.5 * chi2_arr)
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.sum(exp_arr, axis=0), x=z_grid)
    exp_arr = exp_arr / _n2
    pdz_dict = {}
    for key, val in chi2_dict.items():
        chiarr = val - _n1
        joint_pdz = jnp.exp(-0.5 * chiarr) / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        pdz_dict.update({key: {"SED evidence": evidence}})
    pdz_dict.update({"PDZ": jnp.sum(exp_arr, axis=0), "z_spec": zs})
    return pdz_dict


def extract_pdz_allseds(pdf_res, z_grid):
    """extract_pdz_allseds _summary_

    :param pdf_res: _description_
    :type pdf_res: _type_
    :param z_grid: _description_
    :type z_grid: _type_
    :return: _description_
    :rtype: _type_
    """
    pdf_dict = pdf_res[0]
    zs = pdf_res[1]
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.sum(pdf_arr, axis=0), x=z_grid)
    pdf_arr = pdf_arr / _n2
    pdz_dict = {}
    for key, val in pdf_dict.items():
        joint_pdz = val / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        pdz_dict.update({key: {"p(z, sed)": joint_pdz, "SED evidence": evidence}})
    pdz_dict.update({"PDZ": jnp.sum(pdf_arr, axis=0), "z_spec": zs})
    return pdz_dict


def load_data_for_analysis(conf_json):
    """load_data_for_analysis _summary_

    :param conf_json: _description_
    :type conf_json: _type_
    :return: _description_
    :rtype: _type_
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
    templ_dict = jax.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, wl_grid, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)

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
