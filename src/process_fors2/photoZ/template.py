#!/usr/bin/env python3
"""
Module to specify and use SED templates for photometric redshifts estimation algorithms.
Insipired by previous developments in [EmuLP](https://github.com/JospehCeh/EmuLP).

Created on Thu Aug 1 12:59:33 2024

@author: joseph
"""

import pickle
from collections import namedtuple

from jax import vmap

from process_fors2.stellarPopSynthesis import SSPParametersFit, mean_mags, mean_spectrum, paramslist_to_dict

_DUMMY_P_ADQ = SSPParametersFit()

BaseTemplate = namedtuple("BaseTemplate", ["name", "flux", "z_sps"])
SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"])


def read_params(pickle_file):
    """read_params _summary_

    :param pickle_file: _description_
    :type pickle_file: _type_
    :return: _description_
    :rtype: _type_
    """
    new_dict = {}
    with open(pickle_file, "rb") as pkl:
        par_dict = pickle.load(pkl)
    for tag, dico in par_dict.items():
        params_dict = paramslist_to_dict(dico["fit_params"], _DUMMY_P_ADQ.PARAM_NAMES_FLAT)
        params_dict.update({"redshift": dico["zobs"], "tag": tag})
        new_dict.update({tag: params_dict})
    return new_dict


v_mags = vmap(mean_mags, in_axes=(None, None, 0))


# @jit
def calc_nuvk(wls, params_dict, zobs):
    """calc_nuvk _summary_

    :param wls: _description_
    :type wls: _type_
    :param params_dict: _description_
    :type params_dict: _type_
    :param zobs: _description_
    :type zobs: _type_
    :return: _description_
    :rtype: _type_
    """
    from process_fors2.photoZ import NIR_filt, NUV_filt, ab_mag

    rest_sed = mean_spectrum(wls, params_dict, zobs)
    nuv = ab_mag(NUV_filt.wavelengths, NUV_filt.transmission, wls, rest_sed)
    nir = ab_mag(NIR_filt.wavelengths, NIR_filt.transmission, wls, rest_sed)
    return nuv - nir


v_nuvk = vmap(calc_nuvk, in_axes=(None, None, 0))


def make_sps_templates(params_dict, filt_tup, redz, wl_grid, id_imag=3):
    """make_sps_templates _summary_

    :param params_dict: _description_
    :type params_dict: _type_
    :param filt_tup: _description_
    :type filt_tup: _type_
    :param redz: _description_
    :type redz: _type_
    :param wl_grid: _description_
    :type wl_grid: _type_
    :param id_imag: _description_, defaults to 3
    :type id_imag: int, optional
    :return: _description_
    :rtype: _type_
    """
    name = params_dict.pop("tag")
    z_sps = params_dict.pop("redshift")
    nuvk = v_nuvk(wl_grid, params_dict, redz)
    ab_mags = v_mags(filt_tup, params_dict, redz)
    colors = ab_mags[:, :-1] - ab_mags[:, 1:]
    i_mag = ab_mags[:, id_imag]
    return SPS_Templates(name, z_sps, redz, i_mag, colors, nuvk)


"""OLD FUNCTIONS FOR REFERENCE
def make_base_template(ident, specfile, wl_grid):
    wl, _lums = np.loadtxt(os.path.abspath(specfile), unpack=True)
    _inds = jnp.argsort(wl)
    wls = wl[_inds]
    lum = _lums[_inds]
    wavelengths, lums = jnp.array(wls), jnp.array([l if l>0. else 1.0e-20 for l in lum])
    lumins = jnp.interp(wl_grid, wavelengths, lums, left=0., right=0., period=None)
    return ident, lumins

def nojit_no_ext_make_template(base_temp_lums, filts, z, cosmo, wl_grid):
    lumins = base_temp_lums
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    print(f"Dist. modulus = {d_modulus}")
    mags = jnp.array([filter.noJit_ab_mag(filt.wavelengths, filt.transmission, zshift_wls, lumins) + d_modulus for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_scaled_template(base_temp_lums, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid,  wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))

    scale = calc_scale_arrs(f_ab, galax_fab, galax_fab_err)
    print(f"Scale={scale}")
    scaled_lumins = ext_lumins*scale
    scaled_mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, scaled_lumins) + d_modulus for filt in filts])
    scaled_f_ab = jnp.power(10., -0.4*(scaled_mags+48.6))
    return scaled_mags

@partial(jit, static_argnums=4)
#@partial(vmap, in_axes=(None, None, 0, 0, None, None))
def make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    #ext_lumins = base_temp_lums*extinc_arr
    #zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    #d_modulus = Cosmology.calc_distMod(cosmo, z)
    #d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid*(1.+z),
    base_temp_lums*extinc_arr*opacities) + cosmology.distMod(cosmo, z) for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def make_dusty_template(base_temp_lums, filts, extinc_arr, wl_grid):
    #ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr)
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid, calc_dusty_transm(base_temp_lums, extinc_arr)) for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def calc_fab(filts, wvls, lums, d_mod=0.):
    mags = jnp.array([filter.ab_mag(filt.wavelengths, filt.transmission, wvls, lums) + d_mod for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def make_scaled_template(base_temp_lums, filts, extinc_arr, galax_fab, galax_fab_err, z, wl_grid, d_modulus, opacities):
    ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr) * opacities
    zshift_wls = (1.+z)*wl_grid
    #f_ab = calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus)
    #scale = calc_scale_arrs(calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus), galax_fab, galax_fab_err)
    #scaled_lumins = ext_lumins*scale
    return calc_fab(filts,
                    zshift_wls,
                    ext_lumins*calc_scale_arrs(calc_fab(filts, zshift_wls, ext_lumins, d_modulus), galax_fab, galax_fab_err),
                    d_modulus)

#@partial(jit, static_argnums=(0,1,2))
@jit
def calc_scale_arrs(f_templ, f_gal, err_gal):
    #_sel1 = jnp.isfinite(f_gal)
    #_sel2 = jnp.isfinite(f_templ)
    #_sel3 = jnp.isfinite(err_gal)
    #_sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
    #_sel = _sel1 * _sel2 * _sel3
    #if len(f_templ[_sel]) > 0 :
        # Scaling as in LEPHARE

    arr_o = f_gal/err_gal
    arr_t = f_templ/err_gal
    #avmago = jnp.sum(arr_o*arr_t)
    #avmagt = jnp.sum(jnp.power(arr_t, 2.))
    return jnp.sum(arr_o*arr_t)/jnp.sum(jnp.power(arr_t, 2.))
    #else:
    #    _scale = 1.
"""
