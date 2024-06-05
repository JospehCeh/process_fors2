#!/usr/bin/env python3
"""
Created on Wed Jun  5 09:58:34 2024

@author: joseph
"""

import numpy as np
import pandas as pd


def lim_HII_comp(log_oi_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII et composites : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right) < \
    −1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163$
    - LINERs : $−1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right) < \\log \\left(\frac{\\left[O_I\right]}{H_\alpha} \right) + 0.7$
    - Seyferts : $−1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right)$ et $\\log \\left(\frac{\\left[O_I\right]}{H_\alpha} \right) + 0.7 <\
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right)$

    Parameters
    ----------
    log_oi_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [OI] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : HII and Composites vs.
        Seyferts and LINERs.
    """
    return -1.701 * log_oi_ha - 2.163


def lim_seyf_liner(log_oi_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII et composites : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right) < \
    −1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163$
    - LINERs : $−1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right) < \\log \\left(\frac{\\left[O_I\right]}{H_\alpha} \right) + 0.7$
    - Seyferts : $−1.701 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) − 2.163 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right)$ et $\\log \\left(\frac{\\left[O_I\right]}{H_\alpha} \right) + 0.7 <\
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[O_{II}\right]} \right)$

    Parameters
    ----------
    log_oi_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [OI] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : LINERs vs. Seyferts.
    """
    return 1.0 * log_oi_ha + 0.7


def Ka03_nii(log_nii_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.05} + 1.3$ (Ka03)
    - Composites : (Ka03) $\frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.05} + 1.3 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.47} + 1.19$ (Ke01)

    Parameters
    ----------
    log_nii_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [NII] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : HII vs. Composites,
        Seyferts and LINERs.
    """
    return 0.61 / (log_nii_ha - 0.05) + 1.3


def Ke01_nii(log_nii_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.05} + 1.3$ (Ka03)
    - Composites : (Ka03) $\frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.05} + 1.3 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.61}{\\log \\left( \frac{\\left[N_{II}\right]}{H_\alpha} \right) − 0.47} + 1.19$ (Ke01)

    Parameters
    ----------
    log_nii_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [NII] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : HII and Composites vs.
        Seyferts and LINERs.
    """
    return 0.61 / (log_nii_ha - 0.47) + 1.19


def Ke01_sii(log_sii_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30$ (Ke01)
    - Seyfert : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    (Kw06) $1.89 \\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) + 0.76 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$
    - LINER : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) > \
    1.89 \\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) + 0.76$ (Ke06)

    Parameters
    ----------
    log_sii_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [SII] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : HII vs. Seyferts and LINERs.
    """
    return 0.72 / (log_sii_ha - 0.32) + 1.30


def Ke06_sii(log_sii_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \
    \frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30$ (Ke01)
    - Seyfert : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    (Kw06) $1.89 \\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) + 0.76 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$
    - LINER : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) > \
    1.89 \\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) + 0.76$ (Ke06)

    Parameters
    ----------
    log_sii_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [SII] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : LINERs vs. Seyferts.
    """
    return 1.89 * log_sii_ha + 0.76


def Ke01_oi(log_oi_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) < \frac{0.73}{\\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 0.59} + 1.33$ (Ke01)
    - Seyfert : (Ke01) $\frac{0.73}{\\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 0.59} + 1.33 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    (Kw06) $1.18 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$
    - LINER : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) > \
    1.18 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 1.30$ (Ke06)

    Parameters
    ----------
    log_oi_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [OI] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : HII vs. Seyferts + LINERs.
    """
    return 0.73 / (log_oi_ha + 0.59) + 1.33


def Ke06_oi(log_oi_ha):
    """
    Critère de [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract):
    - HII (star-forming) : $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) <\
    \frac{0.73}{\\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 0.59} + 1.33$ (Ke01)
    - Seyfert : (Ke01) $\frac{0.73}{\\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 0.59} + 1.33 <\
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    Kw06) $1.18 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$
    - LINER : (Ke01) $\frac{0.72}{\\log \\left( \frac{\\left[S_{II}\right]}{H_\alpha} \right) − 0.32} + 1.30 < \
    \\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right)$ et
    $\\log \\left( \frac{ \\left[O_{III}\right]}{\\left[H_{\beta}\right]} \right) > \
    1.18 \\log \\left( \frac{\\left[O_{I}\right]}{H_\alpha} \right) + 1.30$ (Ke06)

    Parameters
    ----------
    log_oi_ha : float or numpy array
        Decimal logarithm of the amplitude ratio of spectral bands [OI] and H$_\alpha$, often derived from equivalent widths.

    Returns
    -------
    float or numpy array
        The value that classifies galaxies into two kinds depending on whether they are below or above this limit : LINERs vs. Seyferts.
    """
    return 1.18 * log_oi_ha + 1.30


def bpt_classif(gelatoh5, xmatchh5, use_nc=False, return_dict=False):
    """
    Use Restframe Equivalent Widths from GELATO outputs to provide an rudimentary classification of galaxies, using BPT diagrams as described in
    [Kewley et al., 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K/abstract).

    Parameters
    ----------
    gelatoh5 : str or path
        Name or path to the `HDF5` file that contains GELATO outputs.
    xmatchh5 : str or path
        Name or path to the `HDF5` file that contains cross-matched data.
    use_nc : bool, optional
        Whether to use the 'NC' value for amibuous classifications instead of the highest score. The default is False.
    return_dict : bool, optional
        Whether to return the results as a dictionary (similar to `process_fors2.fetchData.gelato_xmatch_todict`) or a DataFrame. The default is False.

    Returns
    -------
    Object
        Return the merged outputs of DSPS and GELATO + classification info. As a dictionary if `return_dict` is `True`, otherwise as a Pandas DataFrame.
    """
    from process_fors2.fetchData import readH5FileAttributes

    gelatout = readH5FileAttributes(gelatoh5)
    xmatchout = readH5FileAttributes(xmatchh5)
    res_table = xmatchout.merge(right=gelatout, how="outer", on=["name", "num"])
    res_table["u-g"] = res_table["MAG_GAAP_u"] - res_table["MAG_GAAP_g"]
    res_table["r-i"] = res_table["MAG_GAAP_r"] - res_table["MAG_GAAP_i"]

    res_table["log([OIII]/[Hb])"] = np.log10(res_table["AGN_[OIII]_5008.24_REW"] / res_table["Balmer_HI_4862.68_REW"])
    res_table["log([NII]/[Ha])"] = np.log10(res_table["AGN_[NII]_6585.27_REW"] / res_table["Balmer_HI_6564.61_REW"])
    res_table["log([SII]/[Ha])"] = np.log10(res_table["AGN_[SII]_6718.29_REW"] / res_table["Balmer_HI_6564.61_REW"])
    res_table["log([OI]/[Ha])"] = np.log10(res_table["SF_[OI]_6302.046_REW"] / res_table["Balmer_HI_6564.61_REW"])
    res_table["log([OIII]/[OII])"] = np.log10(res_table["AGN_[OIII]_5008.24_REW"] / res_table["SF_[OII]_3728.48_REW"])

    cat_nii = []
    for x, y in zip(res_table["log([NII]/[Ha])"], res_table["log([OIII]/[Hb])"], strict=False):
        if not (np.isfinite(x) and np.isfinite(y)):
            cat_nii.append("NC")
        elif y < Ka03_nii(x):
            cat_nii.append("Star-forming")
        elif y < Ke01_nii(x):
            cat_nii.append("Composite")
        else:
            cat_nii.append("AGN")

    res_table["CAT_NII"] = np.array(cat_nii)

    cat_sii = []
    for x, y in zip(res_table["log([SII]/[Ha])"], res_table["log([OIII]/[Hb])"], strict=False):
        if not (np.isfinite(x) and np.isfinite(y)):
            cat_sii.append("NC")
        elif y < Ke01_sii(x):
            cat_sii.append("Star-forming")
        elif y < Ke06_sii(x):
            cat_sii.append("LINER")
        else:
            cat_sii.append("Seyferts")

    res_table["CAT_SII"] = np.array(cat_sii)

    cat_oi = []
    for x, y in zip(res_table["log([OI]/[Ha])"], res_table["log([OIII]/[Hb])"], strict=False):
        if not (np.isfinite(x) and np.isfinite(y)):
            cat_oi.append("NC")
        elif y < Ke01_oi(x):
            cat_oi.append("Star-forming")
        elif y < Ke06_oi(x):
            cat_oi.append("LINER")
        else:
            cat_oi.append("Seyferts")

    res_table["CAT_OI"] = np.array(cat_oi)

    cat_oii = []
    for x, y in zip(res_table["log([OI]/[Ha])"], res_table["log([OIII]/[OII])"], strict=False):
        if not (np.isfinite(x) and np.isfinite(y)):
            cat_oii.append("NC")
        elif y < lim_HII_comp(x):
            cat_oii.append("SF / composite")
        elif y < lim_seyf_liner(x):
            cat_oii.append("LINER")
        else:
            cat_oii.append("Seyferts")

    res_table["CAT_OIII/OIIvsOI"] = np.array(cat_oii)

    mask = np.logical_and(res_table["CAT_NII"] == "NC", np.logical_and(res_table["CAT_SII"] == "NC", np.logical_and(res_table["CAT_OI"] == "NC", res_table["CAT_OIII/OIIvsOI"] == "NC")))
    classif_table = res_table[np.logical_not(mask)]
    nc_table = res_table[mask]
    nc_table["Weight SF"] = 0.0
    nc_table["Weight Composite"] = 0.0
    nc_table["Weight Seyferts"] = 0.0
    nc_table["Weight LINER"] = 0.0
    nc_table["Weight NC"] = 4.0
    nc_table["Classification"] = "NC"

    if use_nc:
        w_sf = []
        w_nc = []
        w_comp = []
        w_seyf = []
        w_liner = []
        classifs = []
        for i, row in classif_table.iterrows():
            pt_sf, pt_nc, pt_comp, pt_seyf, pt_liner = 0, 0, 0, 0, 0

            val = row["CAT_NII"]
            if val == "NC":
                pt_nc += 0.5
            elif val == "Star-forming":
                pt_sf += 1
            elif val == "Composite":
                pt_comp += 1
            else:
                assert val == "AGN", "Unregistered [NII] classification."
                pt_seyf += 0.5
                pt_liner += 0.5

            val = row["CAT_SII"]
            if val == "NC":
                pt_nc += 0.5
            elif val == "Star-forming":
                pt_sf += 0.7
                pt_comp += 0.3
            elif val == "Seyferts":
                pt_seyf += 1
            else:
                assert val == "LINER", "Unregistered [SII] classification."
                pt_liner += 1

            val = row["CAT_OI"]
            if val == "NC":
                pt_nc += 0.5
            elif val == "Star-forming":
                pt_sf += 0.7
                pt_comp += 0.3
            elif val == "Seyferts":
                pt_seyf += 1
            else:
                assert val == "LINER", "Unregistered [OI] classification."
                pt_liner += 1

            val = row["CAT_OIII/OIIvsOI"]
            if val == "NC":
                pt_nc += 0.5
            elif val == "SF / composite":
                pt_sf += 0.5
                pt_comp += 0.5
            elif val == "Seyferts":
                pt_seyf += 1
            else:
                assert val == "LINER", "Unregistered [OIII]/[OII] vs [OI] classification."
                pt_liner += 1

            p_tot = pt_sf + pt_nc + pt_comp + pt_seyf + pt_liner
            pt_sf /= p_tot
            pt_nc /= p_tot
            pt_comp /= p_tot
            pt_seyf /= p_tot
            pt_liner /= p_tot
            w_sf.append(pt_sf)
            w_nc.append(pt_nc)
            w_comp.append(pt_comp)
            w_seyf.append(pt_seyf)
            w_liner.append(pt_liner)

            classid = np.nanargmax(np.array([pt_sf, pt_comp, pt_seyf, pt_liner, pt_nc]))
            classif = ["Star-forming", "Composite", "Seyferts", "LINER", "NC"][classid]
            classifs.append(classif)

        classif_table["Weight SF"] = np.array(w_sf)
        classif_table["Weight Composite"] = np.array(w_comp)
        classif_table["Weight Seyferts"] = np.array(w_seyf)
        classif_table["Weight LINER"] = np.array(w_liner)
        classif_table["Weight NC"] = np.array(w_nc)
        classif_table["Classification"] = np.array(classifs)
    else:
        w_sf = []
        w_comp = []
        w_seyf = []
        w_liner = []
        classifs = []
        for i, row in classif_table.iterrows():
            pt_sf, pt_comp, pt_seyf, pt_liner = 0, 0, 0, 0

            val = row["CAT_NII"]
            if val == "Star-forming":
                pt_sf += 1
            elif val == "Composite":
                pt_comp += 1
            elif val == "AGN":
                pt_seyf += 0.5
                pt_liner += 0.5

            val = row["CAT_SII"]
            if val == "Star-forming":
                pt_sf += 0.7
                pt_comp += 0.3
            elif val == "Seyferts":
                pt_seyf += 1
            elif val == "LINER":
                pt_liner += 1

            val = row["CAT_OI"]
            if val == "Star-forming":
                pt_sf += 0.7
                pt_comp += 0.3
            elif val == "Seyferts":
                pt_seyf += 1
            elif val == "LINER":
                pt_liner += 1

            val = row["CAT_OIII/OIIvsOI"]
            if val == "SF / composite":
                pt_sf += 0.5
                pt_comp += 0.5
            elif val == "Seyferts":
                pt_seyf += 1
            elif val == "LINER":
                pt_liner += 1

            p_tot = pt_sf + pt_comp + pt_seyf + pt_liner
            pt_sf /= p_tot
            pt_comp /= p_tot
            pt_seyf /= p_tot
            pt_liner /= p_tot
            w_sf.append(pt_sf)
            w_comp.append(pt_comp)
            w_seyf.append(pt_seyf)
            w_liner.append(pt_liner)

            classid = np.nanargmax(np.array([pt_sf, pt_comp, pt_seyf, pt_liner]))
            classif = ["Star-forming", "Composite", "Seyferts", "LINER"][classid]
            classifs.append(classif)

        classif_table["Weight SF"] = np.array(w_sf)
        classif_table["Weight Composite"] = np.array(w_comp)
        classif_table["Weight Seyferts"] = np.array(w_seyf)
        classif_table["Weight LINER"] = np.array(w_liner)
        classif_table["Classification"] = np.array(classifs)
    merged_df = pd.concat((nc_table, classif_table), axis=0, verify_integrity=True)
    merged_df.set_index("name", drop=False, inplace=True)
    merged_df.sort_values("num", inplace=True)
    if return_dict:
        merged_attrs = merged_df.to_dict(orient="index")
        return merged_attrs
    else:
        return merged_df
