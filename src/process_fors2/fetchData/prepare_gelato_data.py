from process_fors2.fetchData import crossmatchToGelato, desi_to_gelato, gogreen_to_gelato, smoothe_gelato

do_fors2 = False
do_desi = False
do_gogreen = True
do_desi_qso = False

if do_fors2:
    print("Prepare FORS2 data for GELATO :")
    f2li, _ = crossmatchToGelato("resulting_merge_from_walkthrough_filtered_cleanGALEX.h5", "prep_gelato_fors2", smoothe=False, nsigma=3, interp_step=None)
    f2lism, _ = smoothe_gelato("prep_gelato_fors2", "prep_gelato_fors2_sm3", nsigma=3, interp_step=0.4)

if do_desi:
    print("Prepare DESI data for GELATO :")
    deli, _ = desi_to_gelato("desi_data_table.h5", "prep_gelato_desi", interp_step=None)
    delism, _ = smoothe_gelato("prep_gelato_desi", "prep_gelato_desi_sm3", nsigma=3, interp_step=0.4)

if do_desi_qso:
    print("Prepare DESI-QSO data for GELATO :")
    deliq, _ = desi_to_gelato("desi_qso_data_table.h5", "prep_gelato_desi_qso", interp_step=None)
    deliqsm, _ = smoothe_gelato("prep_gelato_desi_qso", "prep_gelato_desi_qso_sm3", nsigma=3, interp_step=0.4)

if do_gogreen:
    print("Prepare GOGREEN data for GELATO :")
    gogli, _ = gogreen_to_gelato("gogreen_data_table.h5", "prep_gelato_gogreen", interp_step=None)
    goglism, _ = smoothe_gelato("prep_gelato_gogreen", "prep_gelato_gogreen_sm3", nsigma=3, interp_step=0.4)
