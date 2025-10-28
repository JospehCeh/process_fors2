from process_fors2.fetchData import gelatoToH5

do_fors2 = True
do_fors2_sm3 = True
do_gogreen = True
do_gogreen_sm3 = True
do_desi = True
do_desi_sm3 = True
do_desi_qso = True
do_desi_qso_sm3 = True

if do_fors2:
    _ = gelatoToH5("resGelato_fors2.h5", "prep_gelato_fors2", source="FORS2")
if do_fors2_sm3:
    _ = gelatoToH5("resGelato_fors2_sm3.h5", "prep_gelato_fors2_sm3", source="FORS2")

if do_gogreen:
    _ = gelatoToH5("resGelato_gogreen.h5", "prep_gelato_gogreen", source="GOGREEN")
if do_gogreen_sm3:
    _ = gelatoToH5("resGelato_gogreen_sm3.h5", "prep_gelato_gogreen_sm3", source="GOGREEN")

if do_desi:
    _ = gelatoToH5("resGelato_desi.h5", "prep_gelato_desi", source="DESI")
if do_desi_sm3:
    _ = gelatoToH5("resGelato_desi_sm3.h5", "prep_gelato_desi_sm3", source="DESI")

if do_desi_qso:
    _ = gelatoToH5("resGelato_desi_qso.h5", "prep_gelato_desi_qso", source="DESI")
if do_desi_qso_sm3:
    _ = gelatoToH5("resGelato_desi_qso_sm3.h5", "prep_gelato_desi_qso_sm3", source="DESI")
