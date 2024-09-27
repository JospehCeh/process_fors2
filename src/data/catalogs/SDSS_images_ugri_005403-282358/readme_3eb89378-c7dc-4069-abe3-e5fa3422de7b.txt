This file contains an overview of the structure and content of your download request from the ESO Science Archive.


For every downloaded dataset, files are listed below with the following structure:

dataset_name
        - archive_file_name (technical name, as saved on disk)	original_file_name (user name, contains relevant information) category size


Please note that, depending on your operating system and method of download, at download time the colons (:) in the archive_file_name as listed below may be replaced by underscores (_).


In order to rename the files on disk from the technical archive_file_name to the more meaningful original_file_name, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print "test -f",$2,"&& mv",$2,$3}' | sh


In case you have requested cutouts, the file name on disk contains the TARGET name that you have provided as input. To order files by it when listing them, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print $2}' | sort -t_ -k3,3


Your feedback regarding the data quality of the downloaded data products is greatly appreciated. Please contact the ESO Archive Science Group via https://support.eso.org/ , subject: Phase 3 ... thanks!

The download includes contributions from the following collections:
Ref(0)	KIDS	https://doi.eso.org/10.18727/archive/37	release-description-KiDS-ESO-DR4-clean-rev.pdf	https://www.eso.org/rm/api/v1/public/releaseDescriptions/127

Publications based on observations collected at ESO telescopes must acknowledge this fact (please see: http://archive.eso.org/cms/eso-data-access-policy.html#acknowledgement). In particular, please include a reference to the corresponding DOI(s). They are listed in the third column in the table above and referenced below for each dataset. The following shell command lists them:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{print $2, $3}' | sort | uniq


Each collection is described in detail in the corresponding Release Description. They can be downloaded with the following shell command:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{printf("curl -o %s_%s %s\n", $6, $4, $5)}' | sh

ADP.2019-02-11T13:02:24.807_TARGET_00:54:03_-28:23:58 Ref(0)
	- ADP.2019-02-11T13:02:24.807_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_r_sci_TARGET_00:54:03_-28:23:58.fits	SCIENCE.IMAGE	1499284800
	- ADP.2019-02-11T13:02:24.808_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_r_wei_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.WEIGHTMAP	1499279040
	- ADP.2019-02-11T13:02:24.809_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_r_msk_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.MASK	374823360
ADP.2018-12-21T13:00:30.250_TARGET_00:54:03_-28:23:58 Ref(0)
	- ADP.2018-12-21T13:00:30.250_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_g_sci_TARGET_00:54:03_-28:23:58.fits	SCIENCE.IMAGE	1500059520
	- ADP.2018-12-21T13:00:30.251_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_g_wei_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.WEIGHTMAP	1500053760
	- ADP.2018-12-21T13:00:30.252_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_g_msk_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.MASK	375016320
ADP.2018-12-21T12:56:25.448_TARGET_00:54:03_-28:23:58 Ref(0)
	- ADP.2018-12-21T12:56:25.448_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_u_sci_TARGET_00:54:03_-28:23:58.fits	SCIENCE.IMAGE	1458048960
	- ADP.2018-12-21T12:56:25.450_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_u_msk_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.MASK	364512960
	- ADP.2018-12-21T12:56:25.449_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_u_wei_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.WEIGHTMAP	1458043200
ADP.2018-12-21T13:02:11.608_TARGET_00:54:03_-28:23:58 Ref(0)
	- ADP.2018-12-21T13:02:11.608_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_i_sci_TARGET_00:54:03_-28:23:58.fits	SCIENCE.IMAGE	1497029760
	- ADP.2018-12-21T13:02:11.610_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_i_msk_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.MASK	374258880
	- ADP.2018-12-21T13:02:11.609_TARGET_00:54:03_-28:23:58.fits	KiDS_DR4.0_13.5_-28.2_i_wei_TARGET_00:54:03_-28:23:58.fits	ANCILLARY.WEIGHTMAP	1497024000
