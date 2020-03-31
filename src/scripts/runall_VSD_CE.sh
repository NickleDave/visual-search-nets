# re-run just VSD experiments with cross-entropy loss
# script should be run from root of project, paths are written relative to root
VSD_ini=./data/configs/VSD/VSD_*_transfer_CE_*ini

for file in ${VSD_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
