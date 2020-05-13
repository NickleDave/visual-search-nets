# script should be run from root of project, paths are written relative to root
VSD_CORnet_S_ini=./data/configs/VSD/VSD_CORnet_S*ini

for file in ${VSD_CORnet_S_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
