# script should be run from root of project, paths are written relative to root
CORnet_Z_ini=./data/configs/searchstims_experiments/searchstims_CORnet_Z*ini

for file in ${CORnet_Z_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
