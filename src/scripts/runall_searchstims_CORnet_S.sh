# script should be run from root of project, paths are written relative to root
CORnet_S_ini=./data/configs/searchstims/searchstims_CORnet_S*ini

for file in ${CORnet_S_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
