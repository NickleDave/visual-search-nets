# script should be run from root of project, paths are written relative to root
VSD_ini=./data/configs/VSD_*ini

for file in ${VSD_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
