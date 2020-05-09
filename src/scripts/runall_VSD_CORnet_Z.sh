# script should be run from root of project, paths are written relative to root
cornet_z_ini=./data/configs/VSD/VSD_CORnet_Z*ini

for file in ${cornet_z_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
