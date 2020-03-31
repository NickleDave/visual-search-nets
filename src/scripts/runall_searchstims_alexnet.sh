# script should be run from root of project, paths are written relative to root
alexnet_ini=./data/configs/searchstims_experiments/searchstims_alexnet*ini

for file in ${alexnet_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
