VSD_ini=./configs/VSD_*ini

for file in ${VSD_ini};do
	searchnets train ${file}
	searchnets test ${file}
done
