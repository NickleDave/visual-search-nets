STIM_INI_FILES = $(wildcard ./src/configs/searchstims/*.ini)

.PHONY : clean all

variables :
	@echo STIM_INI_FILES: $(STIM_INI_FILES)

clean :
	rm -rf ./data/checkpoints/*
	rm -rf ./data/data_prepd_for_nets
	rm -rf ./data/neural_net_weights/*
	rm -rf ./data/visual_search_stimuli/*
	rm -rf ./results/*

all :
	cd ./src
	bash runall.sh

results :

test :


train :
	searchnets train ./configs/searchnets/config_082918_efficient_10_epochs.ini
	searchnets train ./configs/searchnets/config_082918_efficient_400_epochs.ini
	searchnets train ./configs/searchnets/config_082918_inefficient_10_epochs.ini
	searchnets train ./configs/searchnets/config_082918_inefficient_400_epochs.ini

data :
	searchnets data ./configs/searchnets/config_082918_efficient_10_epochs.ini
	searchnets data ./configs/searchnets/config_082918_efficient_400_epochs.ini
	searchnets data ./configs/searchnets/config_082918_inefficient_10_epochs.ini
	searchnets data ./configs/searchnets/config_082918_inefficient_400_epochs.ini

vis-search-stim : $(STIM_INI_FILES)

./data/data_prepd_for_nets/%.gz : ./configs/searchstims/%.ini
	searchstims ./configs/searchstims/config_number_082818.ini


