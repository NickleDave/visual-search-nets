.PHONY : clean all

clean :
	rm -rf ./data/checkpoints/*
	rm -rf ./data/data_prepd_for_nets/*
	rm -rf ./data/neural_net_weights/*
	rm -rf ./data/visual_search_stimuli/*
	rm -rf ./results/*

all :
	cd ./src; bash runall.sh

