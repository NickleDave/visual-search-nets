.PHONY : clean all tests-clean tests-all

clean :
	rm -rf ./data/checkpoints/*
	rm -rf ./data/data_prepd_for_nets/*
	rm -rf ./data/neural_net_weights/*
	rm -rf ./data/visual_search_stimuli/*
	rm -rf ./results/*

all :
	cd ./src; bash runall.sh

tests-clean :
	rm test_data/visual_search_stimuli/*

tests-all :
	searchstims ./tests/test_data/configs/config_feature_alexnet.ini

