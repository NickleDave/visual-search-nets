ALEXNET_WEIGHTS_URL=https://ndownloader.figshare.com/files/14299136
ALEXNET_WEIGHTS_DST=./data/neural_net_weights/bvlc_alexnet.npy

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
	wget $(ALEXNET_WEIGHTS_URL) -O $(ALEXNET_WEIGHTS_DST)
	searchstims ./tests/test_data/configs/config_feature_alexnet.ini

