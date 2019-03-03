#!/usr/bin/env bash

source ~/anaconda3/bin/activate searchnets

# create visual search stimuli used to train neural nets
searchstims ./configs/searchstims/config_feature_alexnet.ini
searchstims ./configs/searchstims/config_spatial_config_alexnet.ini
searchstims ./configs/searchstims/config_feature_vgg16.ini
searchstims ./configs/searchstims/config_spatial_config_vgg16.ini

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets all ./configs/searchnets_feature_search_alexnet.ini
searchnets all ./configs/searchnets_feature_search_vgg16.ini
searchnets all ./configs/searchnets_spatial_config_search_alexnet.ini
searchnets all ./configs/searchnets_spatial_config_search_vgg16.ini
