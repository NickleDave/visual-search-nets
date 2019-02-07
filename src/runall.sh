#!/usr/bin/env bash

source ~/anaconda3/bin/activate searchnets

# create visual search stimuli used to train neural nets
searchstims ./configs/searchstims/config_number_082818.ini
searchstims ./configs/searchstims/config_rectangle_082818.ini

# prepare data sets for training neural nets
searchnets data ./configs/searchnets/config_082918_efficient_10_epochs.ini
searchnets data ./configs/searchnets/config_082918_efficient_400_epochs.ini
searchnets data ./configs/searchnets/config_082918_inefficient_10_epochs.ini
searchnets data ./configs/searchnets/config_082918_inefficient_400_epochs.ini

# train neural nets
searchnets train ./configs/searchnets/config_082918_efficient_10_epochs.ini
searchnets train ./configs/searchnets/config_082918_efficient_400_epochs.ini
searchnets train ./configs/searchnets/config_082918_inefficient_10_epochs.ini
searchnets train ./configs/searchnets/config_082918_inefficient_400_epochs.ini

# test
searchnets test ./configs/searchnets/config_082918_efficient_10_epochs.ini
searchnets test ./configs/searchnets/config_082918_efficient_400_epochs.ini
searchnets test ./configs/searchnets/config_082918_inefficient_10_epochs.ini
searchnets test ./configs/searchnets/config_082918_inefficient_400_epochs.ini

