#!/usr/bin/env bash

source ~/anaconda3/bin/activate searchnets

# create visual search stimuli used to train neural nets
searchstims ./configs/searchstims/config_efficient.ini
searchstims ./configs/searchstims/config_inefficient.ini

# prepare data sets for training neural nets
# only need to do this once for efficient + inefficient stimuli
searchnets data ./configs/searchnets/config_efficient_10_epochs.ini
searchnets data ./configs/searchnets/config_inefficient_10_epochs.ini

# train neural nets
searchnets train ./configs/searchnets/config_efficient_10_epochs.ini
searchnets train ./configs/searchnets/config_efficient_400_epochs.ini
searchnets train ./configs/searchnets/config_inefficient_10_epochs.ini
searchnets train ./configs/searchnets/config_inefficient_400_epochs.ini

# test
searchnets test ./configs/searchnets/config_efficient_10_epochs.ini
searchnets test ./configs/searchnets/config_efficient_400_epochs.ini
searchnets test ./configs/searchnets/config_inefficient_10_epochs.ini
searchnets test ./configs/searchnets/config_inefficient_400_epochs.ini

