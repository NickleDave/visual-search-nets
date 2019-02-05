#!/usr/bin/env bash

# create visual search stimuli used to train neural nets
searchstims config_number_082818.ini
searchstims config_rectangle_082818.ini

# prepare data sets for training neural nets
searchnets data config_082918_efficient_10_epochs.ini
searchnets data config_082918_efficient_400_epochs.ini
searchnets data config_082918_inefficient_10_epochs.ini
searchnets data config_082918_inefficient_400_epochs.ini

# train neural nets
searchnets train config_082918_efficient_10_epochs.ini
searchnets train config_082918_efficient_400_epochs.ini
searchnets train config_082918_inefficient_10_epochs.ini
searchnets train config_082918_inefficient_400_epochs.ini
