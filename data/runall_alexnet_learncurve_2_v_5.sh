#!/usr/bin/env bash

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets data ./configs/searchnets_alexnet_learncurve_2_v_5.ini
searchnets learncurve ./configs/searchnets_alexnet_learncurve_2_v_5.ini
