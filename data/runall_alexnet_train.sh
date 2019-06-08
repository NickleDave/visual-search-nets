#!/usr/bin/env bash

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets all ./configs/searchnets_alexnet_train_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_2_v_5.ini
