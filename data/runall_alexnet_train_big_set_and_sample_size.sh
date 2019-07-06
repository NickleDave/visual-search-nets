#!/usr/bin/env bash

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_2_v_5.ini
