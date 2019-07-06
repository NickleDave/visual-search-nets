#!/usr/bin/env bash

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets all ./configs/searchnets_alexnet_train_same_num_samples_per_set_size_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_same_num_samples_per_set_size_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_same_num_samples_per_set_size_2_v_5.ini

searchnets all ./configs/searchnets_alexnet_train_big_set_size_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_size_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_size_2_v_5.ini
