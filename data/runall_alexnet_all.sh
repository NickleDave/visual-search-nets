#!/usr/bin/env bash

searchnets all ./configs/searchnets_alexnet_finetune_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_finetune_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_finetune_2_v_5.ini

searchnets all ./configs/searchnets_alexnet_train_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_2_v_5.ini

# do not want to re-make finetune data
searchnets train ./configs/searchnets_alexnet_train_finetune_data_RVvGV.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_RVvGV.ini
searchnets train ./configs/searchnets_alexnet_train_finetune_data_RVvRHGV.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_RVvRHGV.ini
searchnets train ./configs/searchnets_alexnet_train_finetune_data_2_v_5.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_2_v_5.ini

searchnets all ./configs/searchnets_alexnet_train_big_set_size_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_size_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_size_2_v_5.ini

searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_RVvGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_RVvRHGV.ini
searchnets all ./configs/searchnets_alexnet_train_big_set_and_sample_size_2_v_5.ini
