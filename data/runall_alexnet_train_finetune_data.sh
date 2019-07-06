#!/usr/bin/env bash

# prepare data sets for training neural nets, then train, then test accuracy of trained nets
searchnets train ./configs/searchnets_alexnet_train_finetune_data_RVvGV.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_RVvGV.ini

searchnets train ./configs/searchnets_alexnet_train_finetune_data_RVvRHGV.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_RVvRHGV.ini

searchnets train ./configs/searchnets_alexnet_train_finetune_data_2_v_5.ini
searchnets test ./configs/searchnets_alexnet_train_finetune_data_2_v_5.ini
