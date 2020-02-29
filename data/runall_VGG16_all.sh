#!/usr/bin/env bash

searchnets all ./configs/searchnets_VGG16_finetune_RVvGV.ini
searchnets all ./configs/searchnets_VGG16_finetune_RVvRHGV.ini
searchnets all ./configs/searchnets_VGG16_finetune_2_v_5.ini

searchnets all ./configs/searchnets_VGG16_train_RVvGV.ini
searchnets all ./configs/searchnets_VGG16_train_RVvRHGV.ini
searchnets all ./configs/searchnets_VGG16_train_2_v_5.ini

# searchnets all ./configs/searchnets_VGG16_train_RVvGV_RVvRHGV_2_v_5.ini

# searchnets all ./configs/searchnets_VGG16_train_multiple_stims.ini