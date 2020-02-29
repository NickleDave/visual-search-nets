#!/usr/bin/env bash

searchnets train ./configs/VSD_alexnet_transfer_lr_1e-03_no_finetune.ini
searchnets test ./configs/VSD_alexnet_transfer_lr_1e-03_no_finetune.ini

searchnets train ./configs/VSD_CORnet_Z_transfer_lr_1e-03_no_finetune.ini
searchnets test ./configs/VSD_CORnet_Z_transfer_lr_1e-03_no_finetune.ini
