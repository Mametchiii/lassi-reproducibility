#!/bin/bash

# - Naive:

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --enc_sigma 0.325 --cls_sigmas "5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" "$@" 
./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Indian --enc_sigma 0.325 --cls_sigmas "5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Indian --enc_sigma 0.325 --cls_sigmas "0.1" "$@" 

# - LASSI (cls + adv):

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Indian --enc_sigma 0.325 --cls_sigmas "5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Indian --enc_sigma 0.325 --cls_sigmas "0.1" "$@" 