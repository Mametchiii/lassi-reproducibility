#!/bin/bash

# - Naive:

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Bald --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Chubby --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Young --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Bald --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Chubby --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Bald --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Chubby --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Bald --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Chubby --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"


# - LASSI (cls + adv):

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Bald --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Chubby --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"


./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Young --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Earrings --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Bald --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Chubby --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Bald --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Chubby --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Attractive --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"

./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Bald --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Big_Lips --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Chubby --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Necklace --perturb Narrow_Eyes --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
