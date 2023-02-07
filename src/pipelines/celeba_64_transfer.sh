#!/bin/bash

# Transfer (adv=0.05 + recon=0.1), attr_vectors_avg_diff. Sens. attr - Pale_Skin

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes High_Cheekbones --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Oval_Face --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"

# Transfer (adv=0.05 + recon=0.1), attr_vectors_avg_diff. Sens. attr - Young

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes High_Cheekbones --perturb Young --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Oval_Face --perturb Young --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Young --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"

# Transfer (adv=0.05 + recon=0.1), attr_vectors_avg_diff. Sens. attr - Brown hair
./celeba_pipelines.sh e2e --classify_attributes Oval_Face --perturb Brown_Hair --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Brown_Hair --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"

# Transfer (adv=0.05 + recon=0.1), attr_vectors_avg_diff. Sens. attr - Bag under eyes
./celeba_pipelines.sh e2e --classify_attributes Oval_Face --perturb Bags_Under_Eyes --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./celeba_pipelines.sh e2e --classify_attributes Wearing_Hat --perturb Bags_Under_Eyes --enc_sigma 0.65 --cls_sigmas "0.5" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.05 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
