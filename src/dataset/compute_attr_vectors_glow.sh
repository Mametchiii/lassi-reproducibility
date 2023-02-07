#!/bin/bash

# Example usage:

# ./compute_attr_vectors_glow.sh celeba64 [--rewrite False] \
#   [--computation_method perpendicular --epochs 3 --lr 0.001 --normalize_vectors True] OR
#   [--computation_method ramaswamy --epochs 3 --lr 0.001 --target Smiling]
# ./compute_attr_vectors_glow.sh celeba64_discover [--rewrite False]
# ./compute_attr_vectors_glow.sh celeba128 [--rewrite False]
# ./compute_attr_vectors_glow.sh fairface [--rewrite False]
# ./compute_attr_vectors_glow.sh 3dshapes [--rewrite False]

dset=$1
shift

case "$dset" in

celeba64)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_celeba_64_latent_lmdb --image_size 64 \
    --attributes "Pale_Skin,Young,Blond_Hair,Bald,Big_Lips,Chubby,Narrow_Eyes" \
    --gen_model_type Glow --gen_model_name "glow_celeba_64" --glow_n_flow 32 --glow_n_block 4 \
    --rewrite True \
    "$@"
  ;;

fairface)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_fairface_latent_lmdb --image_size 64 --classify_attributes Race \
    --attributes "Black,Indian" \
    --gen_model_type Glow --gen_model_name "glow_fairface" --glow_n_flow 32 --glow_n_block 4 \
    --rewrite True \
    "$@"
  ;;


*)
  echo "Dataset mode $dset not recognized and thus not supported"
  ;;

esac
