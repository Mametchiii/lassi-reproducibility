# Reproducibility study on _'Latent Space Smoothing for Individually Fair Representations'_

In their [ECCV 2022](https://www.sri.inf.ethz.ch/publications/peychev2022latent) [paper](https://arxiv.org/abs/2111.13650), Peychev et al. propose a new state-of-the-art representation learning method for enforcing and certifying the individual fairness of high-dimensional data, such as images. In our [paper](https://openreview.net/forum?id=pYBiG82OPLW) we study the reproducibility of their experiments and we aim to verify and extend their main claims. 

This repository contains the implementation of our reproducibility research. The majority of the code is implemented from the [GitHub repository](https://github.com/eth-sri/lassi) of the original paper. To reproduce all the experiments from the original paper, we direct you to their detailed guide on the mentioned GitHub repository.

**In the instructions below** we detail how all our conducted experiments can be reproduced, this includes our reproduced experiments as well as our additonal experiments. This code differs slightly from the original code, mainly to generate visualizations and to test the fairness score of the prediction of selected faces. The main body of code remains largely unedited compared to the original, with only small changes in how class objects are defined for example.  

The project page of this study can be viewed [here](https://mametchiii.github.io/lassi-reproducibility/).

## Environment Setup Instructions

Clone this repository:
```bash
$ git clone https://github.com/eth-sri/lassi.git
$ cd lassi
```

Create a conda
(see [Anaconda](https://www.anaconda.com/distribution/#download-section) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html))
environment with the required packages:
```bash
lassi $ conda env create -f environment.yml
```

Alternatively, we also list the environment dependencies explicitly.
You can create the conda environment and install the packages manually by running the following commands:
```bash
lassi $ conda create --name lassi python=3.8
lassi $ conda activate lassi
(lassi) lassi $ conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
(lassi) lassi $ conda install h5py matplotlib notebook pandas python-lmdb scikit-learn statsmodels tensorboard tqdm
(lassi) lassi $ pip install loguru
(lassi) lassi $ conda deactivate
```

Finally, activate the environment and set the `PYTHONPATH`:
```bash
lassi $ source setup.sh
```

## Pretrained Generative Models

Download the pretrained generative Glow models
(together with the released attribute vectors, if you decide to use them):
```bash
(lassi) lassi $ wget http://files.sri.inf.ethz.ch/lassi/saved_models.tar.gz
(lassi) lassi $ sha256sum saved_models.tar.gz
390fe8c8726f80195b5f1f86fb32ee6eeaedac5edf0c2dd8e48c2f09663784ea  saved_models.tar.gz
(lassi) lassi $ tar -xvzf saved_models.tar.gz
```

## Datasets

We run experiments on the following datasets:
* [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (download the images in the `img_align_celeba.zip` archive)
* [FairFace](https://github.com/joojs/fairface) (use Padding=0.25)

We used the custom random data splits of FairFace defined by the authors of the original paper, if you would like to use them:
```bash
(lassi) lassi $ wget http://files.sri.inf.ethz.ch/lassi/custom_splits.tar.gz
(lassi) lassi $ sha256sum custom_splits.tar.gz
2e279c3048c3e1e2f6d10fdd36279830b3d5b7f6ba84f2a2e2f43cabd1a68a0f  custom_splits.tar.gz
(lassi) lassi $ tar -xvzf custom_splits.tar.gz
```

Manually download the datasets which you would like to run and extract them in a `data/` folder in the top level
of this repository. In the end, `data/` should have the following structure:

```
lassi
├── data
│   ├── celeba
│   │   ├── img_align_celeba
│   │   │   ├── *.jpg images
│   │   ├── list_attr_celeba.txt
│   │   ├── list_eval_partition.txt
│   │   ├── (The rest of the text files may also be needed...
│   │   └── ... for the PyTorch's CelebA dataset class to work properly)
│   ├── fairface
│   │   ├── train
│   │   │   ├── *.jpg images
│   │   ├── val
│   │   │   ├── *.jpg images
│   │   ├── custom_train_split.txt (optional)
│   │   ├── custom_valid_split.txt (optional)
│   │   ├── fairface_label_train.csv
└───└───└── fairface_label_val.csv
```

In order to speed up the training, we cache the image representations in the latent space of the generative models.
We save the cached representations in [LMDB](https://lmdb.readthedocs.io/en/release/) format:
```bash
(lassi) lassi $ cd src/dataset
(lassi) lassi/src/dataset $ ./convert_to_lmdb.sh glow_celeba_64
(lassi) lassi/src/dataset $ ./convert_to_lmdb.sh glow_fairface
```

This will create new versions of the datasets in the `data/` folder:\
`glow_celeba_64_latent_lmdb`,`glow_fairface_latent_lmdb`.

## Computing the Attribute Vectors

You can recompute the attribute vectors we used for our experiments by running the following commands.
Note that due to hardware or library version differences (e.g., different drivers), exact numerical replication
might not be possible.
```bash
(lassi) lassi/src/dataset $ ./compute_attr_vectors_glow.sh celeba64
(lassi) lassi/src/dataset $ ./compute_attr_vectors_glow.sh celeba64 --computation_method perpendicular --epochs 3 --lr 0.001 --normalize_vectors True
(lassi) lassi/src/dataset $ ./compute_attr_vectors_glow.sh celeba64 --computation_method ramaswamy --epochs 3 --lr 0.001 --target Smiling
(lassi) lassi/src/dataset $ ./compute_attr_vectors_glow.sh fairface
```

## Reproducing the Experiments

To run the experiments we chose to reproduce , please run the following scripts.
By default, the experiments will run for 2 random seeds, but this can be controlled by setting
`--run_only_one_seed True`.
```bash
(lassi) lassi $ cd src/pipelines
(lassi) lassi/src/pipelines $ ./celeba_64_avg_diff.sh > celeba_64_avg_diff.out
(lassi) lassi/src/pipelines $ ./celeba_64_perp.sh > celeba_64_perp.out
(lassi) lassi/src/pipelines $ ./celeba_64_ram.sh > celeba_64_ram.out
(lassi) lassi/src/pipelines $ ./celeba_64_transfer.sh > celeba_64_transfer.out
(lassi) lassi/src/pipelines $ ./fairface_experiments.sh > fairface_experiments.out
```

To aggregate the results in a given `.out` file, you can use the `analyse_results.py` script:
```bash
(lassi) lassi/src/pipelines $ python analyse_results.py --out_file [OUT_FILE]
```

## Testing the Fairness score for certain faces (for visualization purpose)

To test if the trained model can predict fairly for a certain face you are interested, please first manually copy the image of that face and save it in the following structure:

```
lassi
├── data
│   ├── visualizations_celeba
|   |   ├──celeba
│   │   │   ├── img_align_celeba
│   │   │   │   ├── *.jpg image
│   │   │   ├── list_attr_celeba.txt (which only contains the label of the selected image)
│   │   │   ├── list_eval_partition.txt
│   │   │   ├── (The rest of the text files may also be needed...
│   │   │   └── ... for the PyTorch's CelebA dataset class to work properly)
│   ├── visualizations_fairface
│   │   ├── val
│   │   │   ├── *.jpg image
└───└───└── fairface_label_val.csv (which only contains the label of the selected image)
```

Then, we cache the image representation of this image, by running the following script:
```bash
(lassi) lassi $ cd src/dataset
(lassi) lassi/src/dataset $ ./convert_to_lmdb.sh glow_visualizations_celeba_64
```

Alternatively, if the face is from the FairFace dataset:
```bash
(lassi) lassi $ cd src/dataset
(lassi) lassi/src/dataset $ ./convert_to_lmdb.sh glow_visualizations_fairface
```

To test this image on a trained LASSI model, use the following commands 
```bash
(lassi) lassi $ cd src/pipelines
# FairFace
(lassi) lassi/src/pipelines $ ./fairface_pipelines.sh visualization --classify_attributes [your selected task] --perturb [your selected sensitive attribute] --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
# CelebA
(lassi) lassi/src/pipelines $ ./celeba_pipelins.sh visualization --classify_attributes [your selected task] --perturb [your selected sensitive attribute] --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --perform_endpoints_analysis False "$@"
```

Alternatively, to test this image on a trained Naive model, these two commands can be used:
```bash
(lassi) lassi $ cd src/pipelines
# FairFace
(lassi) lassi/src/pipelines $ ./fairface_pipelines.sh visualization --classify_attributes [your selected task] --perturb [your selected sensitive attribute] --enc_sigma 0.325 --cls_sigmas "5" "$@"
# CelebA
(lassi) lassi/src/pipelines $ ./celeba_pipelines.sh visualization --classify_attributes [your selected task] --perturb [your selected sensitive attribute] --enc_sigma 0.65 --cls_sigmas "10" --perform_endpoints_analysis False "$@"
```

## Generating Visualizations
The following command is used for making the visualizations of similar faces generated by the GLOW model. You can select the faces that you want to visualize by using the argument `--visualization_id` and include their ids in the dataset. You can also change how many perturbations you want by using the argument `--nr_of_faces`. 

For example, if you want to see how GLOW model generates similar faces for the 16th and 17th faces in the CelebA dataset, along the vector Pale_Face, and you want to see 7 faces on a row, you can run the following command:

```bash
(lassi) lassi $ cd src/explorations
python explore_data.py --dataset celeba --perturb Pale_Skin --perturb_epsilon 1 --visualization_id 16,17 --nr_of_faces 5     
```

## To cite the original paper

```
@inproceedings{peychev2022latent,
    title={Latent Space Smoothing for Individually Fair Representations},
    author={Momchil Peychev and Anian Ruoss and Mislav Balunovi{\'{c}} and Maximilian Baader and Martin Vechev},
    booktitle={Computer Vision -- ECCV 2022},
    year={2022},
    pages={535--554},
    organization={Springer}
}
```

## Contributors of this reproducibility study

* _Contributors of this reproducibility study remain anonymous until after review_
