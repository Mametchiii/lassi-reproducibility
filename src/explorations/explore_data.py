import argparse

import torch
import torchvision
import numpy as np

from dataset.data_manager import DataManager
from models.gen_model_factory import GenModelFactory

import utils


def glow_reconstructions(dataset: str, attr_vectors_dir: str, perturb, perturb_epsilon, args):
    if 'fairface' in dataset:
        gen_model_name = 'glow_fairface'
    elif 'celeba' in dataset:
        gen_model_name = 'glow_celeba_64'
    else:
        raise ValueError

    params = argparse.Namespace(
        use_cuda=True, batch_size=1, skip=1,
        dataset=dataset, image_size=64, n_bits=5, num_workers=4,
        classify_attributes=[],
        gen_model_type='Glow', gen_model_name=gen_model_name,
        glow_n_flow=32, glow_n_block=4, glow_no_lu=False, glow_affine=False, attr_vectors_dir=attr_vectors_dir,
        perturb=perturb, perturb_epsilon=perturb_epsilon, explore=args.explore, visualization_id=args.visualization_id,
        nr_of_faces = args.nr_of_faces
    )

    device = utils.get_device(params.use_cuda)

    data_manager = DataManager.get_manager(params)
    glow = GenModelFactory.get_and_load(params).to(device)
    attr_vector = None

    def attribute_vector_info(z_delta):
        print('Attribute vector info:')
        z_flattened = []
        print('Shapes:')
        for z_i in z_delta:
            print(z_i.shape)
            z_flattened.append(z_i.view(-1))
        print('Length:')
        z_flattened = torch.cat(z_flattened)
        print(z_flattened.shape)
        print(torch.linalg.norm(z_flattened))
        print(torch.linalg.norm(z_flattened, ord=2))
        print(z_flattened)

    # visualize_id = [1, 129, 193, 257, 321, 385, 449]
    # visualize_id = 52
    if params.visualization_id != '' and params.explore is False:
        id_list = params.visualization_id.split(',')
        id_list = [int(i) for i in id_list]
    else:
        id_list = [np.Inf]

    cnt = 0
    all_imgs = []
    keyword = 'test' if 'fairface' in params.dataset else 'all'
    for x, _ in data_manager.get_dataloader(keyword, shuffle=False):
        cnt += 1
        if cnt > id_list[-1] and params.explore is False:
            break
        if cnt in id_list or params.explore is True:
            print(f'###### visualizing the {cnt}th photo ######')
            x = x.to(device)
            if params.dataset in ['fairface', 'celeba']:
                x = glow.wrap_latents(glow.latents(x))

            if params.skip != 1:
                batch_indices = torch.arange(0, x.size(0), step=params.skip, dtype=torch.long, device=device)
                x = x.index_select(dim=0, index=batch_indices)

            if attr_vector is None:
                attr_vector = glow.get_attribute_vectors(device, x.dtype)
                assert len(attr_vector) == 1
                attr_vector = attr_vector[0]
                attribute_vector_info(attr_vector)
                attr_vector = [z_i.unsqueeze(0) for z_i in attr_vector]
                attr_vector = glow.wrap_latents(attr_vector)
                print(attr_vector.shape)
                print(x.shape)
                assert attr_vector.shape == x.shape

            coeffs = [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]
            if params.nr_of_faces == 5:
                coeffs = [-1.0, -0.5, 0.0, 0.5, 1.0]

            x_recons = []
            for l in coeffs:
                x_recon = glow.reconstruct_from_latents(glow.unwrap_latents(x + l * perturb_epsilon * attr_vector)).cpu()
                x_recons.append(x_recon)
            recons_horizontal = torch.stack(x_recons, dim=1).view(-1, 3, params.image_size, params.image_size)
            if len(id_list) == 1:
                grid = torchvision.utils.make_grid(recons_horizontal, normalize=True, range=(-0.5, 0.5), nrow=len(coeffs))
                utils.show(grid)
                if params.explore is True:
                    inp = input(f'Continue to {cnt} [y/n]?')
                    if inp == 'n':
                        break
                else:
                    break
            elif len(id_list) > 1:
                all_imgs.append(recons_horizontal)

    if all_imgs != []:
        reconstructions_all = torch.stack(all_imgs, dim=0).view(-1, 3, params.image_size, params.image_size)
        grid = torchvision.utils.make_grid(reconstructions_all, normalize=True, range=(-0.5, 0.5), nrow=len(coeffs))
        utils.show(grid)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data exploration (and debugging)")
    parser.add_argument('--dataset', type=str, help="Dataset name, choose in 'celeba' or 'fairface'")
    parser.add_argument('--glow_reconstructions', action='store_true', default=True,
                        help="Show selected Glow reconstructions")
    parser.add_argument('--attr_vectors_dir', type=str, default='attr_vectors_avg_diff',
                        help="Attribute vector directory")
    parser.add_argument('--perturb', help="What attribute to perturb")
    parser.add_argument('--perturb_epsilon', type=float, help="By how much can we perturb?")
    parser.add_argument('--explore', default=False, type=bool, help='set True if you do not know which faces to visualize')
    parser.add_argument('--visualization_id', type=str, default='', help="Id(s) of the faces to visualize, separate the ids with comma")
    parser.add_argument('--nr_of_faces', type=int, default=7, help='number of faces to visualize in the perturbation range')
    params = parser.parse_args()

    utils.init_logger()

    if params.glow_reconstructions:
        glow_reconstructions(params.dataset, params.attr_vectors_dir, params.perturb, params.perturb_epsilon, params)
