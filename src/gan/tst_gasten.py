import argparse
import os
import math
import torch
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm

from src.metrics import fid, LossSecondTerm
from src.utils import load_z
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint, get_gan_path_at_epoch, construct_gan_from_checkpoint
from src.classifier import ClassifierCache


def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Config file from experiments/clustering folder")
    return parser.parse_args()


"""
mnist-5v3
May23T17-11_y8g2jm9z
- cnn-1-1_20_5
- cnn-2-1_30_10
- cnn-4-1_30_10

mnist-5v3-v2
May24T14-49_8ou500nr
- cnn-1-1_30_10
- cnn-2-1_25_10
- cnn-4-1_30_10

May24T07-16_z2pwytm0
- cnn-1-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_5
- cnn-2-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_10
- cnn-4-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_5

"""

def main(config):

    RUN_ID = "May24T07-16_z2pwytm0"
    CNN1 = "cnn-1-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_5"
    CNN2 = "cnn-2-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_10"
    CNN3 = "cnn-4-1_{'gaussian-v2': {'alpha': 1.0, 'var': 0.005}}_5"

    main_path = f"{os.environ['FILESDIR']}/out/{config['project']}/{config['name']}/{RUN_ID}/"

    classifier_paths = config['train']['step-2']['classifier']
    device = config["device"]
    batch_size = config["train"]["step-2"]["batch-size"]
    num_runs = config['num-runs']

    # final info
    rows = []

    for i in range(num_runs):
        # different seeds
        test_noise, _ = load_z(config['test-noise']+f"_{i}")
    
        # FID preparation
        mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        original_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)
        boundary_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

        for c_path in classifier_paths:

            # load classifier
            C_name = os.path.splitext(os.path.basename(c_path))[0]

            C, _, _, _ = construct_classifier_from_checkpoint(c_path, device=device)
            C.to(device)
            C.eval()
            C.output_feature_maps = True

            class_cache = ClassifierCache(C)
            conf_dist = LossSecondTerm(class_cache)

            fid_metrics = {
                'fid': original_fid,
                'boundary_fid': boundary_fid,
                'conf_dist': conf_dist,
            }

            # load GAN
            if C_name == "cnn-1-1":
                original_gan_cp_dir = main_path+CNN1
            elif C_name == "cnn-2-1":
                original_gan_cp_dir = main_path+CNN2
            else:
                original_gan_cp_dir = main_path+CNN3
                
            gan_path = get_gan_path_at_epoch(original_gan_cp_dir, epoch=40)
            G, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)

            # generate images
            G.eval()

            start_idx = 0 
            num_batches = math.ceil(test_noise.size(0) / batch_size)
            boundary_gen = []

            for _ in tqdm(range(num_batches), desc="Evaluating"):
                real_size = min(batch_size, test_noise.size(0) - start_idx)
                batch_z = test_noise[start_idx:start_idx + real_size]

                with torch.no_grad():
                    batch_gen = G(batch_z.to(device))
                    pred = C(batch_gen)
                    boundary_gen.append(batch_gen[(pred >= 0.4) & (pred <= 0.6)])
        
                fid_metrics['fid'].update(batch_gen, (start_idx, real_size))
                fid_metrics['conf_dist'].update(batch_gen, (start_idx, real_size))

                start_idx += batch_z.size(0)

            #  boundary FID calculation
            boundary_gen_torch = torch.cat(boundary_gen, dim=0) 
            n_images = boundary_gen_torch.size(0)
            # we need at least 2048 images to compute FID
            if n_images >= 2048:
                fid_metrics['boundary_fid'].update_shape(n_images)
                num_batches = math.ceil(n_images / batch_size)
                start_idx = 0
                for _ in tqdm(range(num_batches), desc="Evaluating boundary images"):
                    real_size = min(batch_size, n_images - start_idx)
                    batch_b = boundary_gen_torch[start_idx:start_idx + real_size]
                    fid_metrics['boundary_fid'].update(batch_b, (start_idx, real_size))
                    start_idx += batch_b.size(0)

            row_dict = {}
            row_dict['run'] = i
            row_dict['classifier'] = C_name
            row_dict['n_images'] = n_images
            for metric_name, metric in fid_metrics.items():
                result = metric.finalize()
                row_dict[metric_name] = result
                metric.reset()
            rows.append(row_dict)

    df = pd.DataFrame(rows)
    df.to_csv(f"{os.environ['FILESDIR']}/metrics/{config['project']}/{config['name']}/fid_gaussian_results.csv", index=False)

        
if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config(args.config)
    main(config)
