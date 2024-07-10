# AmbiGANs

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Variation of GANs that, generates realistic data that is classiﬁed with low conﬁdence by an set of classifiers. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.


## Create Virtual Environment

```ssh
mamba create -n ambigan python=3.10
mamba activate ambigan
mamba install pip-tools
```

## Run

### .env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
ENTITY=<wandb entity to track experiments>
```

### Preparation

| Step | Description | Command |
|------|-------------|---------|
| 1    | Create FID score for all pairs of numbers | `python src/gen_pairwise_inception.py` |
| 1.1  | Run for one pair only (e.g. 7v1) | `python -m src.metrics.fid --data data/ --dataset mnist --pos 7 --neg 1` |
| 2.a    | Create set of random classifiers with mean output estimator, given a pair of numbers (e.g. 7vs1) | `python src/gen_classifiers.py --classifier-type ensemble:cnn:mean --pos 7 --neg 1 --epochs 50 --nf 50 --seed 4441 --batch-size 816` |
| 2.b  | Create with linear output estimator | `python src/gen_classifiers.py --classifier-type ensemble:cnn:linear --pos 7 --neg 1 --epochs 50 --nf 50 --seed 4441 --batch-size 816` |
| 2.c  | Create with meta-learner output estimator | `python src/gen_classifiers.py --classifier-type ensemble:cnn:meta-learner --pos 7 --neg 1 --epochs 50 --nf 50 --seed 4441 --batch-size 816` |
| 3    | create test noise | `python src/gen_test_noise.py --nz 2048 --z-dim 64` |

* In this example, in step 2 50 random CNNs will be created and trained during 50 epochs. Reproducibility of the CNN parameters is guaranteed by using the same random seed (--seed arg).

* If the user wants to specify an ensemble architecture, it can be specified in the following way (only for --classifier-type ensemble:cnn):

    ```
    --nf 4-8,8-16-32,20
    ```

    Three CNN are specified: the first with 2 layers. The first layer has 4 filters and the second 8. The second has 3 layers (8, 16 and 32 filters). Finally the last CNN has a single layer with 20 filters.

### AmbiGAN Training

Run AmbiGAN to create images in the bounday between **7** and **1**.

`python -m src --config experiments/ensemble/mnist-7v1_dcgan_v2.yml`

* For the "No-Ouput" variants, user must do step 2.a and specify one of the available experiments in `experiments/ensemble` folder that have the `unique` keyword in the filename (e.g.: `experiments/ensemble/mnist-7v1_unique_dcgan_v2.yml`) - loss type: `kl-div` or `gaussian-v2`.

## Credits


This work started with the framework from previous developments of GASTeN from [luispcunha](https://github.com/luispcunha), published as [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)
