# GASTeN Project

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## Create Virtual Environment

```ssh
mamba create -n gasten python=3.10

mamba activate gasten

mamba install pip-tools
```

## Run

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
ENTITY=<wandb entity to track experiments>
```

### GASTEN ORIGINAL

#### Preparation

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | create FID score for all pairs of numbers                     | `python src/gen_pairwise_inception.py`                                   |
| 1.1  | run for one pair only (e.g. 1vs7)                             | `python -m src.metrics.fid --dataset mnist --pos 7 --neg 1` |
| 2    | create binary classifiers given a pair of numbers (e.g. 1vs7) | `python src/gen_classifiers.py --pos 7 --neg 1 --nf 1,2,4 --epochs 1 --seed 1234`    |
| 3    | create test noise                                             | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                      |

### GASTEN WITH ENSEMBLE CLASSIFIERS

#### Preparation

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | create FID score for all pairs of numbers                     | `python src/gen_pairwise_inception.py`                                   |
| 1.1  | run for one pair only (e.g. 1vs7)                             | `python -m src.metrics.fid --data data/ --dataset mnist --pos 7 --neg 1` |
| 2    | create ensemble classifiers given a pair of numbers (e.g. 1vs7) | `python src/gen_classifiers.py --classifier-type ensemble:cnn --pos 7 --neg 1 --epochs 5 --nf 50 --seed 3333`    |
| 3    | create test noise                                             | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                      |

* In this example, in step 2 50 random CNNs will be created and trained during 5 epochs. Reproducibility of the CNN parameters is guaranteed by using the same random seed (--seed arg)

* For running with pre-training models (RESNET and MLP): --classifier-type ensemble:pretrained. Note --nf 1 in this case.

* If the user wants to specify an ensemble architecture, it can be specified in the following way (only for --classifier-type ensemble:cnn):

    ```
    --nf 4-8,8-16-32,20
    ```

    Three CNN are specified: the first with 2 layers. The first layer has 4 filters and the second 8. The second has 3 layers (8, 16 and 32 filters). Finally the last CNN has a single layer with 20 filters.


### GASTeN

Run GASTeN to create images in the bounday between **1** and **7**.

`python -m src --config experiments/mnist_7v1.yml`


