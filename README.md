# T-Autoencoder-KL

This project implements a Transformer-based Autoencoder-KL.

## Dependencies

```xml
- python >= 3.8
- pytorch >= 1.13
- torchvision
- pytorch-cuda=11.7
- pip
- pip:
- timm
- diffusers
- lmdb
- accelerate
- taming-transformers
- einops
- matplotlib
- pytorch_lightning
```

## How to run

For training T-Autoencoder-KL based Latent Diffusions on LSUN dataset, run:

```bash
python train_with_kl.py
```

For sampling, run:

```bash
python sample_with_kl.py
```

For training T-VQ-VAE based Latent Diffusions on LSUN dataset, run:

```bash
python train_with_vq.py
```

For sampling, run:

```bash
python sample_with_vq.py
```

For training Autoencoder-KL based Latent Diffusions on LSUN dataset, run:

```bash
python train_with_ae.py
```

For comparing all autoenocoders, run:

```bash
python compare_ae.py
```

For calculating the fid score, run:

```bash
python fid.py
```

