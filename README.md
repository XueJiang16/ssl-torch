# ssl-torch

This is the code implementation of paper  *Self-supervised Contrastive Learning for EEG-based Sleep Staging*.

### Environment Setup 

We recommend to setup the environment through `conda`.

```shell
$ conda env create -f environment.yml
```

### Data Preparation

The dataset Sleep-edf can be downloaded [here](https://physionet.org/content/sleep-edfx/1.0.0/).

### Training

We use Pytorch 3.6 to build the network, which is trained on the NVIDIA GTX 1080Ti with the batch size of 128. The network is trained for 70 epochs. We use the SGD optimizer with the momentum $= 0.9$.  

For training the network, run

```shell
$ python contrast.py -F1 time_warp -F2 time_warp
```


