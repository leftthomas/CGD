# CGD
A PyTorch implementation of CGD based on the paper [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663v3).

![Network Architecture image from the paper](structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
[Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and 
[In-shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

You should download these datasets by yourself, and extract them into `${data_path}` directory, make sure the dir names are 
`car`, `cub`, `sop` and `isc`. Then run `data_utils.py` to preprocess them.

## Usage
### Train model
```
python train.py --crop_h 512 --crop_w 1024
optional arguments:
--data_path                   Data path for cityscapes dataset [default value is '/home/data/cityscapes']
--crop_h                      Crop height for training images [default value is 1024]
--crop_w                      Crop width for training images [default value is 2048]
--batch_size                  Number of data for each batch to train [default value is 12]
--save_step                   Number of steps to save predicted results [default value is 5]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
```