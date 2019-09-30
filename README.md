# AHRNet
A PyTorch implementation of AHRNet based on CVPR 2019 paper 
[AHRNet: Attentive High Resolution Network for KeyPoint Detection](https://arxiv.org/abs/1904.11490). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- mmdetection
```
python setup.py develop
```
## Datasets
The [COCO2017](http://cocodataset.org/#download) dataset is used.
Download it and set the path in `configs` directory.

## Usage

### Train
```
./train.sh 8 29500 configs/reppoints_moment_x101_dcn_fpn_2x_mt.py --validate --gpus 8 --autoscale-lr
```

### Test
```
./test.sh 8 29501 configs/reppoints_moment_x101_dcn_fpn_2x_mt.py checkpoints/reppoints_moment_x101_dcn_fpn_2x_mt.pth --out results/results.pkl --eval bbox
```

## Results

The results on COCO 2017val are shown in the table below.

| Method | Backbone | convert func | Lr schd | box AP | Download |
| :----: | :------: | :-------: | :-----: | :----: | :------: |
| RepPoints | X-101-FPN-DCN | moment | 2x (ms train)   | 45.6| [model](https://drive.google.com/open?id=1nr9gcVWxzeakbfPC6ON9yvKOuLzj_RrJ) |
| RepPoints | X-101-FPN-DCN | moment | 2x (ms train&ms test)   | 46.8|          |

**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively. 
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `moment`, `partial MinMax`, `MinMax` in the `convert func` column are three functions to convert a point set to a 
pseudo box.
- `ms` denotes multi-scale training or multi-scale test.
