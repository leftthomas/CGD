# AntiMetric
A PyTorch implementation of AntiMetric based on ICCV 2021 paper [AntiMetric: Anti-metric method for computer vision]().

<div align="center">
  <img src="results/architecture.png"/>
</div>

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Dataset
`ImageNet` dataset is used in this repo, change the `data_path` in `utils.py`.

## Usage
### Train Model
```
python train.py --num_epochs 50 --load_ids
optional arguments:
--with_random                 with branch random weight or not [default value is False]
--load_ids                    load already generated ids or not [default value is False]
--batch_size                  train batch size [default value is 10]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
```
