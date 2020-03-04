# CGD
A PyTorch implementation of CGD based on the paper [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663v3).

![Network Architecture image from the paper](results/structure.png)

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
### Train CGD
```
python train.py --feature_dim 512 --gd_config SM
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'resnext50'])
--gd_config                   global descriptors config [default value is 'SG'](choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'])
--feature_dim                 feature dim [default value is 1536]
--smoothing                   smoothing value for label smoothing [default value is 0.1]
--temperature                 temperature scaling used in softmax cross-entropy loss [default value is 0.5]
--margin                      margin of m for triplet loss [default value is 0.1]
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 128]
--num_epochs                  train epoch number [default value is 20]
```

### Test CGD
```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is 'data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_resnet50_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks
The models are trained on one NVIDIA Tesla V100 (32G) GPU with 20 epochs, 
the learning rate is decayed by 10 on 12th and 16th epoch.

### Model Parameters and FLOPs (Params/FLOPs)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>CARS196</th>
      <th>CUB200</th>
      <th>SOP</th>
      <th>In-shop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">26.86M/10.64G</td>
      <td align="center">26.86M/10.64G</td>
      <td align="center">49.85M/10.69G</td>
      <td align="center">34.85M/10.66G</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">26.33M/10.84G</td>
      <td align="center">26.33M/10.84G</td>
      <td align="center">49.32M/10.89G</td>
      <td align="center">34.32M/10.86G</td>
    </tr>
  </tbody>
</table>

### CARS196 (Uncropped/Cropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50(SG)</td>
      <td align="center">86.41%/92.42%</td>
      <td align="center">92.13%/96.10%</td>
      <td align="center">95.55%/97.79%</td>
      <td align="center">97.50%/98.66%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1W3-QKVe5HpCAHJTgxI1M5Q">model</a>&nbsp;|&nbsp;r3sn/<a href="https://pan.baidu.com/s/171Wqa-1TNquzedjlFhaYGg">model</a>&nbsp;|&nbsp;sf5s</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50(SG)</td>
      <td align="center">86.42%/91.66%</td>
      <td align="center">92.02%/95.35%</td>
      <td align="center">95.41%/97.27%</td>
      <td align="center">97.58%/98.57%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1pdp6ePxaxcvGbdlOz1Kmtg">model</a>&nbsp;|&nbsp;dsdx/<a href="https://pan.baidu.com/s/1_dpDM4FNkzPYPvmOsTTR1w">model</a>&nbsp;|&nbsp;fh72</td>
    </tr>
  </tbody>
</table>

### CUB200 (Uncropped/Cropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50(MG)</td>
      <td align="center">66.00%/73.90%</td>
      <td align="center">76.38%/83.12%</td>
      <td align="center">84.81%/89.62%</td>
      <td align="center">90.73%/94.02%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1_Ij-bYHZC31cxEWUnYwqwQ">model</a>&nbsp;|&nbsp;2cfi/<a href="https://pan.baidu.com/s/1deaYb2RWHikztHHsbJyuNw">model</a>&nbsp;|&nbsp;pi4q</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50(MG)</td>
      <td align="center">66.10%/73.73%</td>
      <td align="center">76.32%/82.60%</td>
      <td align="center">84.00%/89.01%</td>
      <td align="center">90.09%/93.32%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1BvhZIBXj9M-Ro9BLmI2lmg">model</a>&nbsp;|&nbsp;nm9h/<a href="https://pan.baidu.com/s/1lu7SYe3tLhp2v1kkI5fO9w">model</a>&nbsp;|&nbsp;6mkf</td>
    </tr>
  </tbody>
</table>

### SOP
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@10</th>
      <th>R@100</th>
      <th>R@1000</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50(SG)</td>
      <td align="center">85.3%</td>
      <td align="center">90.7%</td>
      <td align="center">93.9%</td>
      <td align="center">96.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1_xaiZKwHp3BAp0U1K1ImrQ">model</a>&nbsp;|&nbsp;vsps</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50(SG)</td>
      <td align="center">87.8%</td>
      <td align="center">93.2%</td>
      <td align="center">96.0%</td>
      <td align="center">98.1%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1HCzf6ROjePEyKWs-h3kDsA">model</a>&nbsp;|&nbsp;8588</td>
    </tr>
  </tbody>
</table>

### In-shop
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@10</th>
      <th>R@20</th>
      <th>R@30</th>
      <th>R@40</th>
      <th>R@50</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50(GS)</td>
      <td align="center">78.7%</td>
      <td align="center">93.2%</td>
      <td align="center">95.2%</td>
      <td align="center">96.1%</td>
      <td align="center">96.7%</td>
      <td align="center">97.0%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1yqwTTiGKWnZfkoSZs1LuvQ">model</a>&nbsp;|&nbsp;6dh2</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50(GS)</td>
      <td align="center">87.7%</td>
      <td align="center">96.7%</td>
      <td align="center">97.7%</td>
      <td align="center">98.1%</td>
      <td align="center">98.4%</td>
      <td align="center">98.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1Kf0Tq_q2ODTAp3RXV2_idQ">model</a>&nbsp;|&nbsp;xam8</td>
    </tr>
  </tbody>
</table>

## Results

- CAR/CUB (Uncropped)

![CAR/CUB_Uncropped](results/car_cub.png)

- CAR/CUB (Cropped)

![CAR/CUB_Cropped](results/car_cub_crop.png)

- SOP

![SOP](results/sop.png)

- ISC

![ISC](results/isc.png)