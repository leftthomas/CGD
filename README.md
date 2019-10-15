# MBPL
A PyTorch implementation of MBPL based on CVPR 2020 paper 
[MBPL: Multiple Branches with Progressive Learning for KeyPoint Detection](https://arxiv.org/abs/1910.11490). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- opencv
```
pip install opencv-python
```
- pycocotools
```
pip install pycocotools
```
- fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end R-CNN_FPN training with ResNet-50 backbone on 8 GPUs,
one should execute:
```bash
python train_net.py --config-file configs/keypoint_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/keypoint_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS checkpoints/model.pth
```

## COCO Person Keypoint Detection Baselines with Keypoint R-CNN
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">kp.<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: keypoint_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/keypoint_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.315</td>
<td align="center">0.102</td>
<td align="center">5.0</td>
<td align="center">53.6</td>
<td align="center">64.0</td>
<td align="center">137261548</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/metrics.json">metrics</a></td>
</tr>
</tbody></table>

## <a name="CitingTridentNet"></a>Citing TridentNet
If you use TridentNet, please use the following BibTeX entry.

```
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
