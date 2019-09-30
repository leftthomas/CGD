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
The [COCO2017](http://cocodataset.org/#download) dataset is used. Download it and set the path in `configs` directory.

## Usage

### Train
```shell
# single-gpu training
python train.py ${CONFIG_FILE} [--work_dir ${WORK_DIR}] [--resume_from ${CHECKPOINT_FILE}] [--validate] [--autoscale-lr]
# python train.py configs/faster_rcnn_hrnetv2p_w32_1x.py --validate --autoscale-lr

# multi-gpu training
./train.sh ${GPU_NUM} ${PORT} ${CONFIG_FILE} [--work_dir ${WORK_DIR}] [--resume_from ${CHECKPOINT_FILE}] [--validate] [--autoscale-lr]
# ./train.sh 8 29500 configs/faster_rcnn_hrnetv2p_w32_1x.py --validate --autoscale-lr
```

Optional arguments are:
- `WORK_DIR`: Override the working directory specified in the config file.
- `CHECKPOINT_FILE`: Resume from a previous checkpoint file.
- `--validate`: Perform evaluation at every k (default value is 1) epochs during the training.
- `--autoscale-lr`: Automatically scale lr with the number of gpus.

### Test
```shell
# single-gpu testing
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--json_out ${RESULT_JSON_FILE}] [--eval ${EVAL_METRICS}] [--show]
# python test.py configs/faster_rcnn_hrnetv2p_w32_1x.py checkpoints/faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth --json_out results/results

# multi-gpu testing
./test.sh ${GPU_NUM} ${PORT} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--json_out ${RESULT_JSON_FILE}] [--eval ${EVAL_METRICS}]
# ./test.sh 8 29501 configs/faster_rcnn_hrnetv2p_w32_1x.py checkpoints/faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth --out results/results.pkl  --eval bbox
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `RESULT_JSON_FILE`: Filename of the output results without extension in json format. If not specified, the results will 
not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values are: `proposal_fast`, `proposal`, `bbox`, `segm`, `keypoints`.
- `--show`: If specified, detection results will be ploted on the images and shown in a new window. It is only applicable 
to single GPU testing. Please make sure that GUI is available in your environment, otherwise you may encounter the error 
like `cannot connect to X server`.

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2p-W32   | pytorch |   1x    |  39.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth) |
|   HRNetV2p-W32   | pytorch |   2x    |  40.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_2x_20190810-24e8912a.pth) |
