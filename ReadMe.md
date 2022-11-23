# BECO
This is implementation for the reviewed paper: Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation, paper ID: 5681.

## Prerequisite
* Python 3.8, PyTorch 1.11.0, and more in requirements.txt
* PASCAL VOC 2012 devkit
* NVIDIA GPU with more than 24GB of memory

## Usage

#### Install python dependencies
```bash
$ pip install -r requirements.txt
```
#### Download PASCAL VOC 2012 devkit
* Download Pascal VOC2012 dataset from the [official dataset homepage](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

#### Download Imagenet pretrained model of DeeplabV3+
* Download Imagenet pretrained model of DeeplabV3+ from [mmclassification](https://github.com/open-mmlab/mmclassification) [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth).
* And rename the downloaded pth as "resnetv1d101_mmcv.pth"

#### Generate pseudo-labels and confidence masks
* Pleaase refer to ./first-stage/irn/README.md for details.
* After generating pseudo-labels and confidence masks, please rename their directories as "irn_pseudo_label" and "irn_mask" respectively.

#### Prepare the data directory
```bash
$ cd BECO-5681/
$ mkdir data
$ mkdir data/model_zoo
$ mkdir data/logging
```
And put the data and pretrained model in the corresponding directories like:
```
data/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClass/
        --- ...
    --- irn_pseudo_label/
        --- ****.png
        --- ****.png
    --- irn_mask/
        --- ****.png
        --- ****.png
    --- model_zoo/
        --- resnetv1d101_mmcv.pth
    --- logging/
```

## Weakly Supervised Semantic Segmentation on VOC2012

### Train
```bash
$ CUDA_VISIBLE_DEVICES=0,1 python main.py --config ./configs/beco.json -dist --logging_tag beco --run_id 1
```
This code also supports AMP acceleration to reduce the GPU memory cost in half. Note that the "batch_size" in ./configs/beco.json refers to the batch_size of per GPU. So you should modify it when using different numbers of GPUs to keep the total batch_size of 16.
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/beco.json -dist --logging_tag beco --run_id 1 --amp
```


### Test
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py --test --config data/logging/beco1/config_\*.json --logging_tag beco1
```
```bash
$ python test.py --crf --logits_dir ./data/logging/beco1/logits --mode "val"
```

