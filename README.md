# BECO - Official Pytorch Implementation
**Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation**

Shenghai Rong, Bohai Tu, Zilei Wang, Junjie Li

[![](https://img.shields.io/badge/CVPR-2023-blue)](https://cvpr.thecvf.com/Conferences/2023)
[![Paper](https://img.shields.io/badge/Paper-BECO.pdf-red)](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boundary-enhanced-co-training-for-weakly/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=boundary-enhanced-co-training-for-weakly)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boundary-enhanced-co-training-for-weakly/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=boundary-enhanced-co-training-for-weakly)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boundary-enhanced-co-training-for-weakly/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=boundary-enhanced-co-training-for-weakly)


<img src = "https://github.com/ShenghaiRong/BECO-5681/blob/main/figures/framework.png" width="100%" height="100%">

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

#### Download ImageNet pretrained model of DeeplabV3+
* Download ImageNet pretrained [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth) of DeeplabV3+ from [mmclassification](https://github.com/open-mmlab/mmclassification) .
* And rename the downloaded pth as "resnetv1d101_mmcv.pth"

#### Download ImageNet pretrained model of DeeplabV2 and SegFormer (Optional)
* Download ImageNet pretrained [model](https://download.pytorch.org/models/resnet101-cd907fc2.pth) of DeeplabV2 from [pytorch](https://pytorch.org/) .
* And rename the downloaded pth as "resnet-101_v2.pth"

* Download ImageNet pretrained [model](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia) of MiT-B2 from [SegFormer](https://github.com/NVlabs/SegFormer) .

#### Generate pseudo-labels and confidence masks
* Please refer to ./first-stage/irn/README.md for details.
* After generating pseudo-labels and confidence masks, please rename their directories as "irn_pseudo_label" and "irn_mask" respectively.
* The generated irn_pseudo_label and irn_mask are also provided here for reproducing our method more directly. [[Google Drive]](https://drive.google.com/file/d/1zVCZPhJYiOA3TN3dK4cJhzYHKWPUGdEi/view?usp=sharing) / [[Baidu Drive]](https://pan.baidu.com/s/1szP45tRTx_4sZkk-uRF8XQ?pwd=zb21)

#### Prepare the data directory
```bash
$ cd BECO/
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
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -dist --logging_tag beco1
```
This code also supports AMP acceleration to reduce the GPU memory cost in half. Note that the "batch_size" in main.py refers to the batch_size of per GPU. So you should modify it when using different numbers of GPUs to keep the total batch_size of 16.
```bash
$ CUDA_VISIBLE_DEVICES=0,1 python main.py -dist --logging_tag beco1 --amp
```


### Test
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py --test --logging_tag beco1 --ckpt best_ckpt.pth
```

Please refer to [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) to install CRF python library for testing with the CRF post-processing.

```bash
$ python test.py --crf --logits_dir ./data/logging/beco1/logits --mode "val"
```

## Main Results

<img src = "https://github.com/ShenghaiRong/BECO-5681/blob/main/figures/results.png" width="100%" height="100%">

|Method|Dataset|Backbone | Weights| Val mIoU (w/o CRF)|
|:----:|:-----:|:-------:|:------:|:-----------------:|
|BECO| VOC2012 | ResNet101 | [[Google Drive]](https://drive.google.com/file/d/1CzlsfRZz94r2GonErDkgdUph4JgqSbSD/view?usp=sharing) / [[Baidu Drive]](https://pan.baidu.com/s/1cVTXqnE4LisIBOLDMSITqA?pwd=a3g4) | 70.9|
|BECO| COCO2014| ResNet101| [[Google Drive]](https://drive.google.com/file/d/1JqZQ50lMUOJA7Ts5g5eiu4ogakg3KuHG/view?usp=sharing) / [[Baidu Drive]](https://pan.baidu.com/s/1fSp3vpspjUuIYcSVOa_RyQ?pwd=on7s) | 45.6|


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@InProceedings{Rong_2023_CVPR,
    author    = {Rong, Shenghai and Tu, Bohai and Wang, Zilei and Li, Junjie},
    title     = {Boundary-Enhanced Co-Training for Weakly Supervised Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19574-19584}
}
```