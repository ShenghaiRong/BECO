# **Prerequisite of BECO**

## **Prerequisite**

- Python 3.8, PyTorch 1.11.0, and more in [requirements.txt](./requirements.txt) 
- PASCAL VOC 2012 devkit & MS COCO 2014   
- 2 NVIDIA GPUs with more than 1024MB of memory

## Usage (PASCAL VOC 2012)

**Step 1. Prepare dataset** 

Download PASCAL VOC 2012 devkit from [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).
You need to specify the path ('voc12_root') of your downloaded devkit in the following steps (in [run_voc.sh](./run_voc.sh)). 

**Step 2. Generate pseudo-labels and masks** 

```
sh run_voc.sh
```
You can either mannually edit this files ([run_sample.py](./run_sample.py) and [gen_mask.py](./gen_mask.py)), or specify commandline arguments in [run_voc.sh](./run_voc.sh). 


## Usage (MS COCO 2014)

**Step 1. Prepare dataset** 

- Download MS COCO 2014 images from the [official COCO website](https://cocodataset.org/#download). You need to specify the path ('mscoco_root') of your downloaded dataset in the following steps (in [run_coco.sh](./run_coco.sh)).  
- Generate mask from annotations ([annToMask.py](mscoco/annToMask.py) in ./mscoco/). 
- Download MS COCO image-level labels from [here](https://drive.google.com/drive/folders/1XCu51bAUK3nOvO-VVKD7kE9bIFpAECBR?usp=sharing) and put them in ./mscoco/

**Step 2. Generate pseudo-labels and masks** 

Please specify a workspace to save the model and logs in [run_coco.sh](./run_coco.sh).
```
sh run_coco.sh
```
You can either mannually edit this files ([run_sample_coco.py](./run_sample_coco.py) and [gen_mask_coco.py](./gen_mask_coco.py)), or specify commandline arguments in [run_coco.sh](./run_coco.sh). 

## Acknowledgment

This code is borrowed from [IRN](https://github.com/jiwoon-ahn/irn) and [ReCAM](https://github.com/zhaozhengChen/ReCAM). 