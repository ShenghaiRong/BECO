## BECO
This is implementation for the reviewed paper: Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation, paper ID: 5681.

### Train
```bash
$ CUDA_VISIBLE_DEVICES=0,1 python main.py --config ./configs/beco.json -dist --logging_tag beco --run_id 1
```
Or use AMP:
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/beco.json -dist --logging_tag beco --run_id 1 --amp
```


### Test
```bash
$ python test.py --crf --logits_dir ./data/logging/beco1/logits --mode "val"
```

