import argparse
from typing import Dict, Any
import multiprocessing
from datasets.voc import VOC, VOCTest
from utils.crf import DenseCRF
import PIL.Image as Image
import torch.nn.functional as F
import joblib
import torch
import numpy as np
import os
from utils.distributed import get_device



class MeanSub:
    def __init__(self, mean_rgb):
        self.mean = np.array(mean_rgb)

    def __call__(self, data: Dict) -> Dict:
        img, label = data['img'], data['label']
        img = np.asarray(img, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)
        img -= self.mean
        if label is not None:
            return {'img': img, 'label': label}
        else:
            return {'img': img}


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def get_args() -> Dict[str, Any]:
    """
    All runtime specific configs should be written here to avoid modifying 
    config file
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--logits_dir", type=str, 
                        default="./data/logging/logits",
                        help="Path of pred logits")
    parser.add_argument("--n-jobs", type=int, 
                        default=multiprocessing.cpu_count(),
                        help="Number of parallel jobs")
    parser.add_argument("--crf", action='store_true', default=False,
                        help="Use crf postprocess")
    parser.add_argument("--mode", default='val', type=str,
                        help="Use crf postprocess")
    return parser.parse_args()

def main():
    args = get_args()
    data_trans = MeanSub(mean_rgb=[122.675, 116.669, 104.008])
    if args.mode == 'test':
        dataset = VOCTest(root='./data/VOC2012', mode=args.mode, 
                          transform=data_trans)
    else:
        dataset = VOC(root='./data/VOC2012/', mode=args.mode, is_aug=False, 
                      transform=data_trans)
    
    base_dir = os.path.abspath(os.path.join(args.logits_dir, '..'))
    test_pred_dir = os.path.join(base_dir, args.mode+'_pred')
    os.makedirs(test_pred_dir, exist_ok=True)
    vis_dir = os.path.join(base_dir, 'vis_'+args.mode)
    os.makedirs(vis_dir, exist_ok=True)

    postprocessor = None
    if args.crf:
        postprocessor = DenseCRF()

    def process(i):
        batch = dataset.__getitem__(i)
        image_id = batch['name']
        image = batch['img']
        gt_label = batch['target']

        filename = os.path.join(args.logits_dir, image_id + ".npy")
        logit = np.load(filename)

        #_, H, W = image.shape
        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", 
                              align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        #image = image.astype(np.uint8).transpose(1, 2, 0)
        image = image.astype(np.uint8)
        if postprocessor is not None:
            prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)
        label_img = Image.fromarray(np.uint8(label))
        label_img.save(os.path.join(test_pred_dir, image_id+'.png'))

        return label, gt_label

        # CRF in multi-process
    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    if args.mode == 'val':
        score = scores(gts, preds, n_class=dataset.num_classes)
        print(f"mIoU: {score['Mean IoU']}")


if __name__ == "__main__":
    main()
    
