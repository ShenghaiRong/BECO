import torch
import os
import numpy as np
import torchvision
from tqdm import *
import imageio


if __name__ == '__main__':
    score_path = "./test/result/score"
    score_list=os.listdir(score_path)
    mask_root = "./test/result/mask_irn"

    cls_labels_dict=np.load("./voc12/cls_labels.npy",allow_pickle=True).item()

    def checkpath(path):
        if not os.path.exists(path):
            os.mkdir(path)

    checkpath(mask_root)

    img_name=[a.split(".")[0] for a in score_list]

    thd=0.5

    for i in tqdm(img_name):
        mask_path = os.path.join(mask_root, i+".png")
        path=os.path.join(score_path, i+".pt")
        data_i=torch.load(path)
        pred_i = data_i["pred"]

        score_i = data_i["score"].cpu()
        keys = np.unique(pred_i)

        a=score_i
        rw_pred = pred_i
        rw_max=torch.max(a,dim=0)
        rw_min=torch.min(a,dim=0)
        a[a==rw_max[0]]=0
        a_max2=torch.max(a,dim=0)
        sd = rw_max[0]-a_max2[0]
        mask=torch.zeros(rw_pred.shape)
        for key_value in keys:
            l=np.nonzero(rw_pred==key_value)
            ll=sd[l]
            high=torch.sort(ll,descending=True)[1][:int(len(ll)//(1/thd))]
            l_high=(l[0][high.cpu()],l[1][high.cpu()])
            mask[l_high]=1
        torchvision.utils.save_image(mask,mask_path)

