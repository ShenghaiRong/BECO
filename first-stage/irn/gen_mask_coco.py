import torch
import os
import numpy as np
import torchvision
from tqdm import tqdm


if __name__ == '__main__':
    score_path = "./ReCAM/test_coco/irn/score"
    score_list=os.listdir(score_path)
    mask_root = "./ReCAM/test_coco/irn/mask_irn"

    if not os.path.exists(mask_root):
        os.mkdir(mask_root)

    img_name=[a.split(".")[0] for a in score_list]

    for i in tqdm(img_name):
        mask_path = os.path.join(mask_root, i+".png")
        if os.path.exists(mask_path):
            continue
        path=os.path.join(score_path, i+".pt")
        data_i=torch.load(path)
        pred_i = data_i["pred"]
        score_i = data_i["score"]
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
            high=torch.sort(ll,descending=True)[1][:int(len(ll)//(1/0.5))]
            l_high=(l[0][high.cpu()],l[1][high.cpu()])
            mask[l_high]=1
        torchvision.utils.save_image(mask,mask_path)


    import random
    check_path="./ReCAM/test_coco/irn/check"
    mask_root="./ReCAM/test_coco/irn/mask_irn"
    check_list=os.listdir(check_path)
    for i in check_list:
        a=torch.load(os.path.join(check_path,i))
        name=a["name"]
        mask_path=os.path.join(mask_root,name+".png")
        size=a["cam"].shape[1:]
        mask = torch.zeros(size,dtype=torch.long)
        img_size = mask.shape
        mask =  mask.reshape(-1)
        num_size = mask.shape[0]
        rand_n=list(range(num_size))
        random.shuffle(rand_n)
        mask[rand_n[:num_size//2]]=1
        mask=mask.reshape(img_size)
        torchvision.utils.save_image(mask.float(),mask_path)

