from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast

from utils.logger import R0Logger
from models.base.base_simseg import BaseSimSeg
import numpy as np
import torch.nn.functional as F
import os
from datasets.transforms.imgmix import imgmix_multi_withmask_bdry
from datasets.transforms.imgmix import imgmix_multi_withmaskol_bdry
from datasets.transforms.transform import augment_withmask



class BECO(BaseSimSeg):

    def __init__(self, 
        ignore_bg,
        mix_aug,
        mix_prob,
        bdry_size,
        bdry_whb,
        bdry_whi,
        bdry_wlb,
        bdry_wli,
        warm_up,
        highres_t,
        save_logits,
        test_msc,
        logging_path,
        config,
        nets_dict) -> None:
        super().__init__(config, nets_dict)
        self.logger = R0Logger(__name__)
        self.init_loss()
        self.ignore_bg = ignore_bg
        self.mix_aug = mix_aug
        self.mix_prob = mix_prob
        self.warm_up = warm_up
        self.bdry_size = bdry_size
        self.bdry_whb = bdry_whb
        self.bdry_whi = bdry_whi
        self.bdry_wlb = bdry_wlb
        self.bdry_wli = bdry_wli
        self.T = highres_t
        self.save_logits = save_logits
        self.test_msc = test_msc
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        if self.save_logits:
            self.logging_path = logging_path
            if self.test_msc:
                self.logits_path = os.path.join(self.logging_path, 'logits_msc')
            else:
                self.logits_path = os.path.join(self.logging_path, 'logits')
            os.makedirs(self.logits_path, exist_ok=True)

    def init_loss(self):
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    

    def train_step(self, batch: Dict[str, Any]) -> Tuple[Tensor, float]:
        images, labels, masks = batch['img'], batch['target'], batch['mask']

        if self.epoch >= self.warm_up:
            images_cuda = self.scatter((images.detach()))
            with autocast(enabled=self.is_amp):
                self.change_net_val()
                with torch.no_grad():
                    out = self.forward_cat(images_cuda)
                    pred = F.softmax(out['logits'], dim=1).cpu()
                    maxpred, labels_online = torch.max(pred.detach(), dim=1)
                    mask_online = (maxpred > self.T).unsqueeze(1)
                    mask_online = mask_online.type_as(masks)
                    labels_online = labels_online
                self.change_net_train()
            images, labels, masks, bdry, inside, isbdry = \
                imgmix_multi_withmaskol_bdry(images, labels, masks,
                                             labels_online, mask_online,
                                             self.bdry_size,
                                             self.ignore_bg,
                                             self.mix_prob)
        else:
            images, labels, masks, bdry, inside, isbdry = \
                imgmix_multi_withmask_bdry(images, labels, masks,
                                           self.bdry_size,
                                           self.ignore_bg,
                                           self.mix_prob)
        if self.mix_aug:
            images, labels, masks = augment_withmask(images, labels, masks)


        images, labels, masks = self.scatter((images, labels, masks))
        bdry, inside, isbdry = self.scatter((bdry, inside, isbdry))
        with autocast(enabled=self.is_amp):
            out1, out2 = self.forward(images)

            _, out1_pre = torch.max(out1["logits"], dim=1)
            _, out2_pre = torch.max(out2["logits"], dim=1)
            out1_pre = out1_pre.long()
            out2_pre = out2_pre.long()

            loss_ce1 = self.loss_ce(out1["logits"], labels) * masks
            loss_ce1 = torch.mean(loss_ce1 * (1 + self.bdry_whb * bdry))
            loss_ce2 = self.loss_ce(out2["logits"], labels) * masks
            loss_ce2 = torch.mean(loss_ce2 * (1 + self.bdry_whi * bdry))

            loss_cot1 = self.loss_ce(out1["logits"], out2_pre) * (1-masks)
            loss_cot1 = torch.mean(loss_cot1 * (1 + self.bdry_wlb * bdry))
            loss_cot2 = self.loss_ce(out2["logits"], out1_pre) * (1-masks)
            loss_cot2 = torch.mean(loss_cot2 * (1 + self.bdry_wli * bdry))


        loss_dict = dict(
            loss_ce1 = loss_ce1.clone().detach(),
            loss_cot1 = loss_cot1.clone().detach()
        )
        self.update_metric_train(out1['logits'], labels)
        self.update_loss(loss_dict)

        return dict(
            loss = loss_ce1+loss_ce2+loss_cot1+loss_cot2
        )

    def change_net_val(self):
        for _, net in self.nets.items():
            net.eval()

    def change_net_train(self):
        for _, net in self.nets.items():
            net.train()

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any]) -> None:
        images, labels = batch['img'], batch['target']
        images, labels = self.scatter((images, labels))
        output = self.forward_cat_test(images)

        if self.test_msc:
            logits = output['pre_logits']
            _, _, H, W = logits.shape
            interp = lambda l: F.interpolate(
                l, size=(H, W), mode="bilinear", align_corners=False
            )

            # Scaled
            logits_pyramid = []
            for p in self.scales:
                h = F.interpolate(images, scale_factor=p, mode="bilinear", 
                                  align_corners=False)
                logits_pyramid.append(self.forward_cat_test(h)['pre_logits'])


            # Pixel-wise max
            logits_all = [logits] + [interp(l) for l in logits_pyramid]
            logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
            output['pre_logits'] = logits_max


        if self.save_logits:
            img_names = batch['name']
            B = images.size(0)
            for i in range(B):
                img_name = img_names[i]
                logit_dir = os.path.join(self.logits_path, img_name+'.npy')
                np.save(logit_dir, output["pre_logits"][i].cpu().numpy())

        if len(labels.size()) == 3:
            self.update_metric_val(output["logits"], labels) 

    def forward_cat_test(self, images: Tensor):
        out = {}
        out1 = self.nets['network1'](images)
        out2 = self.nets['network2'](images)
        out["logits"] = (out1["logits"]+out2["logits"])/2
        out["pre_logits"] = (out1["pre_logits"]+out2["pre_logits"])/2
        return out

    def after_train_epoch(self):
        self.epoch += 1
        
    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> None:
        images, labels = batch['img'], batch['target']
        images, labels = self.scatter((images, labels))
        output = self.forward_cat(images)
        self.update_metric_val(output["logits"], labels)  

    def forward(self, images: Tensor):
        out1 = self.nets['network1'](images)
        out2 = self.nets['network2'](images)
        return out1, out2

    def forward_cat(self,images: Tensor):
        out1 = self.nets['network1'](images)
        out2 = self.nets['network2'](images)
        out1["logits"] = (out1["logits"]+out2["logits"])/2
        return out1
    


