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
from datasets.transforms.transform import augment_withmask
from utils.misc import get_logging_path



class Base(BaseSimSeg):

    def __init__(self, config, nets_dict) -> None:
        super().__init__(config, nets_dict)
        self.logger = R0Logger(__name__)
        self.init_loss()

        self.save_logits = config.model.settings.get('save_logits', False)
        self.strong_aug = config.model.settings.get('strong_aug', False)
        if self.save_logits:
            self.logging_path = get_logging_path(config)
            self.logits_path = os.path.join(self.logging_path, 'logits')
            os.makedirs(self.logits_path, exist_ok=True)

    def init_loss(self):
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    

    def train_step(self, batch: Dict[str, Any]) -> Tuple[Tensor, float]:
        images, labels = batch['img'], batch['target']
        masks = batch['mask']

        if self.strong_aug:
            images, labels, masks = augment_withmask(images, labels, masks)
        #vis_data(images, labels)

        images, labels, masks = self.scatter((images, labels, masks))
        with autocast(enabled=self.is_amp):
            out1, out2 = self.forward(images)


            loss_ce1 = torch.mean(self.loss_ce(out1["logits"], labels))
            loss_ce2 = torch.mean(self.loss_ce(out2["logits"], labels))
            

            losses = loss_ce1 + loss_ce2

        loss_dict = dict(
            loss_ce1 = loss_ce1.clone().detach(),
            loss_ce2 = loss_ce2.clone().detach(),
        )
        self.update_metric_train(out1['logits'], labels)
        self.update_loss(loss_dict)
        return dict(loss = losses)



    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any]) -> None:
        images, labels = batch['img'], batch['target']
        images, labels = self.scatter((images, labels))
        output = self.forward_cat_test(images)
        if self.save_logits:
            img_names = batch['name']
            B = images.size(0)
            for i in range(B):
                img_name = img_names[i]
                logit_dir = os.path.join(self.logits_path, img_name+'.npy')
                np.save(logit_dir, output["pre_logits"][i].cpu().numpy())

        self.update_metric_val(output["logits"], labels) 

    def forward_cat_test(self, images: Tensor):
        out1 = self.nets['network1'](images)
        out2 = self.nets['network2'](images)
        out = {}
        out["logits"] = (out1["logits"]+out2["logits"])/2
        out["pre_logits"] = (out1["pre_logits"]+out2["pre_logits"])/2
        return out

        
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



