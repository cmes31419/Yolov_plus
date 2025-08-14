#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

import time

class YOLOV(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None,nms_thresh=0.5,lframe=0,gframe=32):
        # fpn output content features of [dark3, dark4, dark5]

        print("yolov is forwarding !!!")

        current_timestamp = time.time()

        fpn_outs = self.backbone(x)

        print(f"backbone time: {time.time() - current_timestamp}")
        current_timestamp = time.time()

        # inference only
        outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)

        print(f"head time: {time.time() - current_timestamp}")
        
        return outputs
