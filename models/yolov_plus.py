#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
import torch.nn as nn

import csv
import time
import contextlib

# class YOLOV(nn.Module):
#     """
#     YOLOX model module. The module list is defined by create_yolov3_modules function.
#     The network returns loss values from three YOLO layers during training
#     and detection results during test.
#     """

#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head

#     def forward(self, x, targets=None,nms_thresh=0.5,lframe=0,gframe=32):
#         # fpn output content features of [dark3, dark4, dark5]

#         print("yolov is forwarding !!!")

#         current_timestamp = time.time()

#         fpn_outs = self.backbone(x)

#         print(f"backbone time: {time.time() - current_timestamp}")
#         current_timestamp = time.time()

#         # inference only
#         outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)

#         print(f"head time: {time.time() - current_timestamp}")
        
#         return outputs

class YOLOV(nn.Module):
    """
    YOLOX model module with improved timing accuracy.
    """
    def __init__(self, backbone=None, head=None, csv_file="inference_times.csv"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.csv_file = csv_file
        try:
            with open(self.csv_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Backbone Time (s)', 'Head Time (s)'])
        except FileExistsError:
            pass
        
    def _write_to_csv(self, backbone_time, head_time):
        """Helper function to append times to CSV"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{backbone_time:.6f}", f"{head_time:.6f}"])

    def forward(self, x, targets=None, nms_thresh=0.5, lframe=0, gframe=32):
        print("yolov is forwarding !!!")
        
        # Method 1: Basic CPU timing
        if not torch.cuda.is_available():
            start_time = time.perf_counter()
            fpn_outs = self.backbone(x)
            backbone_time = time.perf_counter() - start_time
            print(f"backbone time (CPU): {backbone_time:.6f}s")
            
            start_time = time.perf_counter()
            outputs = self.head(fpn_outs, targets, x, nms_thresh=nms_thresh, 
                              lframe=lframe, gframe=gframe)
            head_time = time.perf_counter() - start_time
            print(f"head time (CPU): {head_time:.6f}s")
            
        else:
            # Method 2: GPU timing with CUDA events
            torch.cuda.synchronize()
            
            # Backbone timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            fpn_outs = self.backbone(x)
            end_event.record()
            torch.cuda.synchronize()
            backbone_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"backbone time (GPU): {backbone_time:.6f}s")
            
            # Head timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            outputs = self.head(fpn_outs, targets, x, nms_thresh=nms_thresh, 
                              lframe=lframe, gframe=gframe)
            end_event.record()
            torch.cuda.synchronize()
            head_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"head time (GPU): {head_time:.6f}s")
        
        # Write times to CSV
        self._write_to_csv(backbone_time, head_time)
        
        return outputs

        
