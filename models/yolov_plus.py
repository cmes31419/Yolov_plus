#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
import torch.nn as nn

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
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x, targets=None, nms_thresh=0.5, lframe=0, gframe=32):
        print("yolov is forwarding !!!")
        
        # Method 1: Basic CPU timing (your current approach, but improved)
        if not torch.cuda.is_available():
            start_time = time.perf_counter()  # More precise than time.time()
            fpn_outs = self.backbone(x)
            backbone_time = time.perf_counter() - start_time
            print(f"backbone time (CPU): {backbone_time:.6f}s")
            
            start_time = time.perf_counter()
            outputs = self.head(fpn_outs, targets, x, nms_thresh=nms_thresh, 
                              lframe=lframe, gframe=gframe)
            head_time = time.perf_counter() - start_time
            print(f"head time (CPU): {head_time:.6f}s")
            
        else:
            # Method 2: GPU timing with CUDA events (RECOMMENDED for GPU inference)
            torch.cuda.synchronize()  # Ensure all previous operations are complete
            
            # Backbone timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            fpn_outs = self.backbone(x)
            end_event.record()
            torch.cuda.synchronize()  # Wait for completion
            backbone_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
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
        
        return outputs

# Alternative: Context manager approach for cleaner code
# @contextlib.contextmanager
# def timer(name, device='cpu'):
#     """Context manager for timing operations"""
#     if device == 'gpu' and torch.cuda.is_available():
#         torch.cuda.synchronize()
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         yield
#         end.record()
#         torch.cuda.synchronize()
#         elapsed = start.elapsed_time(end) / 1000.0
#         print(f"{name}: {elapsed:.6f}s")
#     else:
#         start = time.perf_counter()
#         yield
#         elapsed = time.perf_counter() - start
#         print(f"{name}: {elapsed:.6f}s")

# class YOLOVWithContextTimer(nn.Module):
#     """Version using context manager for cleaner timing code"""
#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
        
#     def forward(self, x, targets=None, nms_thresh=0.5, lframe=0, gframe=32):
#         print("yolov is forwarding !!!")
#         device = 'gpu' if torch.cuda.is_available() and x.is_cuda else 'cpu'
        
#         with timer("backbone", device):
#             fpn_outs = self.backbone(x)
            
#         with timer("head", device):
#             outputs = self.head(fpn_outs, targets, x, nms_thresh=nms_thresh, 
#                               lframe=lframe, gframe=gframe)
        
#         return outputs

# # For more detailed profiling, you can also use:
# class YOLOVWithProfiler(nn.Module):
#     """Version with PyTorch profiler for detailed analysis"""
#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
        
#     def forward(self, x, targets=None, nms_thresh=0.5, lframe=0, gframe=32):
#         print("yolov is forwarding !!!")
        
#         with torch.profiler.profile(
#             activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#             record_shapes=True,
#             profile_memory=True,
#             with_stack=True
#         ) as prof:
#             with torch.profiler.record_function("backbone"):
#                 fpn_outs = self.backbone(x)
            
#             with torch.profiler.record_function("head"):
#                 outputs = self.head(fpn_outs, targets, x, nms_thresh=nms_thresh, 
#                                   lframe=lframe, gframe=gframe)
        
#         # Print profiler results
#         print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
#         return outputs