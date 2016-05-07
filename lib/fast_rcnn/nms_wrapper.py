# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
try:
	from nms.gpu_nms import gpu_nms
	force_cpu=False
except:
	force_cpu=False
from nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=force_cpu):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
    	try:
        	return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        except:
        	return cpu_nms(dets, thresh)
    else:
        return cpu_nms(dets, thresh)
