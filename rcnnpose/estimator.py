# -*- coding: utf-8 -*-
"""
RCNNPose estimator.
Created on Wed Sep 18 10:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/rcnnpose-pytorch

"""


import numpy as np
import torch
import torchvision


class BodyPoseEstimator(object):
    
    def __init__(self, pretrained=False):
        self._estimator_m = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
        self._estimator_k = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=pretrained)
        if torch.cuda.is_available():
            self._estimator_m = self._estimator_m.cuda()
            self._estimator_k = self._estimator_k.cuda()
        self._estimator_m.eval()
        self._estimator_k.eval()
    
    def __call__(self, image, masks=True, keypoints=True):
        x = self._transform_image(image)
        if torch.cuda.is_available():
            x = x.cuda()
        m = self._predict_masks(x) if masks else [None]
        k = self._predict_keypoints(x) if keypoints else [None]
        return {'estimator_m': m[0], 'estimator_k': k[0]}
    
    def _transform_image(self, image):
        return torchvision.transforms.ToTensor()(image)
    
    def _predict_masks(self, x):
        with torch.no_grad():
            return self._estimator_m([x])
    
    def _predict_keypoints(self, x):
        with torch.no_grad():
            return self._estimator_k([x])
    
    @staticmethod
    def get_masks(dictionary, label=1, score_threshold=0.5):
        masks = []
        if dictionary:
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    mask = dictionary['masks'][i].detach().cpu().squeeze().numpy() > 0.5
                    masks.append(mask)
        return np.asarray(masks, dtype=np.uint8)
    
    @staticmethod
    def get_keypoints(dictionary, label=1, score_threshold=0.5):
        keypoints = []
        if dictionary:
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    keypoint = dictionary['keypoints'][i].detach().cpu().squeeze().numpy()
                    keypoints.append(keypoint)
        return np.asarray(keypoints, dtype=np.int32)
