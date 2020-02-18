# -*- coding: utf-8 -*-
"""
RCNNPose utility functions.
Created on Wed Sep 18 10:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/rcnnpose-pytorch

"""


import cv2
import numpy as np


def _colorize_mask(mask, color=None):
    b = mask * np.random.randint(0, 255) if not color else mask * color[0]
    g = mask * np.random.randint(0, 255) if not color else mask * color[1]
    r = mask * np.random.randint(0, 255) if not color else mask * color[2]
    return cv2.merge((b, g, r))


def _draw_keypoint(image, point, color, radius=1):
    x, y, v = point
    if int(v):
        cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return image


def _draw_connection(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if int(v1) and int(v2):
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image


def _draw_box(image, point1, point2, color, thickness=1):
    x1, y1 = point1
    x2, y2 = point2
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image


def draw_masks(image, masks, color=None, alpha=0.5):
    overlay = image.copy()
    for mask in masks:
        mask_bin = np.uint8(mask > 0)
        mask_inv = cv2.merge([1 - mask_bin] * 3)
        mask_rgb = _colorize_mask(mask_bin, color)
        overlay = cv2.multiply(overlay, mask_inv)
        overlay = cv2.add(overlay, mask_rgb)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for p in kp:
            overlay = _draw_keypoint(overlay, p, (0, 255, 0), radius)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_body_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    h_conn = [(0, 1), (0, 2), (1, 3), (2, 4)]
    l_conn = [(5, 7), (7, 9), (11, 13), (13, 15)]
    r_conn = [(6, 8), (8, 10), (12, 14), (14, 16)]
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in h_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in l_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in r_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 0, 255), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_boxes(image, boxes, thickness=1, alpha=1.0):
    overlay = image.copy()
    for box in boxes:
        overlay = _draw_box(overlay, box[:2], box[2:], (0, 255, 0), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
