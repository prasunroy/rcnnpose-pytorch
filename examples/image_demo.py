import sys
sys.path.append('../')

import cv2
import numpy as np
from rcnnpose.estimator import BodyPoseEstimator
from rcnnpose.utils import draw_body_connections, draw_keypoints, draw_masks


estimator = BodyPoseEstimator(pretrained=True)
image_src = cv2.imread('media/example.jpg')
pred_dict = estimator(image_src, masks=True, keypoints=True)
masks = estimator.get_masks(pred_dict['estimator_m'], score_threshold=0.99)
keypoints = estimator.get_keypoints(pred_dict['estimator_k'], score_threshold=0.99)

image_dst = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
image_dst = cv2.merge([image_dst] * 3)
overlay_m = draw_masks(image_dst, masks, color=(0, 255, 0), alpha=0.5)
overlay_k = draw_body_connections(image_src, keypoints, thickness=4, alpha=0.7)
overlay_k = draw_keypoints(overlay_k, keypoints, radius=5, alpha=0.8)
image_dst = np.hstack((image_src, overlay_m, overlay_k))

while True:
    cv2.imshow('Image Demo', image_dst)
    if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
        break
cv2.destroyAllWindows()
