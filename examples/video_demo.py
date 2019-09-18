import sys
sys.path.append('../')

import cv2
import numpy as np
from rcnnpose.estimator import BodyPoseEstimator
from rcnnpose.utils import draw_body_connections, draw_keypoints, draw_masks


estimator = BodyPoseEstimator(pretrained=True)
videoclip = cv2.VideoCapture('media/example.mp4')

while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    
    pred_dict = estimator(frame, masks=True, keypoints=True)
    masks = estimator.get_masks(pred_dict['estimator_m'], score_threshold=0.99)
    keypoints = estimator.get_keypoints(pred_dict['estimator_k'], score_threshold=0.99)
    
    frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_dst = cv2.merge([frame_dst] * 3)
    overlay_m = draw_masks(frame_dst, masks, color=(0, 255, 0), alpha=0.5)
    overlay_k = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    overlay_k = draw_keypoints(overlay_k, keypoints, radius=4, alpha=0.8)
    frame_dst = np.hstack((frame, overlay_m, overlay_k))
    
    cv2.imshow('Video Demo', frame_dst)
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break

videoclip.release()
cv2.destroyAllWindows()
