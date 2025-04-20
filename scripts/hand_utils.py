import cv2
import mediapipe as mp
import math
import numpy as np
import argparse
import time

from picamera import PiCamera
from picamera.array import PiRGBArray

Gray = ['black 黑','grey 灰','white 白']

COLOR_LIST = np.array(['黑','灰','白', 
        "红", "橙", "黄", "绿", "青", "蓝", "紫"])
# COLOR_LIST = np.array(['black 黑','grey 灰','white 白', 
#         "red 红", "orange 橙", "yellow 黄", 
#         "lime 绿", "green 青", "cyans 蓝", "purple 紫"])
# 红橙黄绿青蓝紫
COLOR_BOUNDS = np.array([10, 25, 34, 77, 99, 124, 155]).reshape(1,-1)
GRAY_BOUNDS = np.array([0,43,220]).reshape(1,-1)
# Color correction for Pi camera
COLOR_SHIFT = np.array([0,-15,0]).reshape(1,-1)

def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return np.array([x_px, y_px])

def getPatch(frame, center, scale=10):
    '''Return square patch with radius=scale around given center'''
    return frame[center[1] - scale:center[1] + scale,
            center[0] - scale:center[0] + scale].copy()

def detColor(img_patch, grey_thresh=15, cc=True):
    '''Determine major color of an image patch'''
    if img_patch.size<=0: return
    if cc: img_patch = (img_patch.astype(np.int_)+COLOR_SHIFT).clip(min=0,max=255).astype(np.uint8)
    hsv_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
    h_channel = hsv_patch[:,:,0].reshape(-1)
    s_channel = hsv_patch[:,:,1].reshape(-1)
    v_channel = hsv_patch[:,:,2].reshape(-1)

    grey_flags = (np.max(img_patch, axis=2)-np.min(img_patch, axis=2)).reshape(-1)<=grey_thresh
    grey_flags = np.sum(v_channel[:,np.newaxis]>=GRAY_BOUNDS, axis=1)*grey_flags
    color_idx = np.sum(h_channel[:,np.newaxis]>=COLOR_BOUNDS, axis=1)
    color_idx = color_idx%COLOR_BOUNDS.size+COLOR_LIST.size-COLOR_BOUNDS.size
    color_idx = color_idx*(grey_flags==0)+(grey_flags-1)*(grey_flags>0)
    # import ipdb; ipdb.set_trace()
    uniq_color_idx, color_cnt = np.unique(color_idx, return_counts=True)
    major_color_idx = uniq_color_idx[np.argsort(color_cnt)[-1]]

    mean_major_h = np.mean(h_channel[color_idx==major_color_idx])
    mean_major_s = np.mean(s_channel[color_idx==major_color_idx])
    mean_major_v = np.mean(v_channel[color_idx==major_color_idx])

    result_str = f"color: {COLOR_LIST[major_color_idx]}\t h: {mean_major_h:4.1f}\t s: {mean_major_s:4.1f}\t v: {mean_major_v:4.1f}"
    print(result_str)
    return result_str


def getPoint(hand_landmarks, img_shape, extention=50):
    '''Return point of finger top'''
    top_finger = hand_landmarks[8]
    mid_finger = hand_landmarks[7]
    top_finger = _normalized_to_pixel_coordinates(top_finger.x, top_finger.y, *img_shape[1::-1])
    mid_finger = _normalized_to_pixel_coordinates(mid_finger.x, mid_finger.y, *img_shape[1::-1])
    if(top_finger is None or mid_finger is None): return None
    vec = top_finger - mid_finger
    mag = np.linalg.norm(vec)
    far_point = top_finger+extention/mag*vec
    return far_point.astype(np.int_)


def procFrame(frame, hand_sol, hand_proc, hand_drawer):
    '''Given a frame, detect hand, find finger top, and determine
    the color.'''
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_detect_results = hand_proc.process(rgb_frame)
    frame[:20,:20,:] = np.array([255,0,0]).reshape(1,1,3)
    if hand_detect_results.multi_hand_landmarks is None or len(hand_detect_results.multi_hand_landmarks)==0: return frame, None
    hand_landmarks = hand_detect_results.multi_hand_landmarks[0]
    # hand_drawer.draw_landmarks(frame, hand_landmarks, connections=hand_sol.HAND_CONNECTIONS)
    far_point = getPoint(hand_landmarks.landmark, frame.shape)
    scale = 10
    point_shift = np.array([scale, scale])
    if far_point is not None:
        img_patch = getPatch(frame, far_point)
        if img_patch is not None and img_patch.size>0: cv2.imshow('P', img_patch)
        result_str = detColor(img_patch)
        frame = cv2.rectangle(frame,far_point-point_shift, far_point+point_shift, (0, 0, 255), 2)
        # frame = cv2.circle(frame, far_point, 5, (0, 255, 0), -1)
    else: result_str = None
    return frame, result_str


def main():
    hand_sol = mp.solutions.hands
    hand_proc = hand_sol.Hands(max_num_hands=1)
    hand_drawer = mp.solutions.drawing_utils

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", type=str, default="csi",
        help="Type of camera to use")

    args = vars(ap.parse_args())
    if args["camera"] == "csi":
        cam = cv2.VideoCapture(0)
        while cam.isOpened():
            suc, img = cam.read()
            img, result_str = procFrame(img, hand_sol, hand_proc, hand_drawer)
            cv2.imshow('K', img)
            if cv2.waitKey(5) & 0xff == 27:
                cam.release()
                break
    else:
        camera = PiCamera()
        camera.resolution = (480, 360)
        # ------------ camera configure ----------------
        # camera.brightness = 70
        # camera.saturation = 100
        # camera.iso = 200
        rawCapture = PiRGBArray(camera, size=(480, 360))
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array.copy()
            img, result_str = procFrame(img, hand_sol, hand_proc, hand_drawer)
            rawCapture.seek(0)
            rawCapture.truncate()
            cv2.imshow('K', img)
            
            if cv2.waitKey(5) & 0xff == 27:
                camera.close()
                break
if __name__ == '__main__':
    main()