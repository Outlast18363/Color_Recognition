import cv2
import mediapipe as mp
import math
import numpy as np

Gray = ['black 黑','grey 灰','white 白']

def colorDic (pixel):
    i=pixel[0]
    j = pixel[1]
    k = pixel[2]
    if i >= 156 or i <= 10 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'red 红'
    elif i >= 0 and i <= 180 and j >= 0 and j <= 255 and k >= 0 and k <= 46:
        return 'black 黑'
    elif i >= 0 and i <= 180 and j >= 0 and j <= 43 and k >= 46 and k <= 220:
        return 'grey 灰'
    elif i >= 0 and i <= 180 and j >= 0 and j <= 30 and k >= 221 and k <= 255:
        return 'white 白'
    elif i > 10 and i <= 25 and j > 42 and j < 256 and k >= 46 and k <= 255:
        return 'orange 橙色'
    elif i > 25 and i <= 34 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'yellow 黄'
    elif i > 34 and i <= 77 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'lime 绿'
    elif i > 77 and i <= 99 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'green 青'
    elif i > 99 and i <= 124 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'cyans 蓝'
    elif i > 124 and i <= 155 and j >= 43 and j <= 255 and k >= 46 and k <= 255:
        return 'sky blue 紫'


def grey(pixel, threshhold):
    if abs(pixel[0]-pixel[1]) <= threshhold and abs(pixel[2]-pixel[1]) <= threshhold and abs(pixel[0]-pixel[2]) <= threshhold:
        if pixel[0] <=38:
            return 0 #black
        elif pixel[0] <= 100:
            return 1 #grey
        elif pixel[0] <= 255:
            return 2 #white
    else:
        return -1

def clustering(fr,far_point):  # 颜色算法部分
    scale = 5  # 选出的大小
    #far_point=np.array([250,250])
    frr = np.array(fr)
    frr = cv2.cvtColor(frr, cv2.COLOR_BGR2RGB)
    group = frr[far_point[0] - scale:far_point[0] + scale,
            far_point[1] - scale:far_point[1] + scale]  # 以最外点为中心选出一个特定大小的区域，用来做聚类算法
    group = frr[far_point[1] - scale:far_point[1] + scale,
            far_point[0] - scale:far_point[0] + scale]  # 以最外点为中心选出一个特定大小的区域，用来做聚类算法
    #group = group.reshape(-1,3) #变成一维的像素数组
    #cv2.circle(fr, far_point, 5, [255, 0, 255], -1)  # 在图像中标记中点 （红色点）
    if group.size > 0:
        #group = group[:,0:1]
        group = group.astype(np.int_)
        hsvColors=[]
        for i in range(0,group.shape[0]):
            for j in range(0, group.shape[1]):
                pixel = group[i,j]
                #imageSee(pixel)
                # gr = grey(pixel,10) #灰度阈值
                # if gr != -1:
                #     hsvColors.append(Gray[gr])
                # else:
                pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2HSV)[0][0]
                pixel= pixel.astype(np.int_)

                hsvColors.append(colorDic(pixel))

        colorList, colorNum = np.unique(np.array(hsvColors), return_counts=True) #有几类，分别有几个
        ranks = np.argsort(colorNum)
        topRkCor = colorList[ranks[-1]]  # 占比最大的颜色的节点index
        #print(topRkCor)
        print(topRkCor, pixel[0])

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
    return x_px, y_px

def main():
    hand = mp.solutions.hands
    pro = hand.Hands(max_num_hands=1)
    draw = mp.solutions.drawing_utils

    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        suc, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pro.process(img) #找到关键点们
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                draw.draw_landmarks(img, hand_landmarks, connections=hand.HAND_CONNECTIONS)
                normalList = list(hand_landmarks.landmark)
                pixel_7 = normalList[7]; pixel_8 = normalList[8]
                x1, y1 = _normalized_to_pixel_coordinates(pixel_7.x, pixel_7.y, *img.shape[0:2][::-1])
                x2, y2 = _normalized_to_pixel_coordinates(pixel_8.x, pixel_8.y, *img.shape[0:2][::-1])
                pixel_7 = np.array([int(x1), int(y1)])
                pixel_8 = np.array([int(x2), int(y2)])
                vec = np.array([pixel_8[0]-pixel_7[0],pixel_8[1]-pixel_7[1]])
                mag = np.linalg.norm(vec)
                vec = np.divide(vec,mag) #变成单位向量
                length = 50 #定义新模长
                pixel_8[0]+=int(vec[0]*length); pixel_8[1]+=int(vec[1]*length) #加在点的坐标上
                clustering(img, pixel_8)
                cv2.circle(img, pixel_8, 10, (0, 255, 0), -1)

        cv2.imshow('K', img)
        if cv2.waitKey(5) & 0xff == 27:
            break

if __name__ == '__main__':
    main()