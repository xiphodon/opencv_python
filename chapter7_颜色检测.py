import cv2
import numpy as np
from utils import img_stack_util

track_windows_name = 'TrackBars'


def empty_on_change(value):
    """
    跟踪杆空回调
    :return:
    """
    pass


def create_track_bar():
    """
    创建跟踪杆
    :return:
    """
    cv2.namedWindow(track_windows_name)
    cv2.resizeWindow(track_windows_name, 600, 300)

    # 色度
    cv2.createTrackbar('huv min', track_windows_name, 1, 179, empty_on_change)
    cv2.createTrackbar('huv max', track_windows_name, 13, 179, empty_on_change)
    # 饱和度
    cv2.createTrackbar('sat min', track_windows_name, 32, 255, empty_on_change)
    cv2.createTrackbar('sat max', track_windows_name, 255, 255, empty_on_change)
    # 纯度
    cv2.createTrackbar('val min', track_windows_name, 127, 255, empty_on_change)
    cv2.createTrackbar('val max', track_windows_name, 253, 255, empty_on_change)


def color_detect():
    """
    颜色检测
    :return:
    """

    create_track_bar()

    while True:
        img = cv2.imread(r'./resources/car.jpeg')
        # HSV 分别为 色调、饱和度、纯度
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos('huv min', track_windows_name)
        h_max = cv2.getTrackbarPos('huv max', track_windows_name)
        s_min = cv2.getTrackbarPos('sat min', track_windows_name)
        s_max = cv2.getTrackbarPos('sat max', track_windows_name)
        v_min = cv2.getTrackbarPos('val min', track_windows_name)
        v_max = cv2.getTrackbarPos('val max', track_windows_name)

        print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lowerb=lower, upperb=upper)

        img_result = cv2.bitwise_and(img, img, mask=mask)

        big_img = img_stack_util.stack_img(([img, img_hsv], [mask, img_result]), scale=0.5)

        cv2.imshow('img', big_img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 有按键则返回按键ASCII码，无按键则返回-1
            # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
            break


if __name__ == '__main__':
    color_detect()

