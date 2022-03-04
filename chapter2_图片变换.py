import cv2
import numpy as np


def img_converts():
    """
    img转换
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊，模糊半径（奇数， 奇数）
    img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0)
    # 边缘检测，阈值小值用于边缘连接，阈值大值用于阈值发现
    # img_canny_100_100 = cv2.Canny(img_gray, threshold1=100, threshold2=100)
    img_canny_150_200 = cv2.Canny(img_gray, threshold1=150, threshold2=200)
    # 膨胀，kernel为膨胀核，iterations为膨胀次数
    img_dilation = cv2.dilate(img_canny_150_200, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    # 腐蚀，kernel为膨胀核，iterations为腐蚀次数
    img_erode = cv2.erode(img_dilation, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    cv2.imshow('gray img', img_gray)
    cv2.imshow('blur img', img_blur)
    # cv2.imshow('canny_100_100 img', img_canny_100_100)
    cv2.imshow('canny_150_200 img', img_canny_150_200)
    cv2.imshow('dilation img', img_dilation)
    cv2.imshow('erode img', img_erode)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_converts()
