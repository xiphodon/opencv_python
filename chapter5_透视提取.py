import cv2
import numpy as np


def func():
    """

    :return:
    """
    img = cv2.imread(f'./resources/card.jpeg')
    width, height = 300, 200
    pts1 = np.float32([
        [94, 302],
        [205, 243],
        [152, 369],
        [265, 300]
    ])
    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    # 透视矩阵，目标初始四点坐标，拉伸后的坐标
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 透视变换（原图，透视矩阵，输出图像大小）
    img_output = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow('img', img)
    cv2.imshow('img output', img_output)

    cv2.waitKey(0)

if __name__ == '__main__':
    func()
