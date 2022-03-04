import cv2
import numpy as np


def img_reshape():
    """
    img尺寸变换
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    print(img.shape)
    # dsize为宽、高
    img_resize = cv2.resize(img, dsize=(256, 256))
    print(img_resize.shape)
    # 矩阵shape顺序为高、宽、通道
    # img_cropped = img[0:256, 256:512]
    img_cropped = img[0:256, 256:512, :]
    print(img_cropped.shape)

    cv2.imshow('img', img)
    cv2.imshow('img resize', img_resize)
    cv2.imshow('img cropped', img_cropped)

    cv2.waitKey(0)


if __name__ == '__main__':
    img_reshape()
