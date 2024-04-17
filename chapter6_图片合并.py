import cv2
import numpy as np

from utils.img_stack_util import stack_img


def np_stack():
    """
    numpy 堆叠
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    img_h = np.hstack((img, img))
    # img_v = np.vstack((img, img))
    cv2.imshow('img_h', img_h)
    # cv2.imshow('img_v', img_v)
    cv2.waitKey(0)


def my_img_stack():
    """
    图片堆叠
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    # big_img = stack_img(img_arr=(img, img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(img, dsize=(128, 128))
    big_img = stack_img(img_arr=(img, img_gray, img_small), scale=0.5, labels=('origin', 'gray', 'small'))
    # big_img = stack_img(img_arr=([img, img_gray, img_small], ), scale=0.5)
    # big_img = stack_img(img_arr=([img, img_gray], [img, img, img], [img]), scale=0.3)
    cv2.imshow('stack img', big_img)

    cv2.waitKey(0)


if __name__ == '__main__':
    # np_stack()
    my_img_stack()
