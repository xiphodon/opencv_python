import cv2
import numpy as np


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


def stack_img(img_arr: tuple, scale=1.0):
    """
    堆叠图片
    :param img_arr:
    :param scale:
    :return:
    """
    if not isinstance(img_arr[0], list):
        # 仅水平堆叠
        img_arr = (list(img_arr), )

    rows = len(img_arr)

    # 水平+竖直堆叠
    row_max_num = 0
    for i in range(rows):
        row_max_num = max(row_max_num, len(img_arr[i]))
    first_img = img_arr[0][0]
    first_img_shape = list(first_img.shape)
    first_img_shape[0], first_img_shape[1] = int(scale * first_img_shape[0]), int(scale * first_img_shape[1])
    black_img = np.zeros(shape=(first_img_shape[1], first_img_shape[0], 3), dtype=np.uint8)
    col_all_img_list = list()
    for i in range(rows):
        row_all_img_list = list()
        for item_img in img_arr[i]:
            if len(item_img.shape) == 2:
                # 灰度图
                item_img = cv2.cvtColor(item_img, cv2.COLOR_GRAY2BGR)
            row_all_img_list.append(cv2.resize(item_img, dsize=first_img_shape[:2]))
        if len(img_arr[i]) < row_max_num:
            for _ in range(row_max_num - len(img_arr[i])):
                row_all_img_list.append(black_img)
        row_all_img = np.hstack(row_all_img_list)
        col_all_img_list.append(row_all_img)
    big_img = np.vstack(col_all_img_list)
    return big_img


def my_img_stack():
    """
    图片堆叠
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    # big_img = stack_img(img_arr=(img, img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(img, dsize=(128, 128))
    big_img = stack_img(img_arr=(img, img_gray, img_small), scale=0.5)
    # big_img = stack_img(img_arr=([img, img_gray], [img, img, img], [img]), scale=0.3)
    cv2.imshow('stack img', big_img)

    cv2.waitKey(0)


if __name__ == '__main__':
    # np_stack()
    my_img_stack()
