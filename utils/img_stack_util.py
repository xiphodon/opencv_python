import typing

import cv2
import numpy as np


def stack_img(img_arr: tuple, scale=1.0, lables=None):
    """
    堆叠图片
    :param img_arr:
    :param scale:
    :param lables:
    :return:
    """
    if not isinstance(img_arr[0], list):
        # 仅水平堆叠
        img_arr = (list(img_arr),)

    rows = len(img_arr)

    # 水平+竖直堆叠
    row_max_num = 0
    for i in range(rows):
        row_max_num = max(row_max_num, len(img_arr[i]))
    first_img = img_arr[0][0]
    first_img_shape = list(first_img.shape)
    first_img_shape[0], first_img_shape[1] = int(scale * first_img_shape[0]), int(scale * first_img_shape[1])
    black_img = np.zeros(shape=(first_img_shape[0], first_img_shape[1], 3), dtype=np.uint8)
    col_all_img_list = list()
    for i in range(rows):
        row_all_img_list = list()
        for j, item_img in enumerate(img_arr[i]):
            if len(item_img.shape) == 2:
                # 灰度图转为3通道图
                item_img = cv2.cvtColor(item_img, cv2.COLOR_GRAY2BGR)
            _img = cv2.resize(item_img, dsize=(first_img_shape[1], first_img_shape[0]))
            if lables is not None:
                offset = 1
                cv2.rectangle(img=_img, pt1=(offset, offset),
                              pt2=(first_img_shape[0] - offset, int(30 * scale) - offset), color=(255, 255, 255),
                              thickness=cv2.FILLED)
                cv2.putText(img=_img, text=lables[i][j], org=(2 * offset, int(20 * scale)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7 * scale, color=(0, 0, 0),
                            thickness=max(round(2 * scale), 1))
            row_all_img_list.append(_img)
        if len(img_arr[i]) < row_max_num:
            for _ in range(row_max_num - len(img_arr[i])):
                row_all_img_list.append(black_img)
        row_all_img = np.hstack(row_all_img_list)
        col_all_img_list.append(row_all_img)
    _big_img = np.vstack(col_all_img_list)
    return _big_img


if __name__ == '__main__':
    img = cv2.imread(r'../resources/lena.jpeg')
    # big_img = stack_img(img_arr=(img, img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(img, dsize=(128, 128))
    big_img = stack_img(img_arr=(img, img_gray, img_small), scale=0.5)
    # big_img = stack_img(img_arr=([img, img_gray, img_small], ), scale=0.5)
    # big_img = stack_img(img_arr=([img, img_gray], [img, img, img], [img]), scale=0.3)
    cv2.imshow('stack img', big_img)

    cv2.waitKey(0)
