import cv2
import numpy as np


def draw_img():
    """
    绘制图像
    :return:
    """
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    # 通道顺序为BGR
    print(img.shape)
    # 高、宽、通道
    # img[:, :, :] = 255, 0, 0
    # img[200:400, 0:250, :] = 255, 0, 0

    cv2.line(img, pt1=(10, 10), pt2=(100, 200), color=(0, 0, 255), thickness=3)
    cv2.line(img, pt1=(0, 300), pt2=(img.shape[1], 300), color=(0, 128, 255), thickness=1)
    # cv2.rectangle(img, pt1=(200, 100), pt2=(img.shape[1], 200), color=(255, 0, 0), thickness=2)
    cv2.rectangle(img, pt1=(200, 100), pt2=(img.shape[1], 200), color=(255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(img, center=(50, 250), radius=30, color=(255, 255, 0), thickness=1)
    cv2.putText(img, 'opencv Hello World!', org=(200, 300),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(128, 255, 0), thickness=1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    draw_img()
