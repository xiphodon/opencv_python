import json
import time
from pathlib import Path

import cv2
import numpy as np

class CarParking:
    """
    汽车车位识别
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(Path(r'./resources/car_parking/carPark.mp4').as_posix())
        self.car_space_bbox_shape = (100, 40)

        self.car_space_list_json_path = Path(r'./resources/car_parking/car_space.json')
        self.car_space_pos_list = list()

        self.car_space_item_img_list = list()

        self.window_name = 'threshold'
        self.px_count_threshold = 800

        self.read_car_space_pos_list()

    def read_car_space_pos_list(self):
        """
        读取车位位置列表
        :return:
        """
        if self.car_space_list_json_path.exists():
            with open(self.car_space_list_json_path.as_posix(), 'r', encoding='utf8') as fp:
                self.car_space_pos_list = json.load(fp)

    def mouse_onclick(self, events, x, y, flags, params):
        """
        鼠标点击事件
        :param events:
        :param x:
        :param y:
        :param flags:
        :param params:
        :return:
        """
        if events == cv2.EVENT_LBUTTONDOWN:
            self.car_space_pos_list.append((x, y))
        if events == cv2.EVENT_RBUTTONDOWN:
            remove_pos_index = None
            for i in range(len(self.car_space_pos_list)):
                _x, _y = self.car_space_pos_list[i]
                if _x < x < _x + self.car_space_bbox_shape[0] and _y < y < _y + self.car_space_bbox_shape[1]:
                    remove_pos_index = i
                    break
            if remove_pos_index is not None:
                self.car_space_pos_list.pop(remove_pos_index)

    def run(self):

        self.init_threshold_track_bar()

        while self.cap.isOpened():

            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            success, img = self.cap.read()
            if not success:
                break

            img_mask = self.preprocessing(img=img)
            self.split_parking_space(img=img, mask=img_mask)

            # cv2.rectangle(img=img, pt1=(50, 200), pt2=(150, 240), color=(0, 255, 0), thickness=2)

            cv2.imshow('parking', img)
            cv2.setMouseCallback('parking', self.mouse_onclick)
            key = cv2.waitKey(10)
            if key & 0xff == ord('q'):
                break
            if key & 0xff == ord('s'):
                with open(self.car_space_list_json_path.as_posix(), 'w', encoding='utf8') as fp:
                    json.dump(self.car_space_pos_list, fp)
                print('car space info saved')
                break
        self.cap.release()

    def preprocessing(self, img):
        """
        预处理
        :param img:
        :return:
        """
        threshold_min = cv2.getTrackbarPos(trackbarname='min', winname=self.window_name)
        threshold_max = cv2.getTrackbarPos(trackbarname='max', winname=self.window_name)

        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=1)
        img_canny = cv2.Canny(image=img, threshold1=threshold_min, threshold2=threshold_max)
        img = cv2.morphologyEx(src=img_canny, op=cv2.MORPH_CLOSE, kernel=(7, 7), iterations=5)
        # img = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=(7, 7), iterations=2)
        img = cv2.dilate(src=img, kernel=(5, 5), iterations=2)
        img = cv2.medianBlur(src=img, ksize=3)

        cv2.imshow('canny', img_canny)
        cv2.imshow('preprocess', img)
        return img

    def init_threshold_track_bar(self):
        """
        初始化阈值滑动条
        :return:
        """
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(winname=self.window_name, width=640, height=240)
        cv2.createTrackbar('min', self.window_name, 40, 255, lambda _: 0)
        cv2.createTrackbar('max', self.window_name, 180, 255, lambda _: 0)

    def split_parking_space(self, img, mask):
        """
        分割停车空间
        :param img:
        :param mask:
        :return:
        """
        w, h = self.car_space_bbox_shape
        for i, (x, y) in enumerate(self.car_space_pos_list):
            item_img = img[y: y + h, x: x + w]
            item_mask = mask[y: y + h, x: x + w]
            # cv2.imshow(f'car_parking_{i}', item_img)
            px_count = cv2.countNonZero(src=item_mask)
            if px_count > self.px_count_threshold:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.putText(img=img, text=f'{px_count}', org=(x + 4, y + h - 4), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=color, thickness=2)
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)


if __name__ == '__main__':
    cp = CarParking()
    cp.run()
