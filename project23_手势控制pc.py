import math
import time
from typing import List

import cv2
import numpy as np
import pyautogui

from utils.hands_detector import HandsDetector


class GestureControl:
    """
    手势控制
    """
    def __init__(self):
        self.camera_width, self.camera_height = (800, 480)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=self.camera_width)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=self.camera_height)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=200)
        self.hands_detector = HandsDetector(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        # HandsDetector 探测返回的手部地标字典
        self.hands_detector_dict = dict()

        # 拇指、食指、中指指尖在摄像头画面中的坐标
        self.thumb_tip = list()
        self.index_finger_tip = list()
        self.middle_finger_tip = list()

        # 摄像头画面中“鼠标”的坐标（拇指、食指指尖连线的中点坐标）
        self.index_point = list()

        # 摄像头画面中“鼠标”左键未按下与按下时，画面中对应光点颜色
        self.index_point_color_unselected = (255, 255, 0)
        self.index_point_color_selected = (255, 0, 255)
        self.index_point_color = self.index_point_color_unselected

        # 拇指指尖与拇指远心端第一个关节的距离
        self.thumb_first_joint_len = 0

        # pyautogui 各个命令间停顿时间设置为0
        pyautogui.PAUSE = 0
        self.screen_width, self.screen_height = pyautogui.size()

        # 鼠标平滑系数
        self.mouse_smoothing = 5

        # 上一次鼠标在屏幕上的坐标
        self.last_mouse_point = [self.screen_width/2, self.screen_height/2]

        # 摄像头画面内边距，内边距内为手势可操作区域
        self.camera_padding = 100

        # 鼠标左键抬起、按下状态
        self.mouse_left_status_up = 1
        self.mouse_left_status_down = 2
        self.mouse_left_status = self.mouse_left_status_up

        # 鼠标右键抬起、按下状态
        self.mouse_right_status_up = 1
        self.mouse_right_status_down = 2
        self.mouse_right_status = self.mouse_right_status_up

    @staticmethod
    def two_point_distance(pt1: List[int], pt2: List[int]):
        """
        两点距离
        :param pt1:
        :param pt2:
        :return:
        """
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def extract_gesture_finger_keypoint(self):
        """
        提取手势手指关键点
        :return:
        """
        if self.hands_detector_dict:
            self.thumb_tip = self.hands_detector_dict[0][4]
            self.index_finger_tip = self.hands_detector_dict[0][8]
            self.middle_finger_tip = self.hands_detector_dict[0][12]
            # 拇指远心端第一个关节坐标
            thumb_ip = self.hands_detector_dict[0][3]

            self.index_point = [
                (self.index_finger_tip[0] + self.thumb_tip[0]) / 2,
                (self.index_finger_tip[1] + self.thumb_tip[1]) / 2
            ]

            self.thumb_first_joint_len = self.two_point_distance(self.thumb_tip, thumb_ip)

    def move_mouse_point(self):
        """
        移动鼠标
        :return:
        """
        if not self.hands_detector_dict:
            return

        # 屏幕上鼠标需要移动到的位置坐标
        screen_index_point = self.camera_to_screen_point(camera_point=self.index_point)

        # 鼠标平滑防抖
        last_x, last_y = self.last_mouse_point
        current_x, current_y = screen_index_point
        smoothing_screen_index_point_x = last_x + (current_x - last_x) / self.mouse_smoothing
        smoothing_screen_index_point_y = last_y + (current_y - last_y) / self.mouse_smoothing
        screen_index_point = [smoothing_screen_index_point_x, smoothing_screen_index_point_y]

        pyautogui.moveTo(x=screen_index_point[0], y=screen_index_point[1], duration=0.02)
        self.last_mouse_point = screen_index_point

    def camera_to_screen_point(self, camera_point: List[int]):
        """
        相机坐标对应到屏幕等比例位置
        :param camera_point: 相机中坐标
        :return:
        """
        # 屏幕边角安全像素距离，防止鼠标隐藏在屏幕边角处
        safe_px = 10

        # 插值，将摄像头画面中操作区域的坐标映射在屏幕安全区域内的坐标
        screen_x = np.interp(
            x=camera_point[0],
            xp=[self.camera_padding, self.camera_width - self.camera_padding],
            fp=[safe_px, self.screen_width - safe_px],
            left=safe_px,
            right=self.screen_width - safe_px
        )
        screen_y = np.interp(
            x=camera_point[1],
            xp=[self.camera_padding, self.camera_height - self.camera_padding],
            fp=[safe_px, self.screen_height - safe_px],
            left=safe_px,
            right=self.screen_height - safe_px
        )
        return [screen_x, screen_y]

    def control_mouse_button(self):
        """
        点击鼠标
        鼠标左键点击处理（拇指与食指两指捏合）
        鼠标右键点击处理（拇指、食指和中指三指捏合）
        :return:
        """
        if self.hands_detector_dict:
            # 拇指、食指指尖距离
            thumb_index_tip_distance = self.two_point_distance(self.thumb_tip, self.index_finger_tip)
            # 拇指、中指指尖距离
            thumb_middle_tip_distance = self.two_point_distance(self.thumb_tip, self.middle_finger_tip)

            # 判断鼠标按下、抬起的距离阈值
            # 大于大值时判定张开，小于小值时判定闭合，大小阈值中间为过渡区域
            # 【防抖处理，设计大小阈值是防止距离在单阈值上下附近判定状态反复切换】
            threshold_len_min = self.thumb_first_joint_len * 0.5
            threshold_len_max = self.thumb_first_joint_len * 0.6

            if thumb_index_tip_distance >= threshold_len_max and thumb_middle_tip_distance >= threshold_len_max:
                # 拇指食指张开，拇指中指张开
                if self.mouse_left_status == self.mouse_left_status_down:
                    # 若当前左键为按下状态，则抬起鼠标左键并更新状态
                    pyautogui.mouseUp(button='left')
                    self.mouse_left_status = self.mouse_left_status_up
                if self.mouse_right_status == self.mouse_right_status_down:
                    # 若当前右键为按下状态，则抬起鼠标右键并更新状态
                    pyautogui.mouseUp(button='right')
                    self.mouse_right_status = self.mouse_right_status_up
                self.index_point_color = self.index_point_color_unselected
            elif thumb_index_tip_distance < threshold_len_min and thumb_middle_tip_distance >= threshold_len_max:
                # 拇指食指闭合，拇指中指打开【双指捏合】
                if self.mouse_left_status == self.mouse_left_status_up:
                    # 若当前左键为抬起状态，则按下鼠标左键并更新状态
                    pyautogui.mouseDown(button='left')
                    self.mouse_left_status = self.mouse_left_status_down
                if self.mouse_right_status == self.mouse_right_status_down:
                    # 若当前右键为按下状态，则抬起鼠标右键并更新状态
                    pyautogui.mouseUp(button='right')
                    self.mouse_right_status = self.mouse_right_status_up
                self.index_point_color = self.index_point_color_selected
            elif thumb_index_tip_distance < threshold_len_min and thumb_middle_tip_distance < threshold_len_min:
                # 拇指食指闭合，拇指中指闭合【三指捏合】
                if self.mouse_left_status == self.mouse_left_status_up:
                    # 若当前左键为抬起状态，则按下鼠标左键并更新状态
                    pyautogui.mouseDown(button='left')
                    self.mouse_left_status = self.mouse_left_status_down
                if self.mouse_right_status == self.mouse_right_status_up:
                    # 若当前右键为抬起状态，则按下鼠标右键并更新状态
                    pyautogui.mouseDown(button='right')
                    self.mouse_right_status = self.mouse_right_status_down
            else:
                # 其余情况不做处理
                pass

    def run(self):
        last_time = time.time()
        # 初始化鼠标位置为屏幕中间
        pyautogui.moveTo(x=self.last_mouse_point[0], y=self.last_mouse_point[1])

        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.flip(src=img, flipCode=1)
            img = cv2.resize(src=img, dsize=(self.camera_width, self.camera_height))

            self.hands_detector_dict = self.hands_detector.detect_hands_landmarks(
                img=img,
                show_hand_connections=True,
                show_landmarks=False,
                show_landmarks_id=False
            )

            self.extract_gesture_finger_keypoint()
            self.move_mouse_point()
            self.control_mouse_button()

            current_time = time.time()
            fps = round(1.0 / (current_time - last_time), 2)
            last_time = current_time

            if self.hands_detector_dict:
                cv2.line(img=img, pt1=np.int32(self.index_finger_tip), pt2=np.int32(self.thumb_tip),
                         color=(255, 0, 0), thickness=1)
                cv2.circle(img=img, center=np.int32(self.index_point), radius=3, color=self.index_point_color,
                           thickness=cv2.FILLED)

            cv2.putText(img=img, text=f'fps: {fps}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(0, 0, 0), thickness=2)
            cv2.rectangle(img=img, pt1=(self.camera_padding, self.camera_padding),
                          pt2=(self.camera_width - self.camera_padding, self.camera_height - self.camera_padding),
                          color=(255, 0, 0), thickness=1)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        self.cap.release()


if __name__ == '__main__':
    gc = GestureControl()
    gc.run()
