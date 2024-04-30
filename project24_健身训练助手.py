import math
from pathlib import Path

import cv2
import time

import numpy as np

from utils.pose_detector import PoseDetector


class Trainer:
    """
    健身助手
    """
    def __init__(self, video_num_or_path: int or str):
        self.pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.cap = cv2.VideoCapture(video_num_or_path)

        self.pose_landmarks_dict = None

        self.direction_up = 0
        self.direction_down = 1
        self.direction_status = self.direction_up

        self.progress_color_1 = (255, 0, 255)
        self.progress_color_2 = (0, 255, 0)

        self.count = 0

    def find_angle(self, img, p0_index, p1_index, p2_index, is_show=True):
        """
        找寻指定关键点组成角的角度
        :param img:
        :param p0_index: 角点坐标索引
        :param p1_index: 一边上一点坐标索引
        :param p2_index: 另一边上一点坐标索引
        :param is_show:
        :return:
        """
        if not self.pose_landmarks_dict:
            return 0

        p0 = self.pose_landmarks_dict[p0_index]
        p1 = self.pose_landmarks_dict[p1_index]
        p2 = self.pose_landmarks_dict[p2_index]
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        # angle radians in [-pi/2, pi/2]
        angle_radians = math.atan2(y1 - y0, x1 - x0) - math.atan2(y2 - y0, x2 - x0)
        # angle radians in [0, pi/2]
        angle_radians_abs = abs(angle_radians)
        angle = math.degrees(angle_radians_abs)

        if angle > 180:
            angle = 360 - angle

        if is_show:
            cv2.line(img=img, pt1=p0, pt2=p1, color=(0, 255, 0), thickness=2)
            cv2.line(img=img, pt1=p0, pt2=p2, color=(0, 255, 0), thickness=2)
            cv2.circle(img=img, center=p0, radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=p1, radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=p2, radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=p0, radius=15, color=(0, 0, 255), thickness=2)
            cv2.circle(img=img, center=p1, radius=15, color=(0, 0, 255), thickness=2)
            cv2.circle(img=img, center=p2, radius=15, color=(0, 0, 255), thickness=2)
            cv2.putText(img=img, text=f'{int(angle)}', org=(p0[0] + 10, p0[1] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 255), thickness=2)
        return angle

    def draw_progress_and_count(self, img, angle):
        """
        绘制进度条并计数
        :param img:
        :param angle:
        :return:
        """
        progress = 100 - np.interp(x=angle, xp=[50, 150], fp=[0, 100])
        progress_height_px = np.interp(x=angle, xp=[50, 150], fp=[100, 280])

        color = self.progress_color_1

        if progress == 0 and self.direction_status == self.direction_down:
            self.direction_status = self.direction_up
        if progress == 100 and self.direction_status == self.direction_up:
            self.count += 1
            self.direction_status = self.direction_down
        if progress == 100:
            color = self.progress_color_2

        progress_box_top_left_point = [580, 100]
        progress_box_bottom_right_point = [600, 280]
        progress_bar_top_left_point = [580, int(progress_height_px)]
        progress_bar_bottom_right_point = progress_box_bottom_right_point

        cv2.rectangle(img=img, pt1=progress_box_top_left_point, pt2=progress_box_bottom_right_point,
                      color=color, thickness=2)
        cv2.rectangle(img=img, pt1=progress_bar_top_left_point, pt2=progress_bar_bottom_right_point,
                      color=color, thickness=cv2.FILLED)
        cv2.putText(img=img, text=f'{int(progress)}%',
                    org=(progress_box_top_left_point[0] - 5, progress_box_top_left_point[1] - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

        cv2.putText(img=img, text=f'{self.count}', org=(580, 330), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=3)

    def run(self):
        last_time = time.time()
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.resize(src=img, dsize=(640, 360))

            self.pose_landmarks_dict = self.pose_detector.detect_pose_landmarks(
                img=img,
                show_pose_connections=False,
                show_landmarks=False,
                show_landmarks_id=False
            )

            angle = self.find_angle(
                img=img,
                p0_index=13,
                p1_index=11,
                p2_index=15,
                is_show=True
            )

            self.draw_progress_and_count(img=img, angle=angle)

            current = time.time()
            fps = round(1.0 / (current - last_time), 2)
            last_time = current

            cv2.putText(img=img, text=f'fps: {fps}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(0, 0, 0), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        self.cap.release()


if __name__ == '__main__':
    video_path = Path(r'./resources/curls.mp4')
    trainer = Trainer(video_num_or_path=video_path.as_posix())
    trainer.run()
