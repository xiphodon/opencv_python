import cv2
import time

from utils.pose_detector import PoseDetector


class PoseTracking:
    """
    姿态跟踪
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.pose_detector = PoseDetector()

    def run(self):
        last_time = time.time()
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break

            pose_landmarks_dict = self.pose_detector.detect_pose_landmarks(
                img=img,
                show_pose_connections=True,
                show_landmarks=True,
                show_landmarks_id=True
            )
            print(pose_landmarks_dict)

            current_time = time.time()
            fps = round(1.0 / (current_time - last_time), 2)
            last_time = current_time

            cv2.putText(img=img, text=f'fps: {fps}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(0, 0, 0), thickness=2)
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        self.cap.release()


if __name__ == '__main__':
    pt = PoseTracking()
    pt.run()
