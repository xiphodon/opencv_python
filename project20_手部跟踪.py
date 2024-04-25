import cv2
import time

from utils.hands_detector import HandsDetector


class HandsTracking:
    """
    手部跟踪
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.hands_detector = HandsDetector()

    def run(self):
        last_time = time.time()
        while True:
            success, img = self.cap.read()
            if not success:
                break

            hand_landmarks_dict = self.hands_detector.detect_hands_landmarks(
                img=img,
                show_hand_connections=True,
                show_landmarks_id=True,
                show_landmarks=False

            )

            print(hand_landmarks_dict)

            current_time = time.time()
            fps = round(1.0 / (current_time - last_time), 2)
            last_time = current_time

            cv2.putText(img=img, text=f'fps: {fps}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=2)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break


if __name__ == '__main__':
    ht = HandsTracking()
    ht.run()
