from pathlib import Path

import cv2
import numpy as np


class RussianPlateNumberDetection:
    """
    俄罗斯车牌数字检测
    """
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.plate_cascade = cv2.CascadeClassifier(r"./resources/haarcascade_russian_plate_number.xml")
        self.min_area = 200
        self.color = (255, 0, 255)

        self.cap = cv2.VideoCapture(r'./resources/video.mp4')
        self.cap.set(3, self.frame_width)
        self.cap.set(4, self.frame_height)
        self.cap.set(10, 200)
        self.type_count = 0

    def detect(self):
        """
        检测
        :return:
        """
        while True:
            success, img = self.cap.read()
            if not success:
                break
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            number_plates = self.plate_cascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=10)
            img_roi = np.array([])
            for x, y, w, h in number_plates:
                area = w * h
                if area > self.min_area:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(img, 'number plate', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
                    img_roi = img[y: y+h, x: x+w]
                    cv2.imshow('roi', img_roi)
            cv2.imshow('result', img)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.rectangle(img, (0, 200), (self.frame_width, 300), (0, 255, 0), cv2.FILLED)
                if img_roi.size != 0:
                    cv2.imwrite(rf'./temp/number_plate_{self.type_count}.jpg', img_roi)
                    cv2.putText(img, 'scan saved', (150, 265), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(img, 'scan error', (150, 265), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow('result', img)
                cv2.waitKey(500)
                self.type_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    rpnd = RussianPlateNumberDetection()
    rpnd.detect()
