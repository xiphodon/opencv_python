import cv2
import numpy as np
from pyzbar import pyzbar


class CodeFinder:
    """
    二维码、条码检测
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

    def run(self):
        while True:
            success, img = self.cap.read()
            for bar_code in pyzbar.decode(img):
                # print(bar_code.data)  # 二维码数据
                # print(bar_code.type)  # 二维码类型
                # print(bar_code.rect)  # 二维码四周边界（矩形框）
                # print(bar_code.polygon)  # 二维码多边形边框
                # print(bar_code.quality)  # 二维码质量
                # print(bar_code.orientation)   # 二维码方向
                points = np.array(bar_code.polygon, np.int32)
                points = points.reshape((-1, 1, 2))
                # polylines 可以同时绘制多个多边形曲线
                cv2.polylines(img=img, pts=[points], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.putText(
                    img=img,
                    text=bar_code.data.decode('utf8'),
                    org=(bar_code.rect.left, bar_code.rect.top),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=2
                )

            cv2.imshow('code', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    code_finder = CodeFinder()
    code_finder.run()
