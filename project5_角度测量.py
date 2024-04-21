import cv2
import math


class AngleFinder:
    """
    角度测量
    """
    def __init__(self):
        self.img_raw = cv2.imread('./resources/angle.png')
        self.img = self.img_raw.copy()
        self.window_name = 'angle'
        self.point_list = list()

    def gradient(self, pt1, pt2):
        """
        梯度计算（斜率）
        :return:
        """
        dy = pt2[1] - pt1[1]
        dx = pt2[0] - pt1[0]
        if dx == 0:
            dx = 0.001
        return dy / dx

    def get_angle(self):
        """
        获取角度

        射线1与x轴夹角为A，斜率为tanA；
        射线2与x轴夹角为B，斜率为tanB，
        两射线夹角为 abs(A-B) 度。

        tan(A - B) = (tanA - tanB) / (1 + tanA * tanB)  =>
        弧度：r = A - B = arctan((tanA - tanB) / (1 + tanA * tanB))  =>
        角度：d = 180/pi * r
        :return:
        """
        pt1, pt2, pt3 = self.point_list[-3:]
        k1 = self.gradient(pt1, pt2)
        k2 = self.gradient(pt1, pt3)
        ang_r = math.atan((k2 - k1) / (1 + k1 * k2))
        ang_d = round(math.degrees(ang_r), 2)
        cv2.putText(self.img, f'{abs(ang_d)}', (pt1[0] - 60, pt1[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def mouse_callback(self, event, x, y, flags, params):
        """
        鼠标回调
        :return:
        """
        print(event, x, y, flags, params)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.point_list.append((x, y))

            if len(self.point_list) % 3 == 2:
                # 连接最后两个点
                cv2.line(self.img, self.point_list[-2], self.point_list[-1], (0, 0, 255), 2)
            elif len(self.point_list) % 3 == 0 and len(self.point_list) != 0:
                # 连接最后三个点中第一个和第三个点
                cv2.line(self.img, self.point_list[-3], self.point_list[-1], (0, 0, 255), 2)

            cv2.circle(self.img, (x, y), 5, (0, 0, 255), cv2.FILLED)

    def run(self):
        while True:
            if len(self.point_list) > 0 and len(self.point_list) % 3 == 0:
                # 点列表数量为三的倍数
                self.get_angle()

            cv2.imshow(self.window_name, self.img)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.point_list.clear()
                self.img = self.img_raw.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(self.window_name)
                break


if __name__ == '__main__':
    angle_finder = AngleFinder()
    angle_finder.run()
