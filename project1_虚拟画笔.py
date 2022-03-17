import cv2
import numpy as np

from utils import img_stack_util


class VirtualBrush:
    """
    虚拟画笔
    """
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.frame_width)
        self.cap.set(4, self.frame_height)
        self.cap.set(10, 500)

        self.track_windows_name = 'TrackBars'

        # 通过颜色检测，获取各个颜色hsv过滤上下限
        self.my_colors_list = [
            {
                'color_name': '玫红',
                'hsv_min': [132, 78, 71],
                'hsv_max': [179, 255, 255],
                'color_BGR': [85, 85, 205]
            },
            {
                'color_name': '柠檬黄',
                'hsv_min': [18, 98, 0],
                'hsv_max': [69, 255, 255],
                'color_BGR': [32, 165, 218]
            },
            {
                'color_name': '翡翠绿',
                'hsv_min': [30, 92, 0],
                'hsv_max': [90, 255, 255],
                'color_BGR': [154, 250, 0]
            },
            {
                'color_name': '钴蓝',
                'hsv_min': [101, 93, 86],
                'hsv_max': [151, 230, 255],
                'color_BGR': [112, 25, 25]
            }
        ]

        # 颜色绘制点， item:(x,y,color index)
        self.color_points_list = []

    @staticmethod
    def empty_on_change(value):
        """
        跟踪杆空回调
        :return:
        """
        pass

    def create_track_bar(self):
        """
        创建跟踪杆
        :return:
        """
        cv2.namedWindow(self.track_windows_name)
        cv2.resizeWindow(self.track_windows_name, 600, 300)

        # 色度
        cv2.createTrackbar('huv min', self.track_windows_name, 0, 179, self.empty_on_change)
        cv2.createTrackbar('huv max', self.track_windows_name, 179, 179, self.empty_on_change)
        # 饱和度
        cv2.createTrackbar('sat min', self.track_windows_name, 0, 255, self.empty_on_change)
        cv2.createTrackbar('sat max', self.track_windows_name, 255, 255, self.empty_on_change)
        # 纯度
        cv2.createTrackbar('val min', self.track_windows_name, 0, 255, self.empty_on_change)
        cv2.createTrackbar('val max', self.track_windows_name, 255, 255, self.empty_on_change)

    def color_detect(self):
        """
        颜色检测
        :return:
        """
        self.create_track_bar()

        while True:
            success, img = self.cap.read()
            # HSV 分别为 色调、饱和度、纯度
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h_min = cv2.getTrackbarPos('huv min', self.track_windows_name)
            h_max = cv2.getTrackbarPos('huv max', self.track_windows_name)
            s_min = cv2.getTrackbarPos('sat min', self.track_windows_name)
            s_max = cv2.getTrackbarPos('sat max', self.track_windows_name)
            v_min = cv2.getTrackbarPos('val min', self.track_windows_name)
            v_max = cv2.getTrackbarPos('val max', self.track_windows_name)

            print(h_min, h_max, s_min, s_max, v_min, v_max)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(img_hsv, lowerb=lower, upperb=upper)

            img_result = cv2.bitwise_and(img, img, mask=mask)

            big_img = img_stack_util.stack_img(([img, img_hsv], [mask, img_result]), scale=0.5)

            cv2.imshow('img', big_img)
            # cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def color_detect_show(self):
        """
        颜色提取显示
        :return:
        """
        while True:
            success, img = self.cap.read()
            # HSV 分别为 色调、饱和度、纯度
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            for item in self.my_colors_list[:1]:
                hsv_lower = np.array(item['hsv_min'])
                hsv_upper = np.array(item['hsv_max'])
                print(item['color_name'], hsv_lower, hsv_upper)
                mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
                img_result = cv2.bitwise_and(img, img, mask=mask)
                big_img = img_stack_util.stack_img(([img, img_hsv], [mask, img_result]), scale=0.5)
                cv2.imshow('img', big_img)
            # cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def find_color_points(self, img, img_result):
        """
        寻找颜色位置点
        :return:
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, item in enumerate(self.my_colors_list):
            hsv_lower = np.array(item['hsv_min'])
            hsv_upper = np.array(item['hsv_max'])
            mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(1)
            # 通过蒙板获取轮廓画笔绘制点
            x, y = self.get_contours_key_point(mask, img_result)
            # 绘制笔尖
            if x != 0 and y != 0:
                cv2.circle(img_result, (x, y), 10, item['color_BGR'], cv2.FILLED)
                self.color_points_list.append((x, y, i))

    def get_contours_key_point(self, img, img_result):
        """
        获取轮廓关键点
        :return:
        """
        # 参数：img，外部，轮廓线
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = 0, 0, 0, 0
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                if area > max_area:
                    max_area = area
                else:
                    continue
                # 周长
                peri = cv2.arcLength(curve=cnt, closed=True)
                # 近似多边形
                approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02*peri, closed=True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(image=img_result, contours=cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
                cv2.rectangle(img=img_result, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
        return x + w//2, y

    def draw_brush_stroke(self, img):
        """
        绘制画笔路径
        :return:
        """
        # 绘制圆点
        # for point in self.color_points_list:
        #     cv2.circle(img, (point[0], point[1]), 5, self.my_colors_list[point[2]]['color_BGR'], cv2.FILLED)

        # 绘制各个颜色轨迹（直线）
        # 拆分颜色轨迹
        split_color_points_dict = dict()
        for point in self.color_points_list:
            split_color_points_dict.setdefault(point[2], list()).append(point)

        for item_color_points in split_color_points_dict.values():
            for i, point in enumerate(item_color_points):
                if i > 0:
                    cv2.line(
                        img=img,
                        pt1=item_color_points[i - 1][:2],
                        pt2=item_color_points[i][:2],
                        color=self.my_colors_list[item_color_points[i][2]]['color_BGR'],
                        thickness=3
                    )

    def brush_draw(self):
        """
        画笔绘制
        :return:
        """
        while True:
            timer_1 = cv2.getTickCount()
            success, img = self.cap.read()
            # 水平反转180度（沿y轴）
            img = cv2.flip(img, 180)
            img_result = img.copy()
            self.find_color_points(img, img_result)
            self.draw_brush_stroke(img_result)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer_1)
            cv2.putText(img_result, f'{fps: 0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('img', img_result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                self.cap.release()
                cv2.destroyAllWindows()
                break

            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.color_points_list.clear()


if __name__ == '__main__':
    vb = VirtualBrush()
    # vb.color_detect()
    # vb.color_detect_show()
    vb.brush_draw()

