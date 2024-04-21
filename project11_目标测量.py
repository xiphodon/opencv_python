import cv2
import numpy as np


class ObjectMeasurement:
    """
    目标测量
    """

    def __init__(self, is_camera=True):
        self.is_camera = is_camera
        self.file_path = r'./resources/a4_paper.png'
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
        self.scale = 2
        # a4纸同比例尺寸
        self.w_px = 210 * self.scale
        self.h_px = 297 * self.scale

    def get_contours(self, img, canny_threshold=(100, 100), area_threshold=1000,
                     angle_filter=0, show_canny=False, show_contours=False):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, canny_threshold[0], canny_threshold[1])
        kernel = np.ones(shape=(5, 5))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=3)
        img_erode = cv2.erode(img_dilate, kernel, iterations=2)
        if show_canny:
            cv2.imshow('canny', img_erode)

        contours, hiearchy = cv2.findContours(image=img_erode, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        result_contours = list()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                # 周长
                peri = cv2.arcLength(contour, True)
                # 近似多边形取样点
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                # 包含取样点的最小矩形框
                bbox = cv2.boundingRect(approx)
                if len(approx) == angle_filter > 0 or angle_filter == 0:
                    # 满足边角数量约束或者无边角数量约束
                    result_contours.append([len(approx), area, approx, bbox, contour])
                    if show_contours:
                        cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        # 按照面积区域降序排列
        result_contours = sorted(result_contours, key=lambda x: x[1], reverse=True)
        return img, result_contours

    def points_reorder(self, points):
        """
        四顶点重排
        :param points:
        :return:
        """
        # points shape (4, 1, 2)
        _points = points.reshape((4, 2))
        points_new = np.zeros(points.shape, np.int32)
        # 将每个点坐标求和，可得到左上角点（最小）与右下角点（最大）, x + y
        points_add = _points.sum(axis=1)
        points_new[0] = _points[np.argmin(points_add)]
        points_new[3] = _points[np.argmax(points_add)]
        # 将每个点坐标求差，可得到左下角点（最大）与右上角点（最小）, y - x
        points_diff = np.diff(_points, axis=1)
        points_new[1] = _points[np.argmin(points_diff)]
        points_new[2] = _points[np.argmax(points_diff)]
        # print(points_new)
        return points_new

    def warp_img(self, img, points, w, h, padding=20):
        points = self.points_reorder(points)
        points_1 = np.float32(points)
        points_2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(points_1, points_2)
        img_warp = cv2.warpPerspective(img, matrix, (w, h))
        img_crop = img_warp[padding: img_warp.shape[0] - padding, padding: img_warp.shape[1] - padding]
        return img_crop

    def points_distance(self, point_1, point_2):
        return np.sqrt(np.sum(np.power(point_1 - point_2, 2)))

    def run(self):
        while True:
            if self.is_camera:
                success, img = self.cap.read()
            else:
                img = cv2.imread(self.file_path)

            img_contour, contours_data = self.get_contours(img, canny_threshold=(50, 50),
                                                           area_threshold=3000, angle_filter=4,
                                                           show_contours=False)
            if len(contours_data) > 0:
                # 检测a4纸并透视提取出
                biggest_approx_points = contours_data[0][2]
                paper_a4_img = self.warp_img(img, biggest_approx_points, self.w_px, self.h_px)
                img_contour_2, contours_data_2 = self.get_contours(paper_a4_img, area_threshold=200,
                                                                   angle_filter=4, canny_threshold=(50, 50))
                if len(contours_data_2) > 0:
                    # 检测a4纸中的物体尺寸
                    for obj in contours_data_2:
                        angle_count, area, approx_points, bbox, contour = obj
                        cv2.polylines(paper_a4_img, [approx_points], True, (0, 255, 0), 2)

                        obj_point = self.points_reorder(approx_points)
                        obj_point_left_top = obj_point[0][0]
                        obj_point_right_top = obj_point[1][0]
                        obj_point_left_bottom = obj_point[2][0]
                        w_mm = round(self.points_distance(obj_point_left_top, obj_point_right_top) // self.scale, 1)
                        h_mm = round(self.points_distance(obj_point_left_top, obj_point_left_bottom) // self.scale, 1)

                        cv2.arrowedLine(img=paper_a4_img, pt1=obj_point_left_top, pt2=obj_point_right_top,
                                        color=(255, 0, 255), thickness=3, line_type=cv2.LINE_8, shift=0, tipLength=0.05)
                        cv2.arrowedLine(img=paper_a4_img, pt1=obj_point_left_top, pt2=obj_point_left_bottom,
                                        color=(255, 0, 255), thickness=3, line_type=cv2.LINE_8, shift=0, tipLength=0.05)

                        x, y, w, h = bbox
                        cv2.putText(img=paper_a4_img, text=f'{w_mm} mm', org=(x + w // 2, y - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=2)
                        cv2.putText(img=paper_a4_img, text=f'{h_mm} mm', org=(x - 70, y + h // 2),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=2)

                cv2.imshow('A4 paper', paper_a4_img)
            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    om = ObjectMeasurement(is_camera=False)
    om.run()
