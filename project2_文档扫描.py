import cv2
import numpy as np

from utils import img_stack_util


class DocumentScanner:
    """
    文档扫描器
    """
    def __init__(self):
        self.width_img = 400
        self.height_img = 400
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(3, self.width_img)
        # self.cap.set(4, self.height_img)
        self.cap.set(10, 200)

    def pre_processing(self, img):
        """
        图片预处理
        :param img:
        :return:
        """
        kernel = np.ones((5, 5))
        # 灰度
        img_gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
        # 模糊
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        # 边缘
        img_canny = cv2.Canny(img_blur, 200, 200)
        # 膨胀
        img_dila = cv2.dilate(img_canny, kernel, iterations=2)
        # 腐蚀
        img_thres = cv2.erode(img_dila, kernel, iterations=1)
        return img_thres

    def get_max_contour(self, img, img_contour):
        """
        获取最大轮廓
        :param img:
        :param img_contour:
        :return:
        """
        max_approx = np.array([])
        max_area = 0
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                peri = cv2.arcLength(cnt, closed=True)
                # 多边形折线顶点
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if area > max_area and len(approx) == 4:
                    max_approx = approx
                    max_area = area
        cv2.drawContours(image=img_contour, contours=max_approx, contourIdx=-1, color=(255, 0, 0), thickness=20)
        return max_approx

    def points_reorder(self, points):
        """
        四顶点重排
        :param points:
        :return:
        """
        # points shape (4, 1, 2)
        _points = points.reshape((4, 2))
        points_new = np.zeros(points.shape, np.int32)
        # 将每个点坐标求和，可得到左上角点（最小）与右下角点（最大）, h + w
        points_add = _points.sum(axis=1)
        points_new[0] = _points[np.argmin(points_add)]
        points_new[3] = _points[np.argmax(points_add)]
        # 将每个点坐标求差，可得到左下角点（最小）与右上角点（最大）, h - w
        points_diff = np.diff(_points, axis=1)
        points_new[1] = _points[np.argmin(points_diff)]
        points_new[2] = _points[np.argmax(points_diff)]
        return points_new

    def get_warp(self, img, max_approx):
        """
        透视获取内容
        :param img:
        :param max_approx:
        :return:
        """
        max_approx = self.points_reorder(max_approx)
        # 轮廓四顶点
        points_1 = np.float32(max_approx)
        # 与轮廓四顶点对应的透视顶点
        points_2 = np.float32([
            [0, 0],
            [self.width_img, 0],
            [0, self.height_img],
            [self.width_img, self.height_img]
        ])
        matrix = cv2.getPerspectiveTransform(points_1, points_2)
        img_warp = cv2.warpPerspective(img, matrix, (self.width_img, self.height_img))

        # 裁剪四周
        img_cropped = img_warp[10: img_warp.shape[0]-10, 10: img_warp.shape[1]-10]
        img_output = cv2.resize(img_cropped, dsize=(self.width_img, self.height_img))

        # return img_output
        return img_warp

    def process(self, img):
        """
        处理流程
        :param img:
        :return:
        """
        img = cv2.resize(img, (self.width_img, self.height_img))
        img_show_contour = img.copy()
        img_pre_process = self.pre_processing(img)
        img_contour = self.get_max_contour(img_pre_process, img_show_contour)
        if img_contour.size != 0:
            img_content = self.get_warp(img, img_contour)
            img_tuple = ([img, img_pre_process], [img_show_contour, img_content])
        else:
            img_tuple = ([img, img_pre_process], [img_show_contour, img])
        stack_img = img_stack_util.stack_img(img_tuple, scale=0.8)
        cv2.imshow('img', stack_img)

    def run(self, video_flag=True):
        """
        启动
        :param video_flag:
        :return:
        """
        if video_flag:
            while True:
                success, img = self.cap.read()
                self.process(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # 有按键则返回按键ASCII码，无按键则返回-1
                    # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            self.cap.release()

            img = cv2.imread(r'./resources/paper.jpg')
            self.process(img)
            cv2.waitKey(0)


if __name__ == '__main__':
    document_scanner = DocumentScanner()
    document_scanner.run(video_flag=True)
    # document_scanner.run(video_flag=False)
