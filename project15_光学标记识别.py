import cv2
import numpy as np
from utils.img_stack_util import stack_img


class OpticalMarkRecognition:
    """
    光学标记识别（OMR）
    答题卡识别
    """

    def __init__(self):
        self.file_path = r'./resources/omr.png'
        # self.img = cv2.imread(self.file_path)
        self.img_width = 700
        self.img_height = 700
        self.grade_area_width = 325
        self.grade_area_height = 150
        self.questions = 5
        self.choices = 5
        self.answer = np.array([[0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1]], np.uint8)
        self.mark_answer_px_threshold = 5000

    def points_reorder(self, points):
        """
        四顶点重排
        :param points:
        :return:
        """
        # points shape (4, 1, 2)
        _points = points.reshape((4, 2))
        points_new = np.zeros_like(_points, np.int32)
        # 将每个点坐标求和，可得到左上角点（最小）与右下角点（最大）, x + y
        points_add = _points.sum(axis=1)
        points_new[0] = _points[np.argmin(points_add)]
        points_new[3] = _points[np.argmax(points_add)]
        # 将每个点坐标求差，可得到左下角点（最大）与右上角点（最小）, y - x
        points_diff = np.diff(_points, axis=1)
        points_new[1] = _points[np.argmin(points_diff)]
        points_new[2] = _points[np.argmax(points_diff)]
        return points_new.reshape(points.shape)

    def get_warp(self, points, warp_shape_w_h, img, draw_img, color):
        """
        获取内部透视区域
        :param points: 坐标点
        :param warp_shape_w_h: 透视内容最终显示宽高
        :param img: 原图
        :param draw_img: 绘制图
        :param color: 颜色
        :return:
        """
        cv2.drawContours(image=draw_img, contours=[points], contourIdx=-1, color=color,
                         thickness=10)
        points = self.points_reorder(points)
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [warp_shape_w_h[0], 0], [0, warp_shape_w_h[1]],
                           [warp_shape_w_h[0], warp_shape_w_h[1]]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(img, matrix, warp_shape_w_h)
        return warp, matrix, pts1, pts2

    def draw_answers(self, mark_warp_img, mark_answers):
        """
        绘制答案
        :param mark_warp_img:
        :param mark_answers:
        :return:
        """
        box_h = mark_warp_img.shape[0] / self.questions
        box_w = mark_warp_img.shape[1] / self.choices

        for i in range(self.questions):
            for j in range(self.choices):
                real_answer = self.answer[i, j]
                mark_answer = mark_answers[i, j]

                c_x = int(box_w * j + box_w / 2)
                c_y = int(box_h * i + box_h / 2)
                radius_big = int(min(box_w, box_h) / 2 * 0.7)
                radius_small = int(min(box_w, box_h) / 2 * 0.2)

                if real_answer == 1 and mark_answer == 1:
                    # 填图正确
                    cv2.circle(img=mark_warp_img, center=(c_x, c_y),
                               radius=radius_big, color=(0, 255, 0), thickness=cv2.FILLED)
                if real_answer == 1 and mark_answer == 0:
                    # 少填
                    cv2.circle(img=mark_warp_img, center=(c_x, c_y),
                               radius=radius_small, color=(0, 255, 0), thickness=cv2.FILLED)
                if real_answer == 0 and mark_answer == 1:
                    # 错填
                    cv2.circle(img=mark_warp_img, center=(c_x, c_y),
                               radius=radius_big, color=(0, 0, 255), thickness=cv2.FILLED)

    def run(self):
        while True:
            img = cv2.imread(self.file_path)
            img = cv2.resize(src=img, dsize=(self.img_width, self.img_height))
            img_copy_contour = img.copy()
            img_result = img.copy()
            # img_blank = np.zeros_like(img, np.uint8)

            # 图像预处理
            img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(src=img_gray, ksize=(5, 5), sigmaX=1)
            img_canny = cv2.Canny(image=img_blur, threshold1=10, threshold2=70)

            # 获取轮廓
            contours, hierarchy = cv2.findContours(image=img_canny,
                                                   mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(image=img_copy_contour,
            #                  contours=contours,
            #                  contourIdx=-1,
            #                  color=(0, 255, 0),
            #                  thickness=5)

            # 轮廓筛选，并按照面积降序排序
            contours_data = list()
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    arc_length = cv2.arcLength(curve=contour, closed=True)
                    approx = cv2.approxPolyDP(curve=contour, epsilon=0.02 * arc_length, closed=True)
                    if len(approx) == 4:
                        contours_data.append([area, approx])
            contours_data = sorted(contours_data, key=lambda x: x[0], reverse=True)
            # print(contours_data)

            # 涂卡区、评分区，绘制
            mark_area_contour_data = contours_data[0]
            mark_area, mark_area_points = mark_area_contour_data[:2]
            mark_warp_img, mark_warp_matrix, mark_warp_pts1, mark_warp_pts2 = self.get_warp(
                points=mark_area_points,
                warp_shape_w_h=(
                    self.img_width,
                    self.img_height),
                img=img,
                draw_img=img_copy_contour,
                color=(0, 255, 0))

            mark_warp_img_copy = mark_warp_img.copy()

            grade_area_contour_data = contours_data[1]
            grade_area, grade_area_points = grade_area_contour_data[:2]
            grade_warp_img, grade_warp_matrix, grade_warp_pts1, grade_warp_pts2 = self.get_warp(
                points=grade_area_points,
                warp_shape_w_h=(self.grade_area_width,
                                self.grade_area_height),
                img=img,
                draw_img=img_copy_contour,
                color=(255, 0, 0))

            mark_warp_img_gray = cv2.cvtColor(src=mark_warp_img, code=cv2.COLOR_BGR2GRAY)
            _, mark_warp_img_binary = cv2.threshold(src=mark_warp_img_gray, thresh=170, maxval=255,
                                                    type=cv2.THRESH_BINARY_INV)

            # 分割涂卡区
            boxes_data_arr = np.zeros(shape=(self.questions, self.choices), dtype=np.uint16)
            mark_area_rows = np.vsplit(mark_warp_img_binary, self.questions)
            for i, row in enumerate(mark_area_rows):
                items = np.hsplit(row, self.choices)
                for j, box in enumerate(items):
                    no_zero_px_value = cv2.countNonZero(box)
                    boxes_data_arr[i, j] = no_zero_px_value
            # 涂卡区亮部大于阈值则表示填图该选项（用1表示填图）
            boxes_data_arr_binary = np.where(boxes_data_arr > self.mark_answer_px_threshold, 1, 0)
            # 对比答案，将不一致填图位置标记为1
            compare_arr = np.where(boxes_data_arr_binary == self.answer, 0, 1)
            # 按照每道题求和，将和为0（全匹配，即题目做对）的题标记为1，其余多选少选错题标记为0，得到题目对错数组
            questions_arr = np.where(compare_arr.sum(1) == 0, 1, 0)
            # 计算得分
            score = (np.sum(questions_arr) / questions_arr.size) * 100

            # 在答题区透视图和答题区蒙板上绘制答案
            self.draw_answers(mark_warp_img_copy, boxes_data_arr_binary)
            mark_warp_mask = np.zeros_like(mark_warp_img_copy)
            self.draw_answers(mark_warp_mask, boxes_data_arr_binary)

            # 将答题区绘制好的蒙板透视回去
            mark_mask_warp_matrix_inv = cv2.getPerspectiveTransform(mark_warp_pts2, mark_warp_pts1)
            mark_mask_img = cv2.warpPerspective(src=mark_warp_mask, M=mark_mask_warp_matrix_inv,
                                                dsize=(self.img_width, self.img_height))

            # 在分数区和分数区蒙板绘制答案
            grade_warp_mask = np.zeros_like(grade_warp_img)
            cv2.putText(img=grade_warp_mask, text=f'{score} %',
                        org=(self.grade_area_width // 3, self.grade_area_height * 2 // 3),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 255), thickness=5)
            grade_mask_warp_matrix_inv = cv2.getPerspectiveTransform(grade_warp_pts2, grade_warp_pts1)
            grade_mask_img = cv2.warpPerspective(src=grade_warp_mask, M=grade_mask_warp_matrix_inv,
                                                 dsize=(self.img_width, self.img_width))

            # 将绘制蒙板添加至原图上
            img_result = cv2.addWeighted(src1=img_result, alpha=1, src2=mark_mask_img, beta=1, gamma=0)
            img_result = cv2.addWeighted(src1=img_result, alpha=1, src2=grade_mask_img, beta=1, gamma=0)

            img_final = stack_img(img_arr=([img, img_canny, img_copy_contour, mark_warp_img_gray, mark_warp_img_binary],
                                           [mark_warp_img_copy, mark_warp_mask, mark_mask_img, grade_mask_img,
                                            img_result]), scale=0.4,
                                  labels=[['img', 'img_canny', 'img_copy_contour', 'mark_warp_img_gray',
                                           'mark_warp_img_binary'],
                                          ['mark_warp_img_copy', 'mark_warp_mask', 'mark_mask_img', 'grade_mask_img',
                                           'img_result']])

            # cv2.imshow('img_canny', img_canny)
            # cv2.imshow('img_contours', img_copy_contour)
            # cv2.imshow('mark_warp_gray', mark_warp_img_gray)
            # cv2.imshow('mark_warp_binary', mark_warp_img_binary)
            # cv2.imshow('mark_warp_copy', mark_warp_img_copy)
            # cv2.imshow('mark_warp_mask', mark_warp_mask)
            # cv2.imshow('mark_mask_img', mark_mask_img)
            # cv2.imshow('grade_mask_img', grade_mask_img)
            # cv2.imshow('img_result', img_result)
            cv2.imshow('img_final', img_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    omr = OpticalMarkRecognition()
    omr.run()
