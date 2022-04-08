import pytesseract
import cv2
import numpy as np

from utils.img_stack_util import stack_img


class HighlightedTextDetection:
    """
    高亮文本检测
    """

    def __init__(self):
        self.document_img_path = r'./resources/document.png'
        pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

        # 通过detect_object_hsv_value找到hsv上下限
        self.hsv_lower = [0, 30, 90]
        self.hsv_upper = [75, 230, 255]
        self.img = cv2.imread(self.document_img_path)
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    @staticmethod
    def empty(v):
        pass

    def detect_object_hsv_value(self):
        """
        检测目标hsv值
        :return:
        """
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, self.empty)
        cv2.createTrackbar("Hue Max", "TrackBars", 75, 179, self.empty)
        cv2.createTrackbar("Sat Min", "TrackBars", 30, 255, self.empty)
        cv2.createTrackbar("Sat Max", "TrackBars", 230, 255, self.empty)
        cv2.createTrackbar("Val Min", "TrackBars", 90, 255, self.empty)
        cv2.createTrackbar("Val Max", "TrackBars", 255, 255, self.empty)

        while True:
            img = self.img
            h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
            h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
            s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
            s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
            v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
            v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
            print(h_min, h_max, s_min, s_max, v_min, v_max)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(self.img_hsv, lower, upper)
            img_result = cv2.bitwise_and(img, img, mask=mask)

            big_img = stack_img(img_arr=(img, mask, img_result), scale=0.7)
            cv2.imshow("Stacked Images", big_img)

            cv2.waitKey(1)

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
                approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
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

    def detect_color_area_img(self):
        """
        检测颜色区域
        :return:
        """
        mask = cv2.inRange(self.img_hsv, np.array(self.hsv_lower), np.array(self.hsv_upper))
        img_result = cv2.bitwise_and(self.img, self.img, mask=mask)
        # cv2.imshow('img_result', img_result)
        # cv2.waitKey(0)
        return img_result

    def run(self):
        img_result = self.detect_color_area_img()
        img_copy = img_result.copy()
        img_contours, contours_data = self.get_contours(img_result, canny_threshold=[500, 500], angle_filter=4,
                                                        show_canny=False, show_contours=True)
        cv2.imshow('img_result', img_result)
        delta = 5
        for i, contour_data in enumerate(contours_data):
            # bbox
            x, y, w, h = contour_data[3]
            item_highlighted_img = img_copy[y+delta: y+h-delta, x+delta: x+w-delta]
            cv2.imshow(f'highlighted_{i}', item_highlighted_img)
            highlighted_text = pytesseract.image_to_string(item_highlighted_img)
            print(f'highlighted_{i}: ', highlighted_text.strip())

        cv2.waitKey(0)


if __name__ == '__main__':
    htd = HighlightedTextDetection()
    # htd.detect_object_hsv_value()
    htd.run()
