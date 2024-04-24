import random
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import os


class FormsOCR:
    """
    收据表格文字识别
    """

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

        self.forms_dir_path = Path(r'./resources/forms_ocr')
        self.form_path = self.forms_dir_path / 'form.png'
        self.scale = 0.4

        # 先运行self.mark_key_areas(), 将空白form中关键位置坐标、类型、名称标记出来
        self.roi = [[(38, 391), (275, 432), 'text', 'name'],
                    [(295, 391), (532, 431), 'text', 'phone'],
                    [(38, 459), (60, 483), 'box', 'sign'],
                    [(295, 459), (318, 483), 'box', 'allergic'],
                    [(39, 566), (276, 607), 'text', 'email'],
                    [(294, 565), (532, 607), 'text', 'id'],
                    [(39, 634), (274, 677), 'text', 'city'],
                    [(298, 634), (534, 675), 'text', 'country']]

        self.click_count = 0
        self.area_info_list = list()

        self.point_1 = self.point_2 = None
        self.point_list = list()

        self.alpha = 0.7
        self.good_match_pk_threshold = 80

        self.checkbox_px_threshold = 500

    def mouse_points(self, event, x, y, flags, params):
        """
        鼠标点击回调
        :param event:
        :param x:
        :param y:
        :param flags:
        :param params:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
                self.click_count = 1
                self.point_1 = (int(x), int(y))
            elif self.click_count == 1:
                self.click_count = 0
                self.point_2 = (int(x), int(y))
                area_type = input('enter type:')
                area_name = input('enter name:')
                self.area_info_list.append([self.point_1, self.point_2, area_type, area_name])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.point_list.append([x, y, color])

    def mark_key_areas(self):
        """
        标记关键区域
        :return:
        """
        img = cv2.imread(self.form_path.as_posix())
        img = cv2.resize(src=img, dsize=(0, 0), fx=self.scale, fy=self.scale)
        while True:
            for x, y, color in self.point_list:
                cv2.circle(img=img, center=(x, y), radius=3, color=color, thickness=cv2.FILLED)
            cv2.imshow('form img', img)
            cv2.setMouseCallback('form img', self.mouse_points)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(self.area_info_list)
                break

    def extract_text(self):
        """
        抽取文字
        :return:
        """
        form_img = cv2.imread(self.form_path.as_posix())
        form_img = cv2.resize(src=form_img, dsize=(0, 0), fx=self.scale, fy=self.scale)
        h, w, c = form_img.shape

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(form_img, None)

        for item_form_path in self.forms_dir_path.iterdir():
            if item_form_path.name == 'form.png':
                continue
            item_img = cv2.imread(item_form_path.as_posix())
            item_img = cv2.resize(src=item_img, dsize=(0, 0), fx=self.scale, fy=self.scale)

            kp2, des2 = orb.detectAndCompute(item_img, None)
            bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bfm.knnMatch(des1, des2, k=2)

            good_match_kps = list()
            for d_k in matches:
                if len(d_k) < 2:
                    # 最近邻不足k个点
                    continue
                # 每个匹配对的前k个匹配
                m1, m2 = d_k[:2]
                # m1 < a * m2
                # 距离越近，点对匹配度越高；如果匹配度最高的点对距离远小于第二匹配度点对，则第一个点对（匹配对）的可靠度较高
                if m1.distance < self.alpha * m2.distance:
                    good_match_kps.append(m1)
            print(len(good_match_kps))

            img_features = cv2.drawMatches(img1=form_img, keypoints1=kp1, img2=item_img,
                                           keypoints2=kp2, matches1to2=good_match_kps, outImg=None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            if len(good_match_kps) > self.good_match_pk_threshold:
                # 匹配对数量超过阈值，表示检测到目标
                form_img_good_kps = list()
                item_img_good_kps = list()
                for good_match in good_match_kps:
                    form_img_good_kps.append(kp1[good_match.queryIdx].pt)
                    item_img_good_kps.append(kp2[good_match.trainIdx].pt)
                # 找到两张图所对应的匹配点
                form_img_good_kps_arr = np.float32(form_img_good_kps).reshape(-1, 1, 2)
                item_img_good_kps_arr = np.float32(item_img_good_kps).reshape(-1, 1, 2)
                # 单应矩阵
                matrix, mask = cv2.findHomography(srcPoints=item_img_good_kps_arr,
                                                  dstPoints=form_img_good_kps_arr,
                                                  method=cv2.RANSAC,
                                                  ransacReprojThreshold=5)
                if matrix is not None:
                    img_show = cv2.warpPerspective(src=item_img, M=matrix, dsize=(w, h))
                    # cv2.imshow(item_form_path.name, img_show)

                    img_show_copy = img_show.copy()
                    img_mask = np.zeros_like(img_show)

                    item_data_list = list()

                    for i, r in enumerate(self.roi):
                        pt1, pt2, area_type, area_name = r
                        cv2.rectangle(img=img_mask, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=cv2.FILLED)
                        img_show = cv2.addWeighted(src1=img_show, alpha=1, src2=img_mask, beta=0.1, gamma=0)
                        img_crop = img_show_copy[pt1[1]: pt2[1], pt1[0]: pt2[0]]

                        if area_type == 'text':
                            # 文字区域
                            text = pytesseract.image_to_string(img_crop).strip()
                            print(f'{area_name}: {text}')
                            item_data_list.append(text)
                        if area_type == 'box':
                            # checkbox 区域
                            img_crop_gray = cv2.cvtColor(src=img_crop, code=cv2.COLOR_BGR2GRAY)
                            _, img_crop_binary = cv2.threshold(src=img_crop_gray, thresh=170, maxval=255,
                                                               type=cv2.THRESH_BINARY_INV)
                            print(f'threshold: {_} {img_crop_binary.shape}')
                            draw_px_count = cv2.countNonZero(img_crop_binary)
                            if draw_px_count > self.checkbox_px_threshold:
                                item_data_list.append(1)
                            else:
                                item_data_list.append(0)

                        cv2.putText(img=img_show, text=str(item_data_list[i]), org=pt1,
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

                    cv2.imshow(item_form_path.name, img_show)
                    cv2.waitKey(0)


if __name__ == '__main__':
    form_ocr = FormsOCR()
    # form_ocr.mark_key_areas()
    form_ocr.extract_text()
