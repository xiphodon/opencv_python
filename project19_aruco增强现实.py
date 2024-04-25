from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoModule:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.aruco_path = Path(r'./resources/aruco')
        self.aruco_img_shape = (300, 300)
        self.aruco_img_path_dict = {
            23: Path(r'./resources/card.jpeg'),
            40: Path(r'./resources/face.png'),
            124: Path(r'./resources/lena.jpeg')
        }
        self.aruco_img_dict = self.build_aruco_img_dict()

    def build_aruco_img_dict(self):
        """
        构建 id:img 对照字典
        :return:
        """
        aruco_img_dict = dict()
        for k_id, path in self.aruco_img_path_dict.items():
            item_img = cv2.imread(filename=path.as_posix())
            item_img = cv2.resize(src=item_img, dsize=self.aruco_img_shape)
            aruco_img_dict[k_id] = item_img
        return aruco_img_dict

    def find_aruco_markers(self, img, marker_size=6, total_markers=250, draw_marker=True):
        """
        find aruco markers
        :param img:
        :param marker_size:
        :param total_markers:
        :param draw_marker:
        :return: corners, ids
        """
        img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # aruco.DICT_6X6_250
        aruco_dict_size = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
        aruco_dict = aruco.Dictionary_get(aruco_dict_size)
        aruco_param = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(image=img_gray, dictionary=aruco_dict, parameters=aruco_param)
        # print(corners, ids, rejected_img_points)
        if draw_marker:
            aruco.drawDetectedMarkers(image=img, corners=corners, ids=ids)
        return corners, ids

    def augment_aruco(self, marker_corner, marker_id, src_img, augment_img):
        """
        augment aruco
        :param marker_corner:
        :param marker_id:
        :param src_img:
        :param augment_img:
        :return:
        """
        left_top_point = marker_corner[0]
        right_top_point = marker_corner[1]
        right_bottom_point = marker_corner[2]
        left_bottom_point = marker_corner[3]

        h, w, c = augment_img.shape

        pts1 = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        pts2 = np.array([
            left_top_point,
            right_top_point,
            right_bottom_point,
            left_bottom_point
        ])
        matrix = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
        # matrix, mask = cv2.findHomography(srcPoints=pts1, dstPoints=pts2, method=cv2.RANSAC, ransacReprojThreshold=5)
        img_warp = cv2.warpPerspective(src=augment_img, M=matrix, dsize=(src_img.shape[1], src_img.shape[0]))
        src_img = cv2.fillConvexPoly(img=src_img, points=pts2.astype(dtype='int'), color=(0, 0, 0))
        src_img = src_img + img_warp
        return src_img

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                continue

            corners, ids = self.find_aruco_markers(img=img, draw_marker=False)

            if len(corners) != 0:
                for item_corner, item_id in zip(corners, ids):
                    item_corner = item_corner[0]
                    item_id = item_id[0]
                    if item_id in self.aruco_img_dict:
                        img = self.augment_aruco(marker_corner=item_corner, marker_id=item_id,
                                                 src_img=img, augment_img=self.aruco_img_dict[item_id])
                    cv2.putText(img=img, text=str(item_id), org=np.int32(item_corner)[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(255, 0, 0), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break


if __name__ == '__main__':
    am = ArucoModule()
    am.run()
