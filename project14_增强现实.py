import cv2
import numpy as np
from utils.img_stack_util import stack_img


class AugmentedReality:
    def __init__(self):
        self.camera_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.mark_img = cv2.imread(r'./resources/vip_card.jpg')
        self.video_cap = cv2.VideoCapture(r'./resources/video.mp4')

        self.camera_cap.set(cv2.CAP_PROP_BRIGHTNESS, 180)
        self.mark_img = cv2.GaussianBlur(src=self.mark_img, ksize=(7, 7), sigmaX=0)

        self.detection = False
        self.video_frame_counter = 0

        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher()

        self.good_match_pk_threshold = 30
        self.alpha = 0.75

    def run(self):
        success, video_img = self.video_cap.read()
        mark_img_h, mark_img_w, mark_img_c = self.mark_img.shape
        # video_img = cv2.resize(src=video_img, dsize=(mark_img_h, mark_img_w))

        mark_img_kps, mark_img_des = self.orb.detectAndCompute(self.mark_img, None)
        mark_img = cv2.drawKeypoints(self.mark_img, mark_img_kps, None)

        # cv2.imshow('mark_img', mark_img)

        while True:
            success, camera_img = self.camera_cap.read()
            camera_img_result = camera_img.copy()
            camera_img_mark = None
            img_warp = None
            img_result = None

            camera_img_kps, camera_img_des = self.orb.detectAndCompute(camera_img, None)
            if camera_img_des is None:
                continue

            if self.detection is False or self.video_frame_counter == self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # 当前未检测到目标  或  当前帧为视频尾帧
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_frame_counter = 0

            if self.detection is True:
                # 当前检测到目标对象
                success, video_img = self.video_cap.read()
                video_img = cv2.resize(src=video_img, dsize=(mark_img_w, mark_img_h))

            # 返回k个匹配对
            matchs = self.bf.knnMatch(queryDescriptors=mark_img_des, trainDescriptors=camera_img_des, k=2)
            good_match_kps = list()
            for d_k in matchs:
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
            img_features = cv2.drawMatches(img1=mark_img, keypoints1=mark_img_kps, img2=camera_img,
                                           keypoints2=camera_img_kps, matches1to2=good_match_kps, outImg=None, flags=2)

            if len(good_match_kps) > self.good_match_pk_threshold:
                # 匹配对数量超过阈值，表示检测到目标
                self.detection = True
                mark_img_good_kps = list()
                camera_img_good_kps = list()
                for good_match in good_match_kps:
                    mark_img_good_kps.append(mark_img_kps[good_match.queryIdx].pt)
                    camera_img_good_kps.append(camera_img_kps[good_match.trainIdx].pt)
                # 找到两张图所对应的匹配点
                mark_img_good_kps_arr = np.float32(mark_img_good_kps).reshape(-1, 1, 2)
                camera_img_good_kps_arr = np.float32(camera_img_good_kps).reshape(-1, 1, 2)
                # 单应矩阵
                matrix, mask = cv2.findHomography(srcPoints=mark_img_good_kps_arr,
                                                  dstPoints=camera_img_good_kps_arr,
                                                  method=cv2.RANSAC,
                                                  ransacReprojThreshold=5)
                if matrix is not None:
                    # 将目标图片尺寸透视到摄像头图片中指定位置
                    mark_img_pts = np.float32(
                        [[0, 0], [0, mark_img_h], [mark_img_w, mark_img_h], [mark_img_w, 0]]).reshape(-1, 1, 2)
                    # print(mark_img_pts)
                    # print(matrix)
                    dst = cv2.perspectiveTransform(mark_img_pts, matrix)
                    camera_img_mark = cv2.polylines(img=camera_img, pts=[np.int32(dst)], isClosed=True,
                                                    color=(255, 0, 255),
                                                    thickness=3)
                    cv2.fillPoly(img=camera_img_result, pts=[np.int32(dst)], color=(0, 0, 0))

                    # 将视频图片透视到指定位置
                    img_warp = cv2.warpPerspective(src=video_img, M=matrix,
                                                   dsize=(camera_img.shape[1], camera_img.shape[0]))
                    # 组合
                    img_result = cv2.bitwise_or(src1=camera_img_result, src2=img_warp)

            camera_img_mark = camera_img_mark if camera_img_mark is not None else camera_img
            img_warp = img_warp if img_warp is not None else camera_img
            img_result = img_result if img_result is not None else camera_img

            all_img = stack_img(img_arr=([camera_img_mark, camera_img_result],
                                         [img_warp, img_result]),
                                scale=0.7,
                                lables=[['camera_img_mark', 'camera_img_result'],
                                        ['img_warp', 'img_result']])

            cv2.imshow('img_features', img_features)
            cv2.imshow('all_img', all_img)

            self.video_frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                break

        self.video_cap.release()
        self.camera_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ar = AugmentedReality()
    ar.run()
