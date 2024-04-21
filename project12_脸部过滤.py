import cv2
import numpy as np
import dlib


class FaceFilter:
    def __init__(self, is_camera=False):
        self.is_camera = is_camera
        self.cap = cv2.VideoCapture(1)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'./resources/shape_predictor_68_face_landmarks.dat')

        cv2.namedWindow('bgr')
        cv2.resizeWindow('bgr', 640, 240)
        cv2.createTrackbar('blue', 'bgr', 153, 255, self.empty)
        cv2.createTrackbar('green', 'bgr', 0, 255, self.empty)
        cv2.createTrackbar('red', 'bgr', 137, 255, self.empty)

    def empty(self, v):
        pass

    def create_box(self, img, points, scale=8, masked=False, cropped=True, show_mask=False, show_crop=False):
        if masked:
            mask = np.zeros_like(img)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)
            if show_mask:
                cv2.imshow('mask', img)
            return mask

        if cropped:
            bbox = cv2.boundingRect(points)
            x, y, w, h = bbox
            img_crop = img[y: y+h, x: x+w]
            img_crop = cv2.resize(img_crop, (0, 0), None, scale, scale)
            if show_crop:
                cv2.imshow('img_crop', img_crop)
            return img_crop
        return img

    def run(self):
        while True:
            if self.is_camera:
                success, img = self.cap.read()
            else:
                img = cv2.imread(r'./resources/face.png')

            img_copy = img.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.detector(img)
            if len(faces) > 0:
                for face in faces:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.imshow('face', img_copy)
                    landmarks = self.predictor(img_gray, face)
                    points = list()
                    for i in range(68):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        points.append([x, y])

                        cv2.circle(img_copy, (x, y), 2, (50, 50, 255), cv2.FILLED)
                        cv2.putText(img_copy, f'{i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    if len(points) > 0:
                        points = np.array(points)
                        # img_eyebrow_left = self.create_box(img, points[17: 22])
                        # img_eyebrow_right = self.create_box(img, points[22: 27])
                        # img_nose = self.create_box(img, points[27: 36])
                        # img_left_eye = self.create_box(img, points[36: 42])
                        # img_right_eye = self.create_box(img, points[42: 48])
                        # img_lips = self.create_box(img, points[48: 61])

                        # cv2.imshow('left eyebrow', img_eyebrow_left)
                        # cv2.imshow('right eyebrow', img_eyebrow_right)
                        # cv2.imshow('nose', img_nose)
                        # cv2.imshow('left eye', img_left_eye)
                        # cv2.imshow('right eye', img_right_eye)
                        # cv2.imshow('lips', img_lips)

                        # 嘴唇为例
                        mask_lips = self.create_box(img, points[48: 61], masked=True, cropped=False)
                        img_color_lips = np.zeros_like(mask_lips)
                        b = cv2.getTrackbarPos('blue', 'bgr')
                        g = cv2.getTrackbarPos('green', 'bgr')
                        r = cv2.getTrackbarPos('red', 'bgr')

                        # 指定颜色全图
                        img_color_lips[:] = b, g, r
                        # 与蒙板位与出指定部位指定颜色
                        img_color_lips = cv2.bitwise_and(mask_lips, img_color_lips)
                        img_color_lips = cv2.GaussianBlur(img_color_lips, (7, 7), 10)

                        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                        # img_color_lips = cv2.addWeighted(img_gray, 1, img_color_lips, 0.4, 0)
                        img_color_lips = cv2.addWeighted(img, 1, img_color_lips, 0.4, 0)
                        cv2.imshow('bgr', img_color_lips)
            else:
                cv2.imshow('bgr', img)

            cv2.imshow('img', img_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_filter = FaceFilter(is_camera=True)
    face_filter.run()
