import cv2
import time

from utils.face_detector import FaceDetector


class FaceTracking:
    """
    人脸检测跟踪
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.face_detector = FaceDetector()

    def run(self):
        last_time = time.time()
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break

            # 人脸检测
            # face_detections_list = self.face_detector.detect_face_landmarks(
            #     img=img,
            #     show_face_bbox=True,
            #     show_face_score=True,
            #     show_landmarks_id=True,
            #     show_landmarks=True
            # )
            # print(face_detections_list)

            # 人脸网格检测
            face_mesh_list = self.face_detector.detect_face_mesh(
                img=img,
                show_face_landmarks=True
            )
            print(face_mesh_list)

            current_time = time.time()
            fps = round(1.0 / (current_time - last_time), 2)
            last_time = current_time

            cv2.putText(img=img, text=f'fps: {fps}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=2)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        self.cap.release()


if __name__ == '__main__':
    ft = FaceTracking()
    ft.run()
