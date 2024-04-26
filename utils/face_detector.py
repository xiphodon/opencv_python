import cv2
import mediapipe as mp


class FaceDetector:
    """
    人脸检测器
    """
    def __init__(self):
        # 人脸检测
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_detection_instance = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=0
        )
        # 人脸网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def detect_face_landmarks(
            self,
            img,
            show_face_bbox=True,
            show_face_score=True,
            show_landmarks=True,
            show_landmarks_id=True
    ):
        """
        检测人脸地标
        :param img: 原图
        :param show_face_bbox: 显示人脸边框
        :param show_face_score: 显示人脸分数
        :param show_landmarks: 显示关键点地标
        :param show_landmarks_id: 显示关键点地标id
        :return:
        """
        h, w, c = img.shape

        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        # 执行检测
        face_result = self.mp_face_detection_instance.process(image=img_rgb)
        # 多人脸检测结果
        multi_face_detections = getattr(face_result, 'detections')
        # 构造人脸检测结果
        face_detections_list = list()
        if multi_face_detections:
            for face_id, item_face_detection in enumerate(multi_face_detections):
                # 单个人脸
                label_id = item_face_detection.label_id[0]
                score = item_face_detection.score[0]
                location_data = item_face_detection.location_data

                # 位置数据：相对位置边框，相对位置关键点
                relative_bbox = location_data.relative_bounding_box
                relative_keypoints = location_data.relative_keypoints

                # 边框坐标点
                bbox_x_min = int(relative_bbox.xmin * w)
                bbox_y_min = int(relative_bbox.ymin * h)
                bbox_w = int(relative_bbox.width * w)
                bbox_h = int(relative_bbox.height * h)

                if show_face_bbox:
                    cv2.rectangle(img=img, pt1=(bbox_x_min, bbox_y_min),
                                  pt2=(bbox_x_min + bbox_w, bbox_y_min + bbox_h),
                                  color=(255, 0, 0), thickness=2)
                if show_face_score:
                    cv2.putText(img=img, text=f'id_{face_id}: {round(score * 100, 2)}%', org=(bbox_x_min, bbox_y_min - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0), thickness=2)

                keypoint_list = list()
                for item_keypoint_id, item_keypoint in enumerate(relative_keypoints):
                    kp_x = int(item_keypoint.x * w)
                    kp_y = int(item_keypoint.y * h)
                    keypoint_list.append({
                        'kp_id': item_keypoint_id,
                        'kp_x_y': [kp_x, kp_y]
                    })

                    if show_landmarks:
                        cv2.circle(img=img, center=(kp_x, kp_y), radius=3, color=(255, 0, 255), thickness=cv2.FILLED)
                    if show_landmarks_id:
                        cv2.putText(img=img, text=f'{face_id}_{item_keypoint_id}', org=(kp_x, kp_y - 5),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

                face_detections_list.append({
                    'face_id': face_id,
                    'score': score,
                    'bbox_x_min': bbox_x_min,
                    'bbox_y_min': bbox_y_min,
                    'bbox_w': bbox_w,
                    'bbox_h': bbox_h,
                    'keypoint_list': keypoint_list
                })
        return face_detections_list

    def detect_face_mesh(self, img, show_face_landmarks=True):
        """
        人脸网格检测
        :param img:
        :param show_face_landmarks:
        :return:
        """
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        face_result = self.mp_face_mesh_instance.process(image=img)
        multi_face_landmarks = getattr(face_result, 'multi_face_landmarks')
        # print(multi_face_landmarks)
        face_mesh_list = list()
        if multi_face_landmarks:
            for face_id, item_face_landmarks in enumerate(multi_face_landmarks):
                item_face_landmarks_dict = {
                    'face_id': face_id,
                    'landmark_list': list()
                }
                if show_face_landmarks:
                    self.mp_drawing_utils.draw_landmarks(
                        image=img,
                        landmark_list=item_face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.mp_drawing_utils.DrawingSpec(
                            color=(0, 0, 255), thickness=1, circle_radius=1
                        ),
                        connection_drawing_spec=self.mp_drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1
                        )
                    )
                for landmark_id, landmark in enumerate(item_face_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    item_face_landmarks_dict['landmark_list'].append({
                        'id': landmark_id,
                        'x_y': [x, y]
                    })
                face_mesh_list.append(item_face_landmarks_dict)
        return face_mesh_list


if __name__ == '__main__':
    fd = FaceDetector()
    cap = cv2.VideoCapture(1)
    while True:
        success, _img = cap.read()
        # _face_detections_list = fd.detect_face_landmarks(
        #     img=_img,
        #     show_face_bbox=True,
        #     show_landmarks=True,
        #     show_landmarks_id=True,
        #     show_face_score=True
        # )
        # print(_face_detections_list)
        _face_mesh_list = fd.detect_face_mesh(
            img=_img,
            show_face_landmarks=True
        )
        print(_face_mesh_list)
        cv2.imshow('img', _img)
        cv2.waitKey(1)
