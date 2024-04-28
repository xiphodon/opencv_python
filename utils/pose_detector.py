import cv2
import mediapipe as mp


class PoseDetector:
    """
    姿态检测
    """
    def __init__(
            self,
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_pose_instance = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def detect_pose_landmarks(self, img, show_pose_connections=True, show_landmarks=True, show_landmarks_id=True):
        """
        检测姿态地标
        :param img:
        :param show_pose_connections:
        :param show_landmarks:
        :param show_landmarks_id:
        :return:
        """
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        # 执行检测
        pose_result = self.mp_pose_instance.process(image=img_rgb)
        # 获取姿态地标
        pose_landmarks = getattr(pose_result, 'pose_landmarks')
        # 构造地标字典
        pose_landmarks_dict = dict()
        if pose_landmarks:
            if show_pose_connections:
                self.mp_drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=pose_landmarks,
                    connections=self.mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=self.mp_drawing_utils.DrawingSpec(color=(0, 255, 0))
                )
            for landmark_id, landmark in enumerate(pose_landmarks.landmark):
                # 关键点地标
                h, w, c = img_rgb.shape
                landmark_x_px = int(landmark.x * w)
                landmark_y_px = int(landmark.y * h)
                pose_landmarks_dict[landmark_id] = [landmark_x_px, landmark_y_px]
                if show_landmarks:
                    cv2.circle(img=img, center=(landmark_x_px, landmark_y_px), radius=4, color=(255, 0, 255),
                               thickness=cv2.FILLED)
                if show_landmarks_id:
                    cv2.putText(img=img, text=f'{landmark_id}', org=(landmark_x_px, landmark_y_px),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)
        return pose_landmarks_dict


if __name__ == '__main__':
    pd = PoseDetector()
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        success, _img = cap.read()
        if not success:
            break
        pose_landmarks_dict = pd.detect_pose_landmarks(
            img=_img,
            show_pose_connections=True,
            show_landmarks=True,
            show_landmarks_id=True
        )
        print(pose_landmarks_dict)
        cv2.imshow('img', _img)
        cv2.waitKey(1)