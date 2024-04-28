import cv2
import mediapipe as mp


class HandsDetector:
    """
    手部检测器
    """
    def __init__(
            self,
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_hands_instance = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def detect_hands_landmarks(self, img, show_hand_connections=True, show_landmarks=True, show_landmarks_id=True):
        """
        检测手部地标
        :return:
        """
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        # 执行检测
        hands_result = self.mp_hands_instance.process(image=img_rgb)
        # 获取多手手部地标
        multi_hand_landmarks = getattr(hands_result, 'multi_hand_landmarks')
        # 构造地标字典，【嵌套层级 => 手id：地标id：坐标】
        hand_landmarks_dict = dict()
        if multi_hand_landmarks:
            for hand_id, item_hand_landmarks in enumerate(multi_hand_landmarks):
                # 单手
                if show_hand_connections:
                    self.mp_drawing_utils.draw_landmarks(
                        image=img,
                        landmark_list=item_hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        connection_drawing_spec=self.mp_drawing_utils.DrawingSpec(color=(0, 255, 0)),
                        landmark_drawing_spec=self.mp_drawing_utils.DrawingSpec(
                            color=(0, 0, 255), thickness=cv2.FILLED, circle_radius=2),
                        is_drawing_landmarks=True
                    )
                for landmark_id, landmark in enumerate(item_hand_landmarks.landmark):
                    # 关键点地标
                    h, w, c = img_rgb.shape
                    landmark_x_px = int(landmark.x * w)
                    landmark_y_px = int(landmark.y * h)
                    hand_landmarks_dict.setdefault(hand_id, dict())[landmark_id] = [landmark_x_px, landmark_y_px]
                    if show_landmarks:
                        cv2.circle(img=img, center=(landmark_x_px, landmark_y_px), radius=4, color=(255, 0, 255),
                                   thickness=cv2.FILLED)
                    if show_landmarks_id:
                        cv2.putText(img=img, text=f'{hand_id}_{landmark_id}', org=(landmark_x_px, landmark_y_px),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)

        return hand_landmarks_dict


if __name__ == '__main__':
    hd = HandsDetector()
    cap = cv2.VideoCapture(1)
    while True:
        success, _img = cap.read()
        _hand_landmarks_dict = hd.detect_hands_landmarks(
            img=_img,
            show_hand_connections=True,
            show_landmarks=False,
            show_landmarks_id=True
        )
        print(_hand_landmarks_dict)
        cv2.imshow('img', _img)
        cv2.waitKey(1)
