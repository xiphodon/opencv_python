import cv2
import face_recognition

from utils import img_stack_util
#  安装环境（依次按照）：c编辑器  cmake   dlib   face_recognition


class FaceRecognition:
    """
    人脸识别
    """
    def __init__(self):
        musk_1 = face_recognition.load_image_file(r'./resources/faces/musk_1.png')
        musk_2 = face_recognition.load_image_file(r'./resources/faces/musk_2.png')
        gates_1 = face_recognition.load_image_file(r'./resources/faces/gates_1.png')
        gates_2 = face_recognition.load_image_file(r'./resources/faces/gates_2.png')

        self.musk_1 = cv2.cvtColor(musk_1, cv2.COLOR_BGR2RGB)
        self.musk_2 = cv2.cvtColor(musk_2, cv2.COLOR_BGR2RGB)
        self.gates_1 = cv2.cvtColor(gates_1, cv2.COLOR_BGR2RGB)
        self.gates_2 = cv2.cvtColor(gates_2, cv2.COLOR_BGR2RGB)

    def draw_face_box(self, img):
        location_list = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, location_list)
        # box顺序为(top, right, bottom, left)
        location = location_list[0]
        cv2.rectangle(img=img,
                      pt1=(location[3], location[0]),
                      pt2=(location[1], location[2]),
                      color=(255, 0, 255),
                      thickness=2)
        return img, encodings, location

    def draw_face_landmark(self, img, face_landmark):
        """
        绘制面部
        :param img:
        :param face_landmark:
        :return:
        """
        for k, v in face_landmark.items():
            if k == 'chin':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (0, 0, 255), thickness=1)
            if k == 'left_eyebrow' or k == 'right_eyebrow':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (0, 255, 0), thickness=1)
            if k == 'nose_bridge':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (255, 0, 0), thickness=1)
            if k == 'nose_tip':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (0, 0, 255), thickness=1)
            if k == 'left_eye' or k == 'right_eye':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (255, 0, 255), thickness=1)
            if k == 'top_lip':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (255, 255, 0), thickness=1)
            if k == 'bottom_lip':
                for x_y in v:
                    cv2.circle(img, x_y, 1, (0, 255, 255), thickness=1)
        return img

    def recognition(self):
        musk_1_img, musk_1_encodings, musk_1_location = self.draw_face_box(self.musk_1)
        musk_2_img, musk_2_encodings, musk_2_location = self.draw_face_box(self.musk_2)
        gates_1_img, gates_1_encodings, gates_1_location = self.draw_face_box(self.gates_1)
        gates_2_img, gates_2_encodings, gates_2_location = self.draw_face_box(self.gates_2)

        face_encodings = [musk_1_encodings[0], musk_2_encodings[0], gates_1_encodings[0], gates_2_encodings[0]]
        results = face_recognition.compare_faces(face_encodings, musk_1_encodings[0], tolerance=0.65)
        face_distance = face_recognition.face_distance(face_encodings, musk_1_encodings[0])
        print(results)
        print(face_distance)

        musk_1_face_landmarks = face_recognition.face_landmarks(musk_1_img, [musk_1_location])
        print(musk_1_face_landmarks)

        for item_face_landmark in musk_1_face_landmarks:
            musk_1_img = self.draw_face_landmark(musk_1_img, item_face_landmark)

        img = img_stack_util.stack_img(([musk_1_img, musk_2_img], [gates_1_img, gates_2_img]))
        cv2.imshow('face', img)
        cv2.waitKey(0)

    def video_recognition(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

        while True:
            timer_1 = cv2.getTickCount()
            success, img = cap.read()
            img = cv2.resize(src=img, dsize=(500, 300))
            if success:
                location_list = face_recognition.face_locations(img)
                if len(location_list) > 0:
                    face_landmarks = face_recognition.face_landmarks(img, location_list, model='large')
                    for face_landmark in face_landmarks:
                        img = self.draw_face_landmark(img, face_landmark)

                    # box顺序为(top, right, bottom, left)
                    location = location_list[0]
                    cv2.rectangle(img=img,
                                  pt1=(location[3], location[0]),
                                  pt2=(location[1], location[2]),
                                  color=(255, 0, 255),
                                  thickness=2)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer_1)
                cv2.putText(img, f'{fps: 0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.imshow('face', img)
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    # fr.recognition()
    fr.video_recognition()
