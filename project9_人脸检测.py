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
            for k, v in item_face_landmark.items():
                if k == 'chin':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (0, 0, 255), thickness=1)
                if k == 'left_eyebrow' or k == 'right_eyebrow':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (0, 255, 0), thickness=1)
                if k == 'nose_bridge':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (255, 0, 0), thickness=1)
                if k == 'nose_tip':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (0, 0, 255), thickness=1)
                if k == 'left_eye' or k == 'right_eye':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (255, 0, 255), thickness=1)
                if k == 'top_lip':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (255, 255, 0), thickness=1)
                if k == 'bottom_lip':
                    for x_y in v:
                        cv2.circle(musk_1_img, x_y, 1, (0, 255, 255), thickness=1)

        img = img_stack_util.stack_img(([musk_1_img, musk_2_img], [gates_1_img, gates_2_img]))
        cv2.imshow('face', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.recognition()
