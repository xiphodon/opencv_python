import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(r'./resources/haarcascade_frontalface_default.xml')


def detect_img_face():
    """
    检测图片人脸
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=4)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def detect_video_face():
    """
    检测视频人脸
    :return:
    """
    # 视频路径为获取资源，id序号则为摄像头id，0为默认第一个摄像头id
    # cap = cv2.VideoCapture(r'./resources/video_test.mp4')
    cap = cv2.VideoCapture(0)
    # cap propId 0-18
    cap.set(3, 800)  # 3为宽
    cap.set(4, 600)  # 4为高
    cap.set(10, 5)  # 10为亮度

    while True:
        success, img = cap.read()
        if success:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('video', img)
        else:
            break
        print('1')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 有按键则返回按键ASCII码，无按键则返回-1
            # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # detect_img_face()
    detect_video_face()

