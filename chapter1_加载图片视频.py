import cv2


def read_img():
    """
    读取图片
    :return:
    """
    img = cv2.imread(r'./resources/lena.jpeg')
    cv2.imshow('lena img', img)
    cv2.waitKey(0)


def read_video():
    """
    读取视频/摄像头
    :return:
    """
    # 视频路径为获取资源，id序号则为摄像头id，0为默认第一个摄像头id
    # cap = cv2.VideoCapture(r'./resources/video.mp4')
    cap = cv2.VideoCapture(0)
    # cap propId 0-18
    cap.set(3, 800)    # 3为宽
    cap.set(4, 600)  # 4为高
    cap.set(10, 5)  # 10为亮度

    while True:
        success, img = cap.read()
        if success:
            cv2.imshow('video', img)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 有按键则返回按键ASCII码，无按键则返回-1
            # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_img()
    read_video()
