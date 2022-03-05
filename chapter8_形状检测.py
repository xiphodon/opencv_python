import cv2
import numpy as np

from utils import img_stack_util


def get_contours(img, img_contour):
    """
    获取轮廓
    :param img: 预处理图（灰度高斯模糊图）
    :param img_contour: 原图副本，绘制轮廓
    :return:
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"

            cv2.rectangle(img_contour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img_contour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)


def start():
    """
    入口
    :return:
    """
    img = cv2.imread(r'./resources/shapes.png')
    # 副本
    img_contour = img.copy()

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 高斯模糊，降噪
    img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=1)
    img_canny = cv2.Canny(img_blur, 40, 40)
    get_contours(img_canny, img_contour)

    img_stack = img_stack_util.stack_img(
        ([img, img_gray, img_blur],
         [img_canny, img_contour]), scale=0.5)

    cv2.imshow("Stack", img_stack)

    cv2.waitKey(0)


if __name__ == '__main__':
    start()
