import cv2

from utils import img_stack_util


def get_contours(img, img_contour):
    """
    获取轮廓
    :param img: 预处理图（灰度高斯模糊图）
    :param img_contour: 原图副本，绘制轮廓
    :return:
    """
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # 计算轮廓区域面积
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            # 面积大于500
            # 绘制轮廓
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            # 计算曲线周长(轮廓曲线，是否封闭)
            perimeter = cv2.arcLength(curve=cnt, closed=True)
            print(perimeter)
            # 近似多边曲线（轮廓曲线，逼近精度[值越小，两线最大距离越小，折线越多，多边形边数越多]，是否封闭），返回定点向量
            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02*perimeter, closed=True)
            print(approx)
            # 多边形角数
            obj_cor = len(approx)
            print(obj_cor)
            # 计算灰度图像边距
            x, y, w, h = cv2.boundingRect(approx)

            # 根据角数判断形状
            if obj_cor == 3:
                object_type = "Tri"
            elif obj_cor == 4:
                asp_ratio = w/float(h)
                if 1.03 > asp_ratio > 0.98:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif obj_cor > 4:
                object_type = "Circles"
            else:
                object_type = "None"

            # 绘制矩形边框
            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 边框左下角绘制文字
            cv2.putText(img_contour, object_type,
                        (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)


def start():
    """
    入口
    :return:
    """
    img = cv2.imread(r'./resources/shapes.png')
    # 副本
    img_contour = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊，降噪
    img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=1)
    img_canny = cv2.Canny(img_blur, 40, 40)
    get_contours(img_canny, img_contour)

    img_stack = img_stack_util.stack_img(
        img_arr=([img, img_gray, img_blur],
                 [img_canny, img_contour]),
        scale=0.6,
        labels=(['origin', 'gray', 'blur'],
                ['canny', 'contour'])
    )

    cv2.imshow("Stack", img_stack)

    cv2.waitKey(0)


if __name__ == '__main__':
    start()
