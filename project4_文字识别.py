import cv2
import pytesseract


class TextImgOCR:
    """
    文字识别
    """
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
        img = cv2.imread(r'./resources/text_img.png')
        # bgr转rgb
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_characters(self):
        """
        检测文字
        :return:
        """
        img_height, img_width, _ = self.img.shape
        # 直接输出检测文字
        # boxes = pytesseract.image_to_string(self.img)
        # print(boxes)
        # 输出文字框等信息
        boxes = pytesseract.image_to_boxes(self.img)

        for line in boxes.splitlines():
            # print(line)
            line_data_list = line.split(' ')
            print(line_data_list)
            # tesseract 返回识别坐标原点为img左下角
            x1, y1, x2, y2 = list(map(lambda i: int(i), line_data_list[1:5]))
            cv2.rectangle(self.img, (x1, img_height - y1), (x2, img_height - y2), (50, 50, 255), thickness=2)
            cv2.putText(self.img, line_data_list[0], (x1, img_height - y1 + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (50, 50, 255), 2)

        cv2.imshow('ocr', self.img)
        cv2.waitKey(0)

    def detect_words(self):
        """
        检测单词
        :return:
        """
        data = pytesseract.image_to_data(self.img)
        print(data)
        for i, line in enumerate(data.splitlines()):
            if i > 0:
                line_data_list = line.split('\t')
                print(line_data_list)
                if len(line_data_list) == 12 and line_data_list[11] != '':
                    x, y, w, h = list(map(lambda a: int(a), line_data_list[6: 10]))
                    cv2.rectangle(self.img, (x, y), (x+w, y+h), (50, 50, 255), thickness=2)
                    cv2.putText(self.img, line_data_list[11], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.imshow('ocr', self.img)
        cv2.waitKey(0)

    def detect_only_digits(self):
        """
        仅识别数字
        :return:
        """
        img_height, img_width, _ = self.img.shape
        config = r'--oem 3 --psm 6 outputbase digits'
        boxes = pytesseract.image_to_boxes(self.img, config=config)
        for line in boxes.splitlines():
            line_data_list = line.split(' ')
            print(line_data_list)
            # tesseract 返回识别坐标原点为img左下角
            x1, y1, x2, y2 = list(map(lambda i: int(i), line_data_list[1:5]))
            cv2.rectangle(self.img, (x1, img_height - y1), (x2, img_height - y2), (50, 50, 255), 2)
            cv2.putText(self.img, line_data_list[0], (x1, img_height - y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.imshow('ocr', self.img)
        cv2.waitKey(0)


if __name__ == '__main__':
    text_img_ocr = TextImgOCR()
    # text_img_ocr.detect_characters()
    # text_img_ocr.detect_words()
    text_img_ocr.detect_only_digits()
