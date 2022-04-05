import numpy as np
import cv2


#  https://pjreddie.com/darknet/yolo/

class YOLO:
    def __init__(self, model_tiny=True):
        self.cap = cv2.VideoCapture(0)
        self.w_h = 320
        self.cfg_threshold = 0.4
        self.nms_threshold = 0.2

        self.classes_filepath = r'./resources/yolo_v3/coco.names'
        self.classes_names = list()
        self.read_classes_names()

        if model_tiny:
            self.model_cfg_path = r'./resources/yolo_v3/yolov3-tiny.cfg'
            self.model_weights_path = r'./resources/yolo_v3/yolov3-tiny.weights'
        else:
            # 权重文件下载 https://pjreddie.com/media/files/yolov3.weights
            self.model_cfg_path = r'./resources/yolo_v3/yolov3.cfg'
            self.model_weights_path = r'./resources/yolo_v3/yolov3.weights'

        self.net = cv2.dnn.readNetFromDarknet(cfgFile=self.model_cfg_path, darknetModel=self.model_weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    def read_classes_names(self):
        """
        加载类别名称
        :return:
        """
        with open(self.classes_filepath, 'r') as fp:
            for line in fp.readlines():
                self.classes_names.append(line.rstrip())

    def find_objects(self, outputs, img):
        img_h, img_w, img_c = img.shape
        bboxes = list()
        class_ids = list()
        confidences = list()

        for output in outputs:
            for detect in output:
                # cx, cy, w, h, confidence, [各个class对应的概率]
                _cx, _cy, _w, _h, _confidence = detect[:5]
                scores = detect[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.cfg_threshold:
                    w, h = int(img_w * _w), int(img_h * _h)
                    x, y = int(img_w * _cx - w / 2), int(img_h * _cy - h / 2)
                    bboxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(confidence)
        # 非最大抑制
        indeices = cv2.dnn.NMSBoxes(bboxes, confidences, self.cfg_threshold, self.nms_threshold)

        for i in indeices:
            box = bboxes[i]
            x, y, w, h = box
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 255), thickness=2)
            cv2.putText(img=img, text=f'{self.classes_names[class_ids[i]]} {round(confidences[i] * 100, 2)}%',
                        org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 255),
                        thickness=2)

    def run(self):
        layers_names = self.net.getLayerNames()
        output_layers_names = [(layers_names[i - 1]) for i in self.net.getUnconnectedOutLayers()]
        while True:
            timer_1 = cv2.getTickCount()

            success, img = self.cap.read()
            # 水平反转180度（沿y轴）
            img = cv2.flip(img, 180)
            blob = cv2.dnn.blobFromImage(image=img, scalefactor=1 / 255, size=(self.w_h, self.w_h), mean=[0, 0, 0],
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(output_layers_names)
            self.find_objects(outputs, img)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer_1)
            cv2.putText(img, f'fps: {fps: 0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 有按键则返回按键ASCII码，无按键则返回-1
                # 取按键返回ASCII码二进制后8位，为按键'q'退出循环:
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    yolo_v3 = YOLO(model_tiny=True)
    yolo_v3.run()
