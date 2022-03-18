import cv2
import numpy as np


class ObjectDetection:
    """
    物体检测
    """

    def __init__(self):
        # 置信度
        self.threshold = 0.5
        # 趋向于1，则为不抑制；趋向于0，为最大化抑制
        self.nms_threshold = 0.3
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.cap.set(10, 100)

        self.class_names_path = r'./resources/Object_Detection_Files/coco.names'
        self.config_path = r'./resources/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weights_path = r'./resources/Object_Detection_Files/frozen_inference_graph.pb'

        self.class_names = list()
        with open(self.class_names_path, 'r') as fp:
            self.class_names = [line.strip() for line in fp.readlines()]

        self.net = cv2.dnn_DetectionModel(self.weights_path, self.config_path)
        self.net.setInputSize(size=(320, 320))
        self.net.setInputScale(scale=1.0 / 127.5)
        self.net.setInputMean(mean=(127.5, 127.5, 127.5))
        self.net.setInputSwapRB(swapRB=True)

    def detect_object_nms(self):
        """
        检测物体（nms 非最大抑制）
        :return:
        """
        while True:
            success, img = self.cap.read()
            class_ids, confidences, bboxes = self.net.detect(frame=img, confThreshold=self.threshold)

            # 非最大抑制
            indices = cv2.dnn.NMSBoxes(bboxes=bboxes, scores=confidences, score_threshold=self.threshold,
                                       nms_threshold=self.nms_threshold)
            # bboxes = np.array(bboxes).tolist()
            # confidences = np.array(confidences).tolist()
            #
            # if (not isinstance(bboxes, list)) or (not isinstance(confidences, list)):
            #     continue

            for i in indices:
                bbox = bboxes[i]
                x, y, w, h = bbox
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(img=img, text=f'{self.class_names[class_ids[i] - 1]} {round(confidences[i]*100, 2)}%',
                            org=(x + 10, y + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 255, 0), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def detect_object(self):
        """
        检测物体
        :return:
        """
        while True:
            success, img = self.cap.read()
            class_ids, confidences, bboxes = self.net.detect(frame=img, confThreshold=self.threshold)
            # print(class_ids, confidences, boxes)

            if len(class_ids) != 0:
                for class_id, confidence, bbox in zip(class_ids, confidences, bboxes):
                    print(class_id, confidence, bbox)
                    x, y, w, h = bbox
                    cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                    # cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, f'{self.class_names[class_id - 1].upper()} {round(confidence * 100, 2)}%',
                                (bbox[0] + 10, bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    od = ObjectDetection()
    # od.detect_object()
    od.detect_object_nms()
