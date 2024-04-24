import pathlib

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator


class TrafficSignClassification:
    """
    交通标识分类器
    """

    def __init__(self):
        self.data_path = pathlib.Path(r'./resources/traffic_sign')
        self.labels_path = self.data_path / 'labels' / 'labels.csv'
        self.img_dir_path = self.data_path / 'images'

        self.batch_size_value = 64
        self.step_per_epoch_value = 200
        self.epochs_value = 10
        self.img_shape = (32, 32, 3)
        self.test_data_ratio = 0.2
        self.validation_ratio = 0.2

        self.image_list = list()
        self.class_list = list()
        self.class_name_df = None
        self.classes_size = 0

        self.x_train = None
        self.x_validation = None
        self.x_test = None
        self.y_train = None
        self.y_validation = None
        self.y_test = None

        self.model_trained_path = pathlib.Path('./resources/traffic_sign_nn_model_trained.h5')

    def load_data(self):
        """
        加载数据
        :return:
        """
        img_dir_list = list(self.img_dir_path.iterdir())
        self.classes_size = len(img_dir_list)
        print(f'total classes: {self.classes_size}')
        print('importing images data ...')
        for i, item_class_img_dir in enumerate(img_dir_list, start=1):
            class_name = item_class_img_dir.stem
            for item_img_path in item_class_img_dir.iterdir():
                cur_img = cv2.imread(item_img_path.as_posix())
                self.image_list.append(cur_img)
                self.class_list.append(int(class_name))
            print(f'{class_name}', end=' ' if i % 10 != 0 else '\n')
        print()
        print('importing labels csv ...')
        df = pd.read_csv(self.labels_path)
        print(f'class name shape: {df.shape}, {type(df)}')
        self.class_name_df = df
        print('import OK')
        print('###################################')

    def split_data(self):
        image_arr = np.array(self.image_list)
        class_arr = np.array(self.class_list)
        _x_train, x_test, _y_train, y_test = train_test_split(image_arr, class_arr, test_size=self.test_data_ratio)
        x_train, x_validation, y_train, y_validation = train_test_split(_x_train, _y_train,
                                                                        test_size=self.validation_ratio)
        print(f'train x/y shape: {x_train.shape} {y_train.shape}')
        print(f'validation x/y shape: {x_validation.shape} {y_validation.shape}')
        print(f'test x/y shape: {x_test.shape} {y_test.shape}')
        self.x_train, self.x_validation, self.x_test = x_train, x_validation, x_test
        self.y_train, self.y_validation, self.y_test = y_train, y_validation, y_test

    def plot_samples(self):
        """
        绘制样本图
        :return:
        """
        num_of_samples = list()
        # 每个类别显示五张图片，即self.classes_size行5列
        show_cols = 5
        fig, axs = plt.subplots(nrows=self.classes_size, ncols=show_cols, figsize=(5, self.classes_size))
        fig.tight_layout()
        for class_index, row in self.class_name_df.iterrows():
            for col in range(show_cols):
                x_selected = self.x_train[self.y_train == class_index]
                axs[class_index][col].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap='gray')
                axs[class_index][col].axis('off')
                if col == 2:
                    axs[class_index][col].set_title(f'{class_index}_{row["Name"]}')
                    num_of_samples.append(len(x_selected))
        print(num_of_samples)
        # 绘制每个类别样本数量柱状图
        plt.figure(figsize=(12, 4))
        plt.bar(x=range(0, self.classes_size), height=num_of_samples)
        plt.title("Distribution of the training dataset")
        plt.xlabel("Class number")
        plt.ylabel("Number of images")
        plt.show()

    @staticmethod
    def grayscale(img):
        """
        图像转为灰度图
        :param img:
        :return:
        """
        return cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    @staticmethod
    def equalize_hist(img):
        """
        直方图均衡化
        :param img:
        :return:
        """
        return cv2.equalizeHist(src=img)

    def preprocessing(self, img):
        """
        预处理
        :param img:
        :return:
        """
        # 转为灰度图
        img = self.grayscale(img)
        # 直方图均衡化
        img = self.equalize_hist(img)
        # 数据归一化
        img = img / 255
        return img

    def run(self):
        """
        运行方法
        :return:
        """
        # 数据预处理
        x_train = np.array(list(map(self.preprocessing, self.x_train)))
        x_validation = np.array(list(map(self.preprocessing, self.x_validation)))
        x_test = np.array(list(map(self.preprocessing, self.x_test)))
        cv2.imshow('preprocessing img', x_train[random.randint(0, len(x_train) - 1)])

        # 增加灰度图的通道数，设置为1
        x_train = x_train.reshape(*x_train.shape, 1)
        x_validation = x_validation.reshape(*x_validation.shape, 1)
        x_test = x_test.reshape(*x_test.shape, 1)

        # 构造数据生成器
        data_gen = ImageDataGenerator(width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.2,
                                      shear_range=0.1,
                                      rotation_range=10)
        data_gen.fit(x_train)

        # 绘制数据增强的样本
        batches = data_gen.flow(x_train, self.y_train, batch_size=self.batch_size_value)
        x_batch, y_batch = next(batches)
        fig, axs = plt.subplots(nrows=1, ncols=15, figsize=(20, 5))
        fig.tight_layout()
        for i in range(15):
            axs[i].imshow(x_batch[i].reshape(*self.img_shape[:2]), cmap='gray')
            axs[i].axis('off')
        plt.show()

        # one-hot
        y_train = to_categorical(self.y_train, self.classes_size)
        y_validation = to_categorical(self.y_validation, self.classes_size)
        y_test = to_categorical(self.y_test, self.classes_size)

        model = self.create_model()
        print(model.summary())

        result = model.fit(data_gen.flow(x_train, y_train, batch_size=self.batch_size_value),
                           steps_per_epoch=self.step_per_epoch_value, epochs=self.epochs_value,
                           validation_data=(x_validation, y_validation), shuffle=True)

        # 绘制指标
        plt.figure(1)
        plt.plot(result.history['loss'])
        plt.plot(result.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('loss')
        plt.xlabel('epoch')
        plt.figure(2)
        plt.plot(result.history['accuracy'])
        plt.plot(result.history['val_accuracy'])
        plt.legend(['training', 'validation'])
        plt.title('Acurracy')
        plt.xlabel('epoch')
        plt.show()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test Score:', score[0])
        print('Test Accuracy:', score[1])

        model.save(self.model_trained_path)

        cv2.waitKey(0)

    def create_model(self):
        """
        创建模型
        :return:
        """
        filters_size = 60
        filter_1_shape = (5, 5)
        filter_2_shape = (3, 3)
        pool_shape = (2, 2)
        nodes_size = 500

        model = Sequential(name='traffic_sign_model')
        model.add(layer=Conv2D(filters=filters_size, kernel_size=filter_1_shape, input_shape=(*self.img_shape[:2], 1),
                               activation='relu'))
        model.add(layer=Conv2D(filters=filters_size, kernel_size=filter_1_shape, activation='relu'))
        model.add(layer=MaxPooling2D(pool_size=pool_shape))

        model.add(layer=Conv2D(filters=filters_size // 2, kernel_size=filter_2_shape, activation='relu'))
        model.add(layer=Conv2D(filters=filters_size // 2, kernel_size=filter_2_shape, activation='relu'))
        model.add(layer=MaxPooling2D(pool_size=pool_shape))
        model.add(layer=Dropout(0.5))

        model.add(layer=Flatten())
        model.add(layer=Dense(units=nodes_size, activation='relu'))
        model.add(layer=Dropout(0.5))
        model.add(layer=Dense(units=self.classes_size, activation='softmax'))

        model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def camera_run(self):
        """
        摄像头识别
        :return:
        """
        model = load_model(self.model_trained_path)
        df = pd.read_csv(self.labels_path)

        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        threshold = 0.85

        while True:
            success, img_origin = cap.read()
            img = np.array(img_origin)
            img = cv2.resize(src=img, dsize=self.img_shape[:2])
            img = self.preprocessing(img)
            # cv2.imshow('processed image', img)
            img = img.reshape((1, *self.img_shape[:2], 1))

            predictions = model.predict(img)
            # print(predictions)
            class_id = np.argmax(a=predictions[0], axis=0)
            prediction = predictions[0][class_id]
            print(class_id, prediction)

            if prediction > threshold:
                print(f'class name: {df.loc[class_id, "Name"]}')
                cv2.putText(img=img_origin, text=f'{class_id} {df.loc[class_id, "Name"]}  {prediction: .2f}', org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow('img_origin', img_origin)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    tsc = TrafficSignClassification()
    # tsc.load_data()
    # tsc.split_data()
    # tsc.plot_samples()
    tsc.run()
    # tsc.camera_run()
