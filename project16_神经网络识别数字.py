import random
from pathlib import Path

from keras.losses import CategoricalCrossentropy
import numpy as np
import cv2
from keras.optimizer_v2.adam import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle


class DigitsRecognitionNeuralNetwork:
    """
    数字识别（神经网络）
    """

    def __init__(self):
        self.digits_img_path = Path(r'./resources/digits_img')
        self.test_ratio = 0.1
        self.validation_ratio = 0.2
        self.img_shape = (32, 32, 3)
        self.batch_size = 64
        self.epochs = 20
        self.steps_per_epoch = 100
        self.class_no = -1

        self.data_gen = ImageDataGenerator(
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            shear_range=0.1,
            rotation_range=20
        )

    def load_img(self):
        """
        加载数字图片
        :return:
        """
        digit_img_list = list()
        digit_class_no_list = list()
        digits_dir_list = list(self.digits_img_path.iterdir())
        self.class_no = len(digits_dir_list)
        print('load digit images ...')
        for i_digit_dir in digits_dir_list:
            for i_img_path in i_digit_dir.iterdir():
                cur_img = cv2.imread(i_img_path.as_posix())
                cur_img = cv2.resize(src=cur_img, dsize=(self.img_shape[0], self.img_shape[1]))
                digit_img_list.append(cur_img)
                digit_class_no_list.append(int(i_digit_dir.name))
            print(f'{i_digit_dir.name}', end=' ')
        print('')
        print(f'total images: {len(digit_img_list)}')
        print(f'total class no: {len(digit_class_no_list)}')
        return np.array(digit_img_list), np.array(digit_class_no_list)

    def split_data(self, data_x, data_y):
        """
        拆分数据
        :param data_x:
        :param data_y:
        :return:
        """
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=self.test_ratio, random_state=0)
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,
                                                                        test_size=self.validation_ratio, random_state=0)
        print(x_train.shape, x_validation.shape, x_test.shape)
        print(y_train.shape, y_validation.shape, y_test.shape)
        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def plot_train_class_bar_chart(self, y_train):
        """
        绘制各类别分布柱状图
        :param y_train:
        :return:
        """
        class_size_list = list()
        for i_class_name in range(self.class_no):
            class_size_list.append(len(np.where(y_train == i_class_name)[0]))
        print(class_size_list)

        plt.figure(figsize=(10, 5))
        plt.bar(range(self.class_no), class_size_list)
        plt.title('number of images for each class')
        plt.xlabel('class id')
        plt.ylabel('number of images')
        plt.show()

    def pre_processing(self, img):
        """
        预处理
        (灰度、直方图均值化、归一化)
        :param img:
        :return:
        """
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(src=img)
        img = img / 255
        return img

    @property
    def nn_model(self):
        """
        神经网络模型
        :return:
        """
        # 过滤器数量
        filter_count = 60
        # 过滤器1尺寸
        filter_size_1 = (5, 5)
        # 过滤器2尺寸
        filter_size_2 = (3, 3)
        # 池化层尺寸
        pool_size = (2, 2)
        # 节点数量
        nodes_count = 500

        model = Sequential(name='digits_nn')
        model.add(layer=Conv2D(filters=filter_count, kernel_size=filter_size_1, input_shape=(*self.img_shape[:2], 1),
                               activation='relu'))
        model.add(layer=Conv2D(filters=filter_count, kernel_size=filter_size_1, activation='relu'))
        model.add(layer=MaxPooling2D(pool_size=pool_size))

        model.add(layer=Conv2D(filters=filter_count // 2, kernel_size=filter_size_2, activation='relu'))
        model.add(layer=Conv2D(filters=filter_count // 2, kernel_size=filter_size_2, activation='relu'))
        model.add(layer=MaxPooling2D(pool_size=pool_size))

        model.add(layer=Dropout(0.5))
        model.add(layer=Flatten())
        model.add(layer=Dense(units=nodes_count, activation='relu'))
        model.add(layer=Dropout(0.5))
        model.add(layer=Dense(units=self.class_no, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def run(self):
        """
        运行
        :return:
        """
        # 加载图片和对应类别
        digit_img_arr, digit_class_no_arr = self.load_img()
        # 绘制每个类别数目柱状图
        self.plot_train_class_bar_chart(digit_class_no_arr)
        # 数据预处理
        digit_img_arr = np.array(list(map(self.pre_processing, digit_img_arr)))
        digit_img_arr = digit_img_arr.reshape((*digit_img_arr.shape, 1))
        digit_class_no_arr = to_categorical(digit_class_no_arr, self.class_no)  # one-hot
        # 数据集分割
        x_train, x_validation, x_test, y_train, y_validation, y_test = self.split_data(digit_img_arr,
                                                                                       digit_class_no_arr)
        # 数据增强
        self.data_gen.fit(x_train)

        model = self.nn_model
        print(model.summary())

        model_result = model.fit(x=self.data_gen.flow(x_train, y_train,
                                                      batch_size=self.batch_size),
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(x_validation, y_validation),
                                 shuffle=True)

        plt.figure(1)
        plt.plot(model_result.history['loss'])
        plt.plot(model_result.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.figure(2)
        plt.plot(model_result.history['accuracy'])
        plt.plot(model_result.history['val_accuracy'])
        plt.legend(['training', 'validation'])
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.show()

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test Score = ', score[0])
        print('Test Accuracy =', score[1])

        with open(r'./resources/digits_nn_model_trained.p', 'wb') as fp:
            pickle.dump(model, fp)


if __name__ == '__main__':
    drnn = DigitsRecognitionNeuralNetwork()
    drnn.run()
