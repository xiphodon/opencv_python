import pathlib

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
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

        self.batch_size_value = 50
        self.step_per_epoch_value = 2000
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
        pass

if __name__ == '__main__':
    tsc = TrafficSignClassification()
    tsc.load_data()
    tsc.split_data()
    tsc.plot_samples()
