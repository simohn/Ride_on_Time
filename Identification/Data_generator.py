# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

import numpy as np
import tensorflow.contrib.keras as keras
from glob import glob
from tensorflow.contrib.keras.api.keras.preprocessing import image
from tensorflow.contrib.keras.api.keras.applications.resnet50 import preprocess_input
import ntpath


class Data_generator(keras.utils.Sequence):

    def __init__(self, list_ids, batch_size=10, shuffle=True, path_train_images="data/riders_train_images/"):
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.shuffle = shuffle
        self.path_train_images = path_train_images + "rider_"
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size) + 1)

    def __getitem__(self, index):
        list_ids_temp = self.list_ids[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.list_ids)

    def __data_generation(self, list_ids_temp):
        X = []
        y = []

        for i, ID in enumerate(list_ids_temp):
            img_path_and_name_list = glob(self.path_train_images + str(ID) + "/" + "*.jpg")

            for ii, img_path_and_name in enumerate(img_path_and_name_list):
                img_name = ntpath.basename(img_path_and_name)
                code = img_name.split('_')

                img = image.load_img(img_path_and_name)

                # Pre-process image
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                x = x[0, :, :, :]

                # Store image and label
                X.append(x)
                y.append(code[0])

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y
