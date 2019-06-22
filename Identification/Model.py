# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.applications.resnet50 import ResNet50
from tensorflow.contrib.keras.api.keras.optimizers import Adam

from Triplet_loss import batch_hard_triplet_loss_keras
from DataGenerator import data_generator

from tensorflow.contrib.keras.api.keras.preprocessing import image
from tensorflow.contrib.keras.api.keras.applications.resnet50 import preprocess_input
import numpy as np

from Hyperparameters import hyper_para


class Model:

    # public methods

    def __init__(self, trained_model=False, epochs=10, batch_size=10, pooling="avg", trainable_layer=15):
        self.epochs = epochs
        self.batch_size = batch_size
        self.pooling = pooling
        self.trainable_layer = trainable_layer

        self.path_and_name = "data\\models_export\\resnet50_model_" + self.get_model_details() + ".h5"

        if trained_model:
            self.model = load_model(self.path_and_name,
                               custom_objects={"batch_hard_triplet_loss_keras": batch_hard_triplet_loss_keras})
        else:
            self.model = ResNet50(weights='imagenet', input_shape=(640, 640, 3), include_top=False,
                                  pooling=self.pooling)

    @classmethod
    def raw(cls, epochs=10, batch_size=10, pooling="avg", trainable_layer=15):
        return cls(trained_model=False, epochs=epochs, batch_size=batch_size, pooling=pooling,
                   trainable_layer=trainable_layer)

    @classmethod
    def trained(cls, epochs=10, batch_size=10, pooling="avg", trainable_layer=15):
        return cls(trained_model=True, epochs=epochs, batch_size=batch_size, pooling=pooling,
                   trainable_layer=trainable_layer)

    def compile(self):
        for layer in self.model.layers[:len(self.model.layers) - self.trainable_layer]:
            layer.trainable = False

        adam = Adam(lr=1e-3, decay=1e-6)
        self.model.compile(optimizer=adam, loss=batch_hard_triplet_loss_keras, metrics=['accuracy'])

    def set_parameters(self, epochs, batch_size, trainable_layer):
        self.epochs = epochs
        self.batch_size = batch_size
        self.trainable_layer = trainable_layer

    def train(self, num_riders=48):
        training_generator = data_generator(np.arange(num_riders), batch_size=self.batch_size)

        self.model.fit_generator(generator=training_generator, epochs=self.epochs, use_multiprocessing=True, workers=6)

    def save(self):
        self.model.save(self.path_and_name)

    def calc_features(self, image_path):
        img = image.load_img(image_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # type(features)) ... class 'numpy.ndarray'
        features = self.model.predict(x)[0]

        return features

    def get_model_details(self):
        return "epochs_" + str(self.epochs) + "_batchSize_" + str(self.batch_size) \
               + "_trainableLayer_" + str(self.trainable_layer) + "_pooling_" \
               + self.pooling + "_margin_" + str(hyper_para.margin)
