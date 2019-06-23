# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

import os
import numpy as np
import cv2 as cv
import math

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from tensorflow.contrib.keras.api.keras.preprocessing import image
from tensorflow.contrib.keras.api.keras.applications.resnet50 import preprocess_input
from tensorflow.contrib.keras.api.keras.models import load_model

from glob import glob
import pickle

from Triplet_loss import batch_hard_triplet_loss_keras

sprite_img_single_dim = 128

epochs = 40
batch_size = 10
pooling = "avg"
trainable_layer = 20

path_test = "data\\rider_images\\train_v4\\"


def calc_embeddings():
    # ---------parameter---------
    name_saved_model = "data\\resnet50_model_epochs_" + str(epochs) + "_batchSize_" + \
                                   str(batch_size) + "_trainableLayer_" + str(trainable_layer) + \
                                   "_pooling_" + pooling + ".h5"

    # ---------program----------
    model = load_model(name_saved_model,
                       custom_objects={"batch_hard_triplet_loss_keras": batch_hard_triplet_loss_keras})

    img_name_list = glob(path_test + "*.jpg")

    img_name_list.sort(key=lambda img: int(img.split('\\')[3].split('_')[0]))

    features_all = np.zeros((len(img_name_list), 2048))
    labels = np.zeros(len(img_name_list)
                     )
    print("Analyzing...")

    for index, path_and_name in enumerate(img_name_list):
        img = image.load_img(path_and_name)
        label = path_and_name.split('\\')[3].split('_')[0]

        # Pre-process image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = model.predict(x)
        features_all[index] = features
        labels[index] = label

        print("..." + str(index+1) + "/" + str(len(img_name_list)))

    with open('data\\temp\\features_all.pickle', 'wb') as handle:
        pickle.dump(features_all, handle)

    with open('data\\temp\\labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle)


def setup_tensorboard():
    with open('data\\temp\\features_all.pickle', 'rb') as handle:
        features_all = pickle.load(handle)

    with open('data\\temp\\labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)

    log_dir = 'tensorboard_log'
    name_to_visualise_variable = 'riders_embedding'
    path_for_metadata = os.path.join(log_dir, 'metadata.tsv')

    embedding_var = tf.Variable(features_all, name=name_to_visualise_variable)
    summary_writer = tf.summary.FileWriter(log_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = 'metadata.tsv'
    embedding.sprite.image_path = 'sprite.png'
    embedding.sprite.single_image_dim.extend([sprite_img_single_dim, sprite_img_single_dim])

    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

    with open(path_for_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, label))

    # Run tensorboard
    # tensorboard --logdir tensorboard_log


def create_sprite_image():

    with open('data\\labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)

    img_name_list = glob(path_test + "*.jpg")
    img_name_list.sort(key=lambda img: int(img.split('\\')[3].split('_')[0]))

    sprite_dim = int(math.sqrt(len(img_name_list))+1)

    sprite_image = np.ones((sprite_img_single_dim * sprite_dim, sprite_img_single_dim * sprite_dim, 3))

    index = 0
    for i in range(sprite_dim):
        for j in range(sprite_dim):
            if index < len(img_name_list):
                img = cv.imread(img_name_list[index], cv.IMREAD_COLOR)

                img_resized = cv.resize(img, (sprite_img_single_dim, sprite_img_single_dim))

                sprite_image[
                i * sprite_img_single_dim: (i + 1) * sprite_img_single_dim,
                j * sprite_img_single_dim: (j + 1) * sprite_img_single_dim,
                :
                ] = img_resized

                index += 1

    cv.imwrite('sprite.png', sprite_image)


if __name__ == "__main__":
    # calc_embeddings()
    # create_sprite_image()
    setup_tensorboard()
