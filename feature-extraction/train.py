from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

import tensorflow as tf

import numpy as np
import ntpath

from scipy.spatial import distance
from glob import glob

from random import shuffle

from triplet_loss import _pairwise_distances, _get_anchor_negative_triplet_mask, _get_anchor_positive_triplet_mask


def start():
    model = ResNet50(weights='imagenet', input_shape=(640, 640, 3), include_top=False, pooling='avg')

    # model.summary()

    for layer in model.layers[:len(model.layers)-15]:
        layer.trainable = False

    adam = Adam(lr=1e-3, decay=1e-6)
    model.compile(optimizer=adam, loss=batch_hard_triplet_loss_keras, metrics=['accuracy'])

    x_train = []
    y_train = []

    path_train = "Rider_Images/Train_V1"
    img_name_list = glob(path_train + "/*.jpg")

    shuffle(img_name_list)

    for id, path_and_name in enumerate(img_name_list):
        name_only = ntpath.basename(path_and_name)
        code = name_only.split('_')

        img = image.load_img(path_and_name)

        # Pre-process image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        x = x[0, :, :, :]

        x_train.append(x)
        y_train.append(code[0])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    model.fit(x_train, y_train, batch_size=10, epochs=3)
    model.save('my_model.h5')

    # model = load_model('my_model.h5')

    path_test = "Rider_Images/Test_V1"
    img_name_list = glob(path_test + "/*.jpg")
    img_name_list.sort()

    r_features = []
    t_features = []

    for id, path_and_name in enumerate(img_name_list):
        name_only = ntpath.basename(path_and_name)
        code = name_only.split('_')

        img = image.load_img(path_and_name)

        # Pre-process image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = model.predict(x)
        if code[0] == 'r':
            r_features.append(features)
        else:
            t_features.append(features)

    for a, r_feature in enumerate(r_features):
        for i, t_feature in enumerate(t_features):
            dst = distance.euclidean(t_feature, r_feature)
            print(str(a) + "-" + str(i) + " = " + str(dst))


def batch_hard_triplet_loss_keras(y_true, y_pred):
    margin = 0.5

    pairwise_dist = _pairwise_distances(y_pred, squared=False)
    # shape of pairwise_dist: (batch_size, batch_size)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    labels = tf.squeeze(y_true, axis=-1)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    # shape of mask_anchor_positive: (batch_size, batch_size)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    # shape of anchor_positive_dist: (batch_size, batch_size)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    # shape of mask_anchor_negative: (batch_size, batch_size)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    # triplet_loss = tf.reduce_mean(triplet_loss)

    # see keras examples for losses: in case of training with batches -> return (n, 1)
    # However, the API would also accept a scalar (so it doesn't matter if the mean value is calculated here)
    # batch ... n samples
    # y_pred = (n, embeddings)
    # triplet_loss = (n, 1)

    return triplet_loss


if __name__ == "__main__":
    start()
