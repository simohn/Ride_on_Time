import sys

from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

import numpy as np
from scipy.spatial.distance import cdist

import ntpath
from glob import glob
import xlsxwriter

from feature_extraction.triplet_loss import batch_hard_triplet_loss_keras


def train():
    # ---------parameter---------
    load_trained_model = False
    name_loaded_model = "resnet50_model.h5"
    save_model = True
    name_saved_model = "resnet50_model_v3_epochs_1.h5"

    num_layer_trainable = 15

    path_train = "data/rider_images/train_v3/"

    # ---------program----------
    if load_trained_model:
        model = load_model("data/" + name_loaded_model)
    else:
        model = ResNet50(weights='imagenet', input_shape=(640, 640, 3), include_top=False, pooling='avg')
    # model.summary()

    for layer in model.layers[:len(model.layers)-num_layer_trainable]:
        layer.trainable = False

    adam = Adam(lr=1e-3, decay=1e-6)
    model.compile(optimizer=adam, loss=batch_hard_triplet_loss_keras, metrics=['accuracy'])

    x_train = []
    y_train = []

    img_name_list = glob(path_train + "*.jpg")
    # img_name_list.sort()

    for index, path_and_name in enumerate(img_name_list):
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

    model.fit(x_train, y_train, batch_size=50, epochs=1, shuffle=True, callbacks=[tb_callback])

    if save_model:
        model.save("data/" + name_saved_model)


def test():
    # ---------parameter---------
    name_saved_model = "resnet50_model_v3_epochs_1.h5"
    path_test = "data/rider_images/test_v3/"

    excel_file_name = "Dist-matrix_v3_epochs_2.xlsx"

    # ---------program----------
    model = load_model("data/" + name_saved_model,
                       custom_objects={"batch_hard_triplet_loss_keras": batch_hard_triplet_loss_keras})

    img_name_list = glob(path_test + "*.jpg")
    img_name_list.sort()

    features_all = np.zeros((len(img_name_list), 2048))

    print("Analyzing...")

    for index, path_and_name in enumerate(img_name_list):
        img = image.load_img(path_and_name)

        # Pre-process image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = model.predict(x)
        features_all[index] = features

        print("..." + str(index+1) + "/" + str(len(img_name_list)))

    ranking_init = [(0, sys.maxsize), (1, sys.maxsize), (2, sys.maxsize)]
    ranking = ranking_init.copy()

    dist_matrix = cdist(features_all, features_all, metric='euclidean')

    workbook = xlsxwriter.Workbook("data/export/" + excel_file_name)
    worksheet = workbook.add_worksheet()

    cf_1 = workbook.add_format()
    cf_1.set_pattern(1)  # This is optional when using a solid fill.
    cf_1.set_bg_color("#008000")

    cf_2 = workbook.add_format()
    cf_2.set_pattern(1)  # This is optional when using a solid fill.
    cf_2.set_bg_color('#808080')

    cf_3 = workbook.add_format()
    cf_3.set_pattern(1)  # This is optional when using a solid fill.
    cf_3.set_bg_color('#A0A0A0')

    for index_1, path_and_name in enumerate(img_name_list):
        name_only = ntpath.basename(path_and_name)
        worksheet.write_string(0, index_1+1, name_only)
        worksheet.write_string(index_1+1, 0, name_only)

        worksheet.write_row(index_1+1, 1, dist_matrix[index_1])

        for index_2 in range(len(img_name_list)):
            if dist_matrix[index_1][index_2] > 0.0:
                if dist_matrix[index_1][index_2] < ranking[0][1]:
                    ranking[2] = ranking[1]
                    ranking[1] = ranking[0]
                    ranking[0] = (index_2, dist_matrix[index_1][index_2])
                elif dist_matrix[index_1][index_2] < ranking[1][1]:
                    ranking[2] = ranking[1]
                    ranking[1] = (index_2, dist_matrix[index_1][index_2])
                elif dist_matrix[index_1][index_2] < ranking[2][1]:
                    ranking[2] = (index_2, dist_matrix[index_1][index_2])

        worksheet.write(index_1+1, ranking[0][0]+1, ranking[0][1], cf_1)
        worksheet.write(index_1+1, ranking[1][0]+1, ranking[1][1], cf_2)
        worksheet.write(index_1+1, ranking[2][0]+1, ranking[2][1], cf_3)

        ranking = ranking_init.copy()

    workbook.close()


if __name__ == "__main__":
    train()
    # test()
