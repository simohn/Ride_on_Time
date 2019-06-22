# Code written by Simon Schauppenlehner
# Last change: 22.06.2019

from Evaluator import Evaluator
from Rider import Rider
from Image import Image

from glob import glob
import ntpath
from Model import Model
from Hyperparameters import hyper_para


def train():
    print("Train model")
    model = Model.raw(epochs=hyper_para.epochs, batch_size=hyper_para.batch_size,
                      pooling=hyper_para.pooling, trainable_layer=hyper_para.trainable_layer)
    model.compile()
    model.train()
    model.save()


def evaluate():
    # print("Set model of Image class")
    model = Model.trained(epochs=hyper_para.epochs, batch_size=hyper_para.batch_size,
                          pooling=hyper_para.pooling, trainable_layer=hyper_para.trainable_layer)
    Image.set_model(model)

    eval = Evaluator()

    # print("Create riders")
    number_riders = 48

    for i in range(0,number_riders):
        # print(str(i+1) + "/" + str(number_riders) + " created")
        rider = Rider(i)

        if i < 10:
            images_path = glob("data\\riders_train_images\\rider_" + str(i) + "\\" + "*.jpg")
        else:
            images_path = glob("data\\riders_train_images\\rider_" + str(i) + "\\" + "*.jpg")

        for index, img_path in enumerate(images_path):
            img = Image(img_path, rider.get_id())
            rider.add_image(img)

        rider.calc_features_average()
        rider.calc_features_variance()

        eval.add_rider(rider)

    # print("Evaluate test images")
    test_images_path_and_name = glob("data\\riders_test_images_many\\" + "*.jpg")
    test_images_path_and_name.sort()

    test_images = []

    for index, test_image_path_and_name in enumerate(test_images_path_and_name):
        # print(str(index+1) + "/" + str(len(test_images_path_and_name)) + " evaluated")
        img_name = ntpath.basename(test_image_path_and_name)
        code = img_name.split('_')
        rider_id = code[0]

        image = Image(test_image_path_and_name, int(rider_id))
        test_images.append(image)

    # print("Export statistic")
    # eval.export_stat(test_images, "data\\stats\\Stat_" + Image.get_model_details() + ".xlsx")

    print("--------------------")
    print("Model: " + model.get_model_details())
    print("Validation accuracy: " + str(eval.get_val_acc(test_images)))


if __name__ == "__main__":
    hyper_para.epochs = 15
    hyper_para.batch_size = 10
    hyper_para.pooling = "max"
    hyper_para.trainable_layer = 15
    hyper_para.margin = 15

    models_path = glob("data\\models_export\\" + "*.h5")

    for index, model_path in enumerate(models_path):
        splits = model_path.split("_")
        hyper_para.epochs = int(splits[4])
        hyper_para.batch_size = int(splits[6])
        hyper_para.pooling = "max"
        hyper_para.trainable_layer = int(splits[8])
        hyper_para.margin = int(splits[12].split(".")[0])

        evaluate()

    # train()
    # evaluate()

    #for n in range(5, 26, 5):
    #   hyper_para.trainable_layer = n
    #   train()
    #  evaluate()


