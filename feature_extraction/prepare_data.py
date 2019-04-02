import os
import re
import shutil
from glob import glob
import ntpath


def rename_files():
    # ---------parameter---------
    base_path = "data/rider_images/"
    data_folder = "all_v4_raw/"

    # ---------program----------
    version = int(re.search(r'\d+', data_folder).group())

    path_full = base_path + "all_v" + str(version) + "/"

    if os.path.exists(path_full):
        shutil.rmtree(path_full, ignore_errors=True)

    shutil.copytree(base_path+data_folder, base_path + "all_v" + str(version))

    img_name_list = glob(path_full + "*.jpg")
    img_name_list.sort()

    i = 54
    ii = 0
    code_old = "anything except 0"

    for index, path_and_name in enumerate(img_name_list):
        img_name_only = ntpath.basename(path_and_name)
        code = img_name_only.split('_')

        if code[1] != code_old:
            i += 1
            ii = 0

        os.rename(path_and_name, path_full + str(i) + "_" + str(ii) + ".jpg")

        ii += 1
        code_old = code[1]


def split_data_into_train_test_set():
    # ---------parameter---------
    base_path = "data/rider_images/"
    data_folder = "all_v3/"

    # ---------program----------
    version = int(re.search(r'\d+', data_folder).group())

    if os.path.exists(base_path + "train_v" + str(version)):
        shutil.rmtree(base_path + "train_v" + str(version), ignore_errors=True)

    shutil.copytree(base_path+data_folder, base_path + "train_v" + str(version))

    if os.path.exists(base_path + "test_v" + str(version)):
        shutil.rmtree(base_path + "test_v" + str(version), ignore_errors=True)

    os.makedirs(base_path + "test_v" + str(version))

    img_name_list = glob(base_path + "train_v" + str(version) + "/*.jpg")
    img_name_list.sort()

    eval_done = False
    test_done = False

    code_old = "-1"

    for index, path_and_name in enumerate(img_name_list):
        img_name_only = ntpath.basename(path_and_name)
        code = img_name_only.split('_')

        if code[0] != code_old:
            eval_done = False
            test_done = False

        if not eval_done:
            shutil.copy(base_path + "train_v" + str(version) + "/" + img_name_only, base_path + "test_v" +
                        str(version) + "/" + code[0] + "_0.jpg")
            eval_done = True
        elif not test_done:
            shutil.move(base_path + "train_v" + str(version) + "/" + img_name_only, base_path + "test_v" +
                        str(version) + "/" + code[0] + "_1.jpg")
            test_done = True

        code_old = code[0]


if __name__ == "__main__":
    # split_data_into_train_test_set()
    rename_files()
