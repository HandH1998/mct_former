from fileinput import filename
import threading
import numpy as np
import json


dict2npy_dict = {}
train_list = []
val_list = []
classes = 80  # set num_classes
train_txt = open("datasets/coco/train_id.txt", "w")
val_txt = open("datasets/coco/val_id.txt", "w")
dataset_name = "COCO"
train_file_mid_name = "train2014"
val_file_mid_name = "val2014"

def revise_cag_id(id):
    if 13 <= id <= 25:
        id = id - 1
    elif 27 <= id <= 28:
        id = id - 2
    elif 31 <= id <= 44:
        id = id - 4
    elif 46 <= id <= 65:
        id = id - 5
    elif 67 <= id <= 67:
        id = id - 6
    elif 70 <= id <= 70:
        id = id - 8
    elif 72 <= id <= 82:
        id = id - 9
    elif 84 <= id <= 90:
        id = id - 10
    return id


# get data in the instances_train2014.json
with open("datasets/coco/annotations/instances_train2014.json") as ins:
    coco_ins_14json = json.load(ins)
    for i in range(len(coco_ins_14json.get("annotations"))):
        OneHotmatrix = np.zeros(classes).astype(np.float32)
        str_id = str(coco_ins_14json.get("annotations")[i].get("image_id")).rjust(
            12, "0"
        )
        tmp = [dataset_name, train_file_mid_name, str_id]
        filename = '_'.join(tmp)
        for k in range(len(coco_ins_14json.get("categories"))):
            if coco_ins_14json.get("annotations")[i].get(
                "category_id"
            ) == coco_ins_14json.get("categories")[k].get("id"):
                OneHotmatrix[k] = 1
                print(
                    coco_ins_14json.get("annotations")[i].get("image_id"),
                    coco_ins_14json.get("categories")[k].get("id"),
                )
                if filename in train_list:
                    dict2npy_dict[filename] = OneHotmatrix + dict2npy_dict[filename]
                else:
                    dict2npy_dict[filename] = OneHotmatrix
                    train_list.append(filename)
                dict2npy_dict[filename][dict2npy_dict[filename] > 1] = 1
                break
            # print(coco_ins_14json.get("images")[i].get("file_name"))

# get data in the instances_val2014.json
with open("datasets/coco/annotations/instances_val2014.json") as ins:
    coco_ins_14json = json.load(ins)
    for i in range(len(coco_ins_14json.get("annotations"))):
        OneHotmatrix = np.zeros(classes).astype(np.float32)
        str_id = str(coco_ins_14json.get("annotations")[i].get("image_id")).rjust(
            12, "0"
        )
        tmp = [dataset_name, val_file_mid_name, str_id]
        filename = '_'.join(tmp)
        for k in range(len(coco_ins_14json.get("categories"))):
            if coco_ins_14json.get("annotations")[i].get(
                "category_id"
            ) == coco_ins_14json.get("categories")[k].get("id"):
                OneHotmatrix[k] = 1
                print(
                    coco_ins_14json.get("annotations")[i].get("image_id"),
                    coco_ins_14json.get("categories")[k].get("id"),
                )
                if filename in val_list:
                    dict2npy_dict[filename] = OneHotmatrix + dict2npy_dict[filename]
                else:
                    dict2npy_dict[filename] = OneHotmatrix
                    val_list.append(filename)
                dict2npy_dict[filename][dict2npy_dict[filename] > 1] = 1
                break
            # print(coco_ins_14json.get("images")[i].get("file_name"))

np.save("datasets/coco/cls_labels.npy", dict2npy_dict)
str = "\n"

train_txt.write(str.join(train_list))
train_txt.close()

val_txt.write(str.join(val_list))
val_txt.close()
