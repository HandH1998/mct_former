import argparse
from pathlib import Path
import math
import os
import shutil

parser = argparse.ArgumentParser('split image id script')
parser.add_argument('--ori_img_id_path', type=str, default='datasets/voc12/train_aug_id.txt')
parser.add_argument('--ori_img_data_path', type=str, default='datasets/voc12/VOCdevkit/VOC2012/JPEGImages')
parser.add_argument('--dest_dir', type=str, default='datasets/new_voc12')
parser.add_argument('--dataset_name', type=str, default='voc12')
parser.add_argument('--split_num', type=int, default=5)

args = parser.parse_args()

img_id_list = open(args.ori_img_id_path).readlines()
img_name_list = [img_gt_name.strip() for img_gt_name in img_id_list]

def cp_img(ori_dir, file_list, dest_dir, suf=''):
    for file_name in file_list:
        file_path = ori_dir + os.path.sep + file_name + suf
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir)

each_split_img_num = math.ceil(len(img_name_list) / args.split_num)

for i in range(args.split_num):
    start_idx = i * each_split_img_num
    end_idx = start_idx + each_split_img_num
    new_file_list = img_name_list[start_idx : min(len(img_name_list), end_idx)]
    dest_dir = args.dest_dir + os.path.sep + args.dataset_name + '_' + str(i)
    dest_img_dir = dest_dir + os.path.sep + 'JPEGImages'
    Path(dest_img_dir).mkdir(parents=True, exist_ok=True)
    cp_img(args.ori_img_data_path, new_file_list, dest_img_dir, '.jpg')

    if args.dataset_name == 'voc12':
        with open(dest_dir + os.path.sep + 'train_aug_id.txt', 'w') as f:
            f.write('\n'.join(new_file_list))
    elif args.dataset_name == 'coco':
        with open(dest_dir + os.path.sep + 'train_id.txt', 'w') as f:
            f.write('\n'.join(new_file_list))

print('done!')

