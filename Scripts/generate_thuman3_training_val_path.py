# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = '/home/lixing/Documents/Thuman/'  # root path

# the file's path and format
# RAW_DATA_FOLDER = 'Kitti2012/training/%s/'
RAW_DATA_FOLDER = 'human3_fb_train/'
RGB_FOLDER = 'RENDER/%s/'
DEPTH_FOLDER = 'DEPTH/%s/'
UV_FOLDER = 'uv/%s/'
FILE_NAME = '%04d_0_00'
#UV_FILE_NAME = '%04d_0_00_dp.0001'
UV_FILE_NAME = 'dp.%04d'
DT_DATA_FOLDER = 'thuman3_fb_gt/'


# file type
RAW_COLOR_TYPE = '.jpg'
RAW_DEPTH_TYPE = '.png'

# the output's path,
TRAIN_LIST_PATH = './Datasets/thuman3_training_list_all_1.csv'
VAL_TRAINLIST_PATH = './Datasets/thuman3_val_list_all_1.csv'

# IMG_NUM = 194  # the dataset's total image
IMG_NUM = 100    # the dataset's total image
TIMES = 20      # the sample of val

TEST_FLAG = True


def gen_color_path(folder: str, file: str,  num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_FOLDER % folder + file + '/' + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_path(folder: str, file: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_FOLDER %folder + file + '/' + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path


def gen_uv_path(folder: str, file: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + UV_FOLDER % folder + file + '/' + UV_FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def gen_color_gt_path(folder: str, file: str, num: int) -> str:
    path = ROOT_PATH + DT_DATA_FOLDER + RGB_FOLDER % folder + file + '/' + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_gt_path(folder: str, file: str, num: int) -> str:
    path = ROOT_PATH + DT_DATA_FOLDER +  DEPTH_FOLDER %folder + file + '/' + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)
    if os.path.exists(VAL_TRAINLIST_PATH):
        os.remove(VAL_TRAINLIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_val_train_list = open(VAL_TRAINLIST_PATH, 'a')

    data_str = "color_img,depth_img,uv_img,color_gt,depth_gt"
    output_data(fd_train_list, data_str)
    output_data(fd_val_train_list, data_str)

    return fd_train_list, fd_val_train_list


def output_data(output_file: object, data: str) -> None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def produce_list(folder, file_list, fd_train_list, fd_val_train_list):
    total = 0
    off_set = 1
    for i in range(len(file_list)):
        num_uv = 0
        for num in (list(range(0,30))+list(range(330,360))):
            color_path = gen_color_path(folder, file_list[i], num)
            depth_path = gen_depth_path(folder, file_list[i], num)
            num_uv = num_uv + 1
            print(i, num, num_uv)

            uv_path = gen_uv_path(folder, file_list[i], num_uv)

            if num < 180:
                num_label = abs(num + 180)
            if num > 180:
                num_label = abs(num - 180)
            color_lable_path = gen_color_gt_path(folder, file_list[i], num_label)            
            depth_lable_path = gen_depth_gt_path(folder, file_list[i], num_label)
            

            color_path_is_exists = os.path.exists(color_path)

            depth_path_is_exists = os.path.exists(depth_path)

            uv_path_is_exists = os.path.exists(uv_path)
            depth_lable_path_is_exists = os.path.exists(depth_lable_path)
            color_lable_path_is_exists = os.path.exists(color_lable_path)

            if (not color_path_is_exists) and (not color_lable_path_is_exists)\
                    (not depth_path_is_exists) and (not depth_lable_path_is_exists)\
                    and (not uv_path_is_exists):
                break
            

            if (off_set + i) % TIMES == 0 and num==0:
                data_str = color_path + ',' + depth_path + ',' + uv_path + ',' + color_lable_path + ',' + depth_lable_path 
                output_data(fd_val_train_list, data_str)
            if (off_set + i) % TIMES != 0:
                data_str = color_path + ',' + depth_path + ',' + uv_path + ',' + color_lable_path + ',' + depth_lable_path 
                output_data(fd_train_list, data_str)

            total = total + 1

    return total

def gen_list(fd_train_list, fd_val_train_list):

    filePath = '/home/lixing/Documents/Thuman/human3_fb_train/RENDER'
    folder_list = os.listdir(filePath)
    for i in range(len(folder_list)):
        file_list = os.listdir(filePath + '/' + folder_list[i])
        total = produce_list(folder_list[i], file_list, fd_train_list, fd_val_train_list)
    return total


def main() -> None:
    fd_train_list, fd_val_train_list = open_file()
    total = gen_list(fd_train_list, fd_val_train_list)
    print(total)


if __name__ == '__main__':
    main()
