# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = 'G:/lx/Datasets/thuman2.0_v2/'  # root path

# the file's path and format
# RAW_DATA_FOLDER = 'Kitti2012/training/%s/'
RAW_DATA_FOLDER = 'testing/'
RGB_FOLDER = 'color/%s/'
DEPTH_FOLDER = 'depth/%s/'
UV_FOLDER = 'uv/%s/'
FILE_NAME = '%04d_0_00'
UV_FILE_NAME = '%04d_0_00_dp.0001'

# file type
RAW_COLOR_TYPE = '.jpg'
RAW_DEPTH_TYPE = '.png'

# the output's path,
TEST_LIST_PATH = './Datasets/thuman_testing_list_4.csv'

IMG_NUM = 60    # the dataset's total image



def gen_color_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def gen_uv_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + UV_FOLDER % file_folder + UV_FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def open_file() -> object:
    if os.path.exists(TEST_LIST_PATH):
        os.remove(TEST_LIST_PATH)

    fd_test_list = open(TEST_LIST_PATH, 'a')
    data_str = "color_img,depth_img,uv_img,color_gt,depth_gt"
    output_data(fd_test_list, data_str)
    return fd_test_list


def output_data(output_file: object, data: str) -> None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def produce_list(folder_list, fd_test_list):
    total = 0
    off_set = 1
    for i in range(len(folder_list)):

        for num in (list(range(1,31))+list(range(330,360))):

            color_path = gen_color_path(folder_list[i], num)
            depth_path = gen_depth_path(folder_list[i], num)
            uv_path = gen_uv_path(folder_list[i], num)

            color_path_is_exists = os.path.exists(color_path)
            depth_path_is_exists = os.path.exists(depth_path)
            uv_path_is_exists = os.path.exists(uv_path)

            if (not color_path_is_exists) and (not depth_path_is_exists)\
                   (not uv_path_is_exists):
                break

            data_str = color_path + ',' + depth_path + ',' + uv_path 
            output_data(fd_test_list, data_str)


            total = total + 1

    return total

def gen_list(fd_test_list):
    folder_list = ['0356','0357','0376','0377','0378']
    total = produce_list(folder_list, fd_test_list)
    return total


def main() -> None:
    fd_test_list = open_file()
    total = gen_list(fd_test_list)
    print(total)


if __name__ == '__main__':
    main()
