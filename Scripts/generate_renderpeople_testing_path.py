# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = '/home/lixing/Documents/BodyReconstruction_test/'  # root path

# the file's path and format
# RAW_DATA_FOLDER = 'Kitti2012/training/%s/'
RAW_DATA_FOLDER = 'RenderPeople/'
RGB_FOLDER = 'RENDER/%s/'
DEPTH_FOLDER = 'DEPTH/%s/'
UV_FOLDER = 'UV/%s/'
FILE_NAME = '%04d_0_00'
UV_FILE_NAME = '%04d_0_00.0001'
#UV_FILE_NAME = 'dp.%04d'

RGB_GT_FOLDER = 'RENDER_GT/%s/'
DEPTH_GT_FOLDER = 'DEPTH_GT/%s/'

# file type
RAW_COLOR_TYPE = '.jpg'
RAW_DEPTH_TYPE = '.png'

# the output's path,
TRAIN_LIST_PATH = './Datasets/renderpeople_testing_list.csv'


# IMG_NUM = 194  # the dataset's total image
IMG_NUM = 100    # the dataset's total image
TIMES = 70      # the sample of val

TEST_FLAG = True


def gen_color_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def gen_color_gt_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_GT_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_gt_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_GT_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path


def gen_uv_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + UV_FOLDER % file_folder + UV_FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')


    data_str = "color_img,depth_img,uv_img,color_gt,depth_gt"
    output_data(fd_train_list, data_str)
    

    return fd_train_list


def output_data(output_file: object, data: str) -> None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def produce_list(folder_list, fd_train_list):
    total = 0
    off_set = 1
    for i in range(len(folder_list)):
        num_gt = 0
        for num in (list(range(0,31,5))+list(range(330,360,5))):
        #for num in (list(range(1,16))+list(range(345,360))):
            print(num)
            color_path = gen_color_path(folder_list[i], num)
            depth_path = gen_depth_path(folder_list[i], num)

            uv_path = gen_uv_path(folder_list[i], num)

            if num <180:
                num_gt = num + 180
            else:
                num_gt = num - 180
            color_gt_path = gen_color_gt_path(folder_list[i], num_gt)
            depth_gt_path = gen_depth_gt_path(folder_list[i], num_gt)
            

            color_path_is_exists = os.path.exists(color_path)
            depth_path_is_exists = os.path.exists(depth_path)
            uv_path_is_exists = os.path.exists(uv_path)
            color_gt_path_is_exists = os.path.exists(color_gt_path)
            depth_gt_path_is_exists = os.path.exists(depth_gt_path)
            
            if (not color_path_is_exists) and \
                    (not depth_path_is_exists) and\
                    (not uv_path_is_exists) and \
                    (not color_gt_path_is_exists) and \
                    (not depth_gt_path_is_exists):
                break
            
                


            data_str = color_path + ',' + depth_path + ',' + uv_path + ',' + color_gt_path + ',' + depth_gt_path
            output_data(fd_train_list, data_str)

            total = total + 1

    return total

def gen_list(fd_train_list):

    filePath = '/home/lixing/Documents/BodyReconstruction_test/RenderPeople/DEPTH'
    folder_list = os.listdir(filePath)

    print(folder_list[0])
    total = produce_list(folder_list, fd_train_list)
    return total


def main() -> None:
    fd_train_list = open_file()
    total = gen_list(fd_train_list)
    print(total)


if __name__ == '__main__':
    main()
