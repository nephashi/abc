import os
import cv2
import utils
import json

def load_images(path):
    '''
    从cisia-b读取轮廓图像，文件在文件夹中的排列应该是有序的
    :param path:
    :return:
    '''
    file_names = os.listdir(path)
    rt = []
    for fn in file_names:
        rt.append(cv2.imread(path + '/' + fn))
    return rt

def load_ts(path, subjects):
    '''
    从目录中读取时间序列数据
    tmp = load_ts('H:/gait_experiment/TS')
    :param path:
    :param subjects: 人id列表：[1,8,105,...]
    :return:
    '''
    rt = {}
    for id in subjects:
        rt[id] = []
        path1 = path + '/' + utils.extend_digit_string(str(id), 3) + '/'
        file_names = os.listdir(path1)
        for fn in file_names:
            full_file_name = path1 + '/' + fn
            with open(full_file_name) as f:
                jsn = f.read()
            rt[id].append(json.loads(jsn))
    return rt

if __name__ == '__main__':
    tmp = load_ts('H:/gait_experiment/TS')