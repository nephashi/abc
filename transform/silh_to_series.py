import math
'''
该脚本将轮廓图像序列转化为时间序列
'''
from transform import SILH_POSE_ESTIMATOR
from copy import deepcopy
import utils

class SilhoueteeTransformer(object):

    def __update_dict(self, list_dict, src_dict):
        for k, v in src_dict.items():
            if k in list_dict:
                list_dict[k].append(v)

    def __get_angle(self, x1, y1, x2, y2):
        '''
        给定两个点，计算和垂线的夹角。向右倾斜为正，向左倾斜为负
        :param point1: (x1, y1)，处在上方的点
        :param point2: (x2, y2)，处在下方的点
        :return:
        '''
        theta = math.atan((float(x2) - float(x1)) / (float(y2) - float(y1)))
        return theta

    def __slide_window_smooth(self, data, n):
        '''
        滑动窗口取平均进行平滑处理
        :param data: [e1,e2,e3,e4,.....]
        :param n: 如果是偶数会被减一处理
        :return:
        '''
        n -= 1
        rt = []
        for i in range(len(data)):
            sum = 0.0
            count = 0.0
            for j in range(int(-n/2), int(n/2) + 1):
                if i + j >= 0 and i + j < len(data):
                    sum += float(data[i + j])
                    count += 1
            rt.append(sum / count)
        return rt

    def __get_area_and_centre(self, crop_img):
        count = 0
        y_sum = 0
        for i in range(len(crop_img)):
            for j in range(len(crop_img[0])):
                if (crop_img[i][j] == 255):
                    count += 1
                    y_sum += i
        return count, len(crop_img) - y_sum / count

    def get_images_feature_with_fcn_pose_estimate(self, images):
        pass


    def get_images_feature_with_silh_pose_estimate(self, images):
        '''
        输入图像序列，返回特征集
        :param images:
        :return: 一个字典，保存多维步态特征
        {
            'lknee_angle': [,,,,,,,],
            'rknee_angle': [,,,,,,,],
            ......,
            'area': [,,,,,,],
            'w-h-ratio': [,,,,,,]
        }
        '''
        rt = {
            'l_thigh_angle': [],
            'r_thigh_angle': [],
            'l_shin_angle': [],
            'r_shin_angle': [],
            'neck_angle': [],
            'hw_ratio': [],
            'area': [],
            'height': [],
            'width': [],
            'y_centre': []
        }
        crop_images = []
        for idx, im in enumerate(images):
            crop_images.append(SILH_POSE_ESTIMATOR.crop_silhouetee(im))
        slope = SILH_POSE_ESTIMATOR.estimate_thigh_slope_for_image_series(crop_images)
        for idx, im in enumerate(crop_images):
            feature = self.__get_image_feature_with_silh_pose_estimate(im, slope[idx])
            self.__update_dict(rt, feature)
        # for k, v in rt.items():
        #     tmp = v
        #     tmp2 = self.__slide_window_smooth(tmp, 3)
        #     rt[k] = tmp2
        return rt

    def __get_image_feature_with_silh_pose_estimate(self, image, slope):
        '''
        返回一个图像的特征字典
        :param image:
        :param feature_dic:
        :return:
        '''
        pose_dict = SILH_POSE_ESTIMATOR.pose_estimate_for_single_silh(image, slope[0], slope[1], slope[2],
                                                            slope[3])
        return self.__get_image_feature_with_pose_dict(image, pose_dict)

    def __get_image_feature_with_pose_dict(self, image, pose_dict):
        '''
        从关节点字典得到特征字典
        :param image:
        :param pose_dict:
        :return:
        '''
        l_thigh_angle = self.__get_angle(pose_dict['l_hip'][0], pose_dict['l_hip'][1],
                                         pose_dict['l_knee'][0], pose_dict['l_knee'][1])
        r_thigh_angle = self.__get_angle(pose_dict['r_hip'][0], pose_dict['r_hip'][1],
                                         pose_dict['r_knee'][0], pose_dict['r_knee'][1])
        l_shin_angle = self.__get_angle(pose_dict['l_knee'][0], pose_dict['l_knee'][1],
                                        pose_dict['l_ankle'][0], pose_dict['l_ankle'][1])
        r_shin_angle = self.__get_angle(pose_dict['r_knee'][0], pose_dict['r_knee'][1],
                                        pose_dict['r_ankle'][0], pose_dict['r_ankle'][1])
        neck_angle = self.__get_angle(pose_dict['neck'][0], pose_dict['neck'][1],
                                      pose_dict['shoulder'][0], pose_dict['shoulder'][1])
        height = len(image)
        width = len(image[0])
        area, y_centre = self.__get_area_and_centre(image)
        hw_ratio = len(image) / len(image[0])
        rt = {
            'l_thigh_angle': l_thigh_angle,
            'r_thigh_angle': r_thigh_angle,
            'l_shin_angle': l_shin_angle,
            'r_shin_angle': r_shin_angle,
            'neck_angle': neck_angle,
            'hw_ratio': hw_ratio,
            'area': area,
            'height': height,
            'width': width,
            'y_centre': y_centre
        }
        return rt


def transfer_data(path, output_path):
    '''
    按照目录中的数据格式调用转换函数
    transfer_data('H:/gait_experiment/silh', 'H:/gait_experiment/TS')
    :param path:
    :return:
    '''
    import json
    from load import load_images
    st = SilhoueteeTransformer()
    subject_list = []
    for i in range(1, 60):
        subject_list.append(i)
    for id in subject_list:
        print('subject:' + str(id))
        for i in range(1, 7):
            pre = utils.extend_digit_string(str(id), 3)
            folder_name = pre + '-0' + str(i)
            full_path = path + '/' + folder_name
            images = load_images(full_path)
            feature = st.get_images_feature_with_silh_pose_estimate(images)

            diction = {
                'subject': id,
                'scene': i,
                'feature': feature
            }

            js = json.dumps(diction)

            full_output_path = output_path + '/' + pre + '/' + str(i) + '.json'
            with open(full_output_path, 'w') as f:
                f.write(js)


def demo1():
    st = SilhoueteeTransformer()
    from load import load_images
    images = load_images('H:/gait_experiment/silh/001-02')
    feature = st.get_images_feature_with_silh_pose_estimate(images)

def to_dataset():
    from load import load_ts
    objs = [90,91,92,93,94]
    feature = 'l_shin_angle'
    ts = load_ts('H:/gait_experiment/TS', objs)
    with open('H:/gait_experiment/dataset/B/B_TRAIN.txt', 'w') as f:
        count = 0
        for sub_id in objs:
            for i in range(0, 4):
                data = ts[sub_id][i]['feature'][feature][0:44]
                print(len(data))
                f.write(str(count))
                f.write(',')
                for idx, d in enumerate(data):
                    f.write('%.4f' % d)
                    if idx < len(data) - 1:
                        f.write(',')
                f.write('\n')
            count += 1
    with open('H:/gait_experiment/dataset/B/B_TEST.txt', 'w') as f:
        count = 0
        for sub_id in objs:
            for i in range(4, 6):
                data = ts[sub_id][i]['feature'][feature][0:50]
                f.write(str(count))
                f.write(',')
                for idx, d in enumerate(data):
                    f.write('%.4f' % d)
                    if idx < len(data) - 1:
                        f.write(',')
                f.write('\n')
            count += 1

def to_multi_dataset():
    from load import load_ts
    objs = [1, 3,4,5,7,104,113,117]
    ts = load_ts('H:/gait_experiment/TS', objs)
    features = [
        'l_thigh_angle',
        'r_thigh_angle',
        'l_shin_angle',
        'r_shin_angle',
        'neck_angle',
        'hw_ratio',
        'area',
        'height',
        'width',
        'y_centre'
    ]
    for feature in features:
        with open('H:/gait_experiment/dataset/B/B_' + feature + '_TRAIN.txt', 'w') as f:
            count = 0
            for sub_id in objs:
                for i in range(0, 4):
                    data = ts[sub_id][i]['feature'][feature][0:44]
                    print(len(data))
                    f.write(str(count))
                    f.write(',')
                    for idx, d in enumerate(data):
                        f.write('%.4f' % d)
                        if idx < len(data) - 1:
                            f.write(',')
                    f.write('\n')
                count += 1
        with open('H:/gait_experiment/dataset/B/B_' + feature + '_TEST.txt', 'w') as f:
            count = 0
            for sub_id in objs:
                for i in range(4, 6):
                    data = ts[sub_id][i]['feature'][feature][0:44]
                    f.write(str(count))
                    f.write(',')
                    for idx, d in enumerate(data):
                        f.write('%.4f' % d)
                        if idx < len(data) - 1:
                            f.write(',')
                    f.write('\n')
                count += 1



if __name__ == '__main__':
    transfer_data('H:/gait_experiment/silh', 'H:/gait_experiment/TS')
    #to_multi_dataset()