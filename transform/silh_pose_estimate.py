import math
from copy import deepcopy

import cv2

from load import load_images
from utils import COUNTER

class SilhPoseEstimator(object):

    __body_part_location = {
        'head': 0.064,
        'neck': 0.130,
        'shoulder': 0.182,
        'chest': 0.280,
        'waist': 0.470,
        'pelvis': 0.520,
        'hip': 0.580, #大腿根部
        'knee': 0.752,
        'ankle': 0.920
    }

    __thigh_length = 0.185
    __shin_length = 0.193

    # 在检测大腿边缘时，有可能在上方误检测到手臂，这个参数限制了启动误检测的时机
    __thigh_abnormal_abortion_limit = 10
    # 和上一行像素点的横坐标差出这么多，就认为之前的点是手臂
    abnormal_interval = 5

    def __update_dict(self, list_dict, src_dict):
        for k, v in src_dict.items():
            if k in list_dict:
                list_dict[k].append(v)

    def crop_silhouetee(self, image):
        '''
        这个函数把步态轮廓的部分从整个图片里找出来
        :param image:
        :return:
        '''
        if (len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_rows = len(image)
        f_cols = len(image[0])
        max_x = -1
        max_y = -1
        min_x = f_cols
        min_y = f_rows
        for i in range(f_rows):
            for j in range(f_cols):
                if image[i][j] == 255:
                    if i < min_y:
                        min_y = i
                    if i > max_y:
                        max_y = i
                    if j < min_x:
                        min_x = j
                    if j > max_x:
                        max_x = j
        return image[min_y:max_y, min_x:max_x]

    def __get_mid_coordinate(self, row):
        '''

        :param row:
        :return:
        '''
        max_idx = -1
        min_idx = len(row)
        for i in range(len(row)):
            if (row[i] == 255):
                if (i < min_idx):
                    min_idx = i
                if (i > max_idx):
                    max_idx = i

        return (min_idx + max_idx) / 2

    def __get_largest_intervals_mid_corrdinate(self, row):
        ''''''
        start = False
        largest_count = -1
        largest_start = -1
        largest_end = -1
        start_index = -1
        for index, pixel in enumerate(row):
            if pixel == 255:
                if start == False:
                    start_index = index
                    start = True
            else:
                if start == True:
                    start = False
                    if (index - start_index > largest_count):
                        largest_count = index - start_index
                        largest_start = start_index
                        largest_end = index
        if (start == True):
            if (len(row) - 1 - start_index > largest_count):
                largest_count = len(row) - 1 - start_index
                largest_start = start_index
                largest_end = len(row) - 1
        if largest_count == -1:
            return len(row) / 2
        else:
            return (largest_start + largest_end) / 2


    def __get_largest_intervals_one_third_and_two_third_corrdinate(self, row):
        ''''''
        start = False
        largest_count = -1
        largest_start = -1
        largest_end = -1
        start_index = -1
        for index, pixel in enumerate(row):
            if pixel == 255:
                if start == False:
                    start_index = index
                    start = True
            else:
                if start == True:
                    start = False
                    if (index - start_index > largest_count):
                        largest_count = index - start_index
                        largest_start = start_index
                        largest_end = index
        if (start == True):
            if (len(row) - 1 - start_index > largest_count):
                largest_count = len(row) - 1 - start_index
                largest_start = start_index
                largest_end = len(row) - 1
        if largest_count == -1:
            return len(row) / 2
        else:
            return largest_start + (largest_end - largest_start) / 3, largest_start + (largest_end - largest_start) * 2 / 3


    def __get_border_corrdinate(self, row):
        '''

        :param row:
        :return:
        '''
        left = -1
        right = -1
        for i in range(0, len(row)):
            if (row[i] == 255):
                left = i
                break
        for i in range(len(row) - 1, -1, -1):
            if (row[i] == 255):
                right = i
                break

        return left, right

    def check_range(self, tuple, height, width):
        x = tuple[0]
        y = tuple[1]
        if (y < 0):
            y = 0
        if (y >= height):
            y = height - 1
        if (x < 0):
            x = 0
        if (x >= width):
            x = width - 1
        return (x, y)

    def pose_estimate_for_single_silh(self, silhouetee_image, l_thigh_k, r_thigh_k, l_shin_k, r_shin_k):
        '''

        :param silhouetee_image: crop_silhoutee得到的轮廓图像
        :return: 一个string-tuple字典，代表各个位置的坐标
        '''
        if (len(silhouetee_image.shape) == 3):
            silhouetee_image = cv2.cvtColor(silhouetee_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./tmp3.png', silhouetee_image)
        height = len(silhouetee_image)
        width = len(silhouetee_image[0])
        thigh_length = height * self.__thigh_length
        shin_length = height * self.__shin_length
        body_part_dic = {}
        list1 = ['head', 'neck', 'shoulder', 'chest']
        for part in list1:
            y = height * self.__body_part_location[part]
            x = self.__get_mid_coordinate(silhouetee_image[math.floor(y)])
            body_part_dic[part] = (x, y)

        # y_waist = height * self.__body_part_location['waist']
        #
        # tmp1 = float(body_part_dic['chest'][0] - body_part_dic['shoulder'][0])
        # tmp2 = float(body_part_dic['chest'][1] - body_part_dic['shoulder'][1])
        # tmp3 = float(y_waist - body_part_dic['chest'][1])
        #
        # x_waist = body_part_dic['chest'][0] + tmp1 * tmp3 / tmp2
        # body_part_dic['waist'] = (x_waist, y_waist)
        # body_part_dic['pelvis'] = (x_waist, height * self.__body_part_location['pelvis'])

        left_x_hip, right_x_hip = self.__get_largest_intervals_one_third_and_two_third_corrdinate(silhouetee_image[math.floor(height * self.__body_part_location['hip'])])
        body_part_dic['l_hip'] = (left_x_hip, height * self.__body_part_location['hip'])
        body_part_dic['r_hip'] = (right_x_hip, height * self.__body_part_location['hip'])

        l_thigh_theta = math.atan(l_thigh_k)
        r_thigh_theta = math.atan(r_thigh_k)
        l_shin_theta = math.atan(l_shin_k)
        r_shin_theta = math.atan(r_shin_k)

        body_part_dic['l_knee'] = self.check_range((body_part_dic['l_hip'][0] + thigh_length * math.cos(l_thigh_theta) * (1 if l_thigh_k > 0 else -1),
                                   body_part_dic['l_hip'][1] + thigh_length * abs(math.sin(l_thigh_theta))), height, width)
        body_part_dic['r_knee'] = self.check_range((body_part_dic['r_hip'][0] + thigh_length * math.cos(r_thigh_theta) * (1 if r_thigh_k > 0 else -1),
                                   body_part_dic['r_hip'][1] + thigh_length * abs(math.sin(r_thigh_theta))), height, width)
        body_part_dic['l_ankle'] = self.check_range((body_part_dic['l_knee'][0] + shin_length * math.cos(l_shin_theta) * (1 if l_shin_k > 0 else -1),
                                    body_part_dic['l_knee'][1] + shin_length * abs(math.sin(l_shin_theta))), height, width)
        body_part_dic['r_ankle'] = self.check_range((body_part_dic['r_knee'][0] + shin_length * math.cos(r_shin_theta) * (1 if r_shin_k > 0 else -1),
                                    body_part_dic['r_knee'][1] + shin_length * abs(math.sin(r_shin_theta))), height, width)

        im_cp = deepcopy(silhouetee_image)
        im_cp = cv2.cvtColor(im_cp, cv2.COLOR_GRAY2BGR)
        for k, v in body_part_dic.items():
            try:
                im_cp[math.floor(v[1])][math.floor(v[0])] = [0, 255, 0]
            except Exception as e:
                print(k)
                print(e)
        #cv2.imwrite('./imgs/' + str(COUNTER.count) + '.png', im_cp)
        COUNTER.increase()

        return body_part_dic

    def __linear_regression_for_thigh(self, data):
        total_k, total_f = self.__linear_regression(data)
        upper_data = data[:math.floor(len(data) / 2)]
        latter_data = data[math.floor(len(data) / 2):]
        upper_k, upper_f = self.__linear_regression(upper_data)
        latter_k, latter_f = self.__linear_regression(latter_data)
        if upper_k * latter_k > 0:
            return total_k, total_f
        else:
            return upper_k, upper_f

    def __linear_regression_for_shin(self, data):
        total_k, total_f = self.__linear_regression(data)
        upper_data = data[:math.floor(len(data) / 2)]
        latter_data = data[math.floor(len(data) / 2):]
        upper_k, upper_f = self.__linear_regression(upper_data)
        latter_k, latter_f = self.__linear_regression(latter_data)
        if upper_k * latter_k > 0 and upper_k != 1000 and latter_k != 1000:
            return total_k, total_f
        else:
            return upper_k, upper_f

    def __linear_regression(self, data):
        '''
        一元线性回归，返回斜率
        :param data: [(x1,y1), (x2, y2), ....]
        :return:
        '''
        import numpy as np
        x = []
        y = []
        for i in range(len(data)):
            x.append(data[i][1])
            y.append(data[i][0])
        meanx = np.mean(x)
        meany = np.mean(y)
        tmpa = np.subtract(y, meany)
        tmpb = np.subtract(x, meanx)
        tmpc = np.dot(tmpa, tmpb)
        tmpd = np.dot(tmpb, tmpb)
        if tmpd == 0:
            #垂直线
           return 1000, 1
        k = tmpc / tmpd
        b = meany - k * meanx
        f = 1
        for i in range(len(data)):
            pred = k * data[i][0] + b
            f += abs(data[i][1] - pred)
        return 1000 if k == 0 else 1.0 / k, 1.0 / math.sqrt(f)

    def __slide_window_smooth(self, data, factor, n):
        n -= 1
        rt = []
        for i in range(len(data)):
            sum = 0.0
            sum_f = 0.0
            for j in range(int(-n/2), int(n/2) + 1):
                if i + j >= 0 and i + j < len(data):
                    sum += float(data[i + j]) * float(factor[i + j])
                    sum_f += float(factor[i + j])
            rt.append(sum / sum_f)
        return rt

    def estimate_thigh_slope_for_image_series(self, images):
        '''
        输入步态轮廓序列，估计每一帧的左右腿倾斜角度
        :param images: 步态轮廓序列，一个图像list
        :return: 和输入长度相同的list，元素是一个四元tuple，存储左右大小腿的倾斜斜率: [(左大腿斜率，右大腿斜率，左小腿斜率，右小腿斜率),(....),(....),....]
        '''
        rt = []
        l_thigh_k_list = []
        l_thigh_f_list = []
        r_thigh_k_list = []
        r_thigh_f_list = []
        l_shin_k_list = []
        l_shin_f_list = []
        r_shin_k_list = []
        r_shin_f_list = []
        for idx, image in enumerate(images):
            if (len(image.shape) == 3):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_cp = deepcopy(image)
            img_cp = cv2.cvtColor(img_cp, cv2.COLOR_GRAY2BGR)
            cv2.imwrite('./tmp.png', img_cp)
            height = len(image)
            y_hip = height * self.__body_part_location['hip']
            y_knee = height * self.__body_part_location['knee']
            y_ankle = height * self.__body_part_location['ankle']
            left_thigh_border_data = []
            right_thigh_border_data = []
            left_shin_border_data = []
            right_shin_border_data = []

            for tmp_y in range(math.floor(y_hip), math.floor(y_knee)):
                l, r = self.__get_border_corrdinate(image[tmp_y])
                if len(left_thigh_border_data) > 0 and tmp_y - math.floor(y_hip) < self.__thigh_abnormal_abortion_limit \
                        and abs(l - left_thigh_border_data[-1][0]) > self.abnormal_interval:
                    left_thigh_border_data = []
                if len(right_thigh_border_data) > 0 and tmp_y - math.floor(y_hip) < self.__thigh_abnormal_abortion_limit \
                        and abs(r - right_thigh_border_data[-1][0]) > self.abnormal_interval:
                    right_thigh_border_data = []
                if len(left_thigh_border_data) == 0 or abs(l - left_thigh_border_data[-1][0]) <= self.abnormal_interval:
                    left_thigh_border_data.append((l, tmp_y))
                if len(right_thigh_border_data) == 0 or abs(r - right_thigh_border_data[-1][0]) <= self.abnormal_interval:
                    right_thigh_border_data.append((r, tmp_y))

            for tmp_y in range(math.floor(y_knee), math.floor(y_ankle)):
                l, r = self.__get_border_corrdinate(image[tmp_y])
                if len(left_shin_border_data) == 0 or abs(l - left_shin_border_data[-1][0]) <= self.abnormal_interval:
                    left_shin_border_data.append((l, tmp_y))
                if len(right_shin_border_data) == 0 or abs(r - right_shin_border_data[-1][0]) <= self.abnormal_interval:
                    right_shin_border_data.append((r, tmp_y))

            for l, tmp_y in left_thigh_border_data:
                img_cp[tmp_y][l] = [255, 0, 0]

            for r, tmp_y in right_thigh_border_data:
                img_cp[tmp_y][r] = [255, 0, 0]

            for l, tmp_y in left_shin_border_data:
                img_cp[tmp_y][l] = [0, 255, 0]

            for r, tmp_y in right_shin_border_data:
                img_cp[tmp_y][r] = [0, 255, 0]

            cv2.imwrite('./border_imgs/tmp' + str(idx) + '.png', img_cp)

            l_thigh_k, l_thigh_f = self.__linear_regression_for_thigh(left_thigh_border_data)
            r_thigh_k, r_thigh_f = self.__linear_regression_for_thigh(right_thigh_border_data)
            l_shin_k, l_shin_f = self.__linear_regression_for_shin(left_shin_border_data)
            r_shin_k, r_shin_f = self.__linear_regression_for_shin(right_shin_border_data)

            l_thigh_k_list.append(l_thigh_k)
            l_thigh_f_list.append(l_thigh_f)
            r_thigh_k_list.append(r_thigh_k)
            r_thigh_f_list.append(r_thigh_f)
            l_shin_k_list.append(l_shin_k)
            l_shin_f_list.append(l_shin_f)
            r_shin_k_list.append(r_shin_k)
            r_shin_f_list.append(r_shin_f)

        # l_thigh_k_list = self.__slide_window_smooth(l_thigh_k_list, l_thigh_f_list, 3)
        # r_thigh_k_list = self.__slide_window_smooth(r_thigh_k_list, r_thigh_f_list, 3)
        # l_shin_k_list = self.__slide_window_smooth(l_shin_k_list, l_shin_f_list, 3)
        # r_shin_k_list = self.__slide_window_smooth(r_shin_k_list, r_shin_f_list, 3)

        for i in range(len(l_thigh_k_list)):
            rt.append((l_thigh_k_list[i], r_thigh_k_list[i], l_shin_k_list[i], r_shin_k_list[i]))

        return rt


if __name__ == '__main__':
    images = load_images('H:/gait_experiment/silh/008-06')
    pe = SilhPoseEstimator()
    crop_images = []
    for idx, im in enumerate(images):
        print(idx)
        crop_images.append(pe.crop_silhouetee(im))
    slope = pe.estimate_thigh_slope_for_image_series(crop_images)
    for idx, im in enumerate(crop_images):
        dic = pe.pose_estimate_for_single_silh(im, slope[idx][0], slope[idx][1], slope[idx][2], slope[idx][3])
        im_cp = deepcopy(im)
        im_cp = cv2.cvtColor(im_cp, cv2.COLOR_GRAY2BGR)
        for k, v in dic.items():
            try:
                im_cp[math.floor(v[1])][math.floor(v[0])] = [0, 255, 0]
            except Exception as e:
                print(k)
                print(e)
        cv2.imwrite('./imgs/' + str(idx) + '.png', im_cp)
