from transform import FCN_POSE_ESTIMATOR
import math
import cv2
from utils import COUNTER
import utils
import json

class FCNVideoTransformer(object):

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

    def __update_dict(self, list_dict, src_dict):
        for k, v in src_dict.items():
            if k in list_dict:
                list_dict[k].append(v)

    def __get_area_and_centre(self, crop_img):
        cv2.imwrite('./1111.png', crop_img)
        count = 0
        y_sum = 0
        for i in range(len(crop_img)):
            for j in range(len(crop_img[0])):
                if (crop_img[i][j] == 255):
                    count += 1
                    y_sum += i
        return count, len(crop_img) - y_sum / count

    def __get_angle(self, x1, y1, x2, y2):
        '''
        给定两个点，计算和垂线的夹角。向右倾斜为正，向左倾斜为负
        :param point1: (x1, y1)，处在上方的点
        :param point2: (x2, y2)，处在下方的点
        :return:
        '''
        theta = math.atan((float(x2) - float(x1)) / (float(y2) - float(y1)))
        return theta

    def extract_feature_for_image(self, image, bkgd):
        sub_image, pixel_count, coor = FCN_POSE_ESTIMATOR.silh_extract(image, bkgd)
        if sub_image is None:
            return None
        pose_dict = FCN_POSE_ESTIMATOR.pose_estimate_for_single_image(image, sub_image, pixel_count, coor)
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
        height = coor[3] - coor[2]
        width = coor[1] - coor[0]
        hw_ratio = float(height) / float(width)
        area, y_centre = self.__get_area_and_centre(sub_image[coor[2]:coor[3], coor[0]:coor[1]])
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

    def transform_video_to_dict(self, path):
        COUNTER.reset()
        vidcap = cv2.VideoCapture(path)
        success, bkgd = vidcap.read()
        cv2.imwrite('./imgs/bkgd.png', bkgd)
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
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            cv2.imwrite('./imgs/frame' + str(COUNTER.count) + '.png', frame)
            feature = self.extract_feature_for_image(frame, bkgd)
            if feature is not None:
                self.__update_dict(rt, feature)
        # for k, v in rt.items():
        #     tmp = v
        #     tmp2 = self.__slide_window_smooth(tmp, 3)
        #     rt[k] = tmp2
        return rt

    def transform_video(self, path, output_path):
        '''
        用这个把视频转换成特征序列
        :param path:
        :param output_path:
        :return:
        '''
        feature_dic = self.transform_video_to_dict(path)
        diction = {
            'subject': 0,
            'scene': 0,
            'feature': feature_dic
        }
        jsonstr = json.dumps(diction)
        with open(output_path, 'w') as f:
            f.write(jsonstr)

def transfer_casia_b(path, output_path):
    fcn = FCNVideoTransformer()
    object_ids = [1, 8]
    for oid in object_ids:
        for i in range(1, 7):
            print(str(oid) + '-' + str(i))
            filename = path + '/' + utils.get_video_file_name(oid, i)
            output_filename = output_path + '/' + utils.get_output_json_file_name(oid, i)
            fcn.transform_video(filename, output_filename)

def demo1():
    fcn = FCNVideoTransformer()
    fcn.transform_video('H:/gait_experiment/videos1/008-nm-01-090.avi', 'H:/gait_experiment/fcn_TS')


if __name__ == '__main__':
    transfer_casia_b('Z:/DatasetB/videos', 'H:/gait_experiment/fcn_TS')