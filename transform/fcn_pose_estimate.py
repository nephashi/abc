import cv2
import numpy as np
from transform.human_pose_nn import HumanPoseIRNetwork
from utils import COUNTER

class FCNPoseEstimator(object):

    def __init__(self, model_path):
        self.__net_pose = HumanPoseIRNetwork()
        self.__net_pose.restore(model_path)

    def __swap(self, dlist, idxa, idxb):
        tmp = dlist[idxa]
        dlist[idxa] = dlist[idxb]
        dlist[idxb] = tmp

    def silh_extract(self, frame, bkgd):
        '''
        返回背景减除后的二值图像，以及前景的像素大小，以及前景四角的坐标
        如果没前景或者前景太小会返回空
        如果前景左右侧沾边会返回空
        :param frame:
        :param bkgd:
        :return:
        '''
        def pixel_identity(x, y):
            diff = abs(int(x[0]) - int(y[0])) + abs(int(x[1]) - int(y[1])) + abs(int(x[2]) - int(y[2]))
            return diff <= 100

        b_rows = len(bkgd)
        b_cols = len(bkgd[0])
        f_rows = len(frame)
        f_cols = len(frame[0])

        if (b_rows != f_rows or b_cols != f_cols):
            raise Exception

        sub = np.zeros((b_rows, b_cols), dtype='uint8')
        for i in range(b_rows):
            for j in range(b_cols):
                if (pixel_identity(bkgd[i][j], frame[i][j])):
                    sub[i][j] = 0
                else:
                    sub[i][j] = 255

        # label: 二维矩阵，每个元素是对应元素属于哪个联通区的标号
        ret, labels = cv2.connectedComponents(sub, connectivity=4)

        label_count = {}
        for i in range(b_rows):
            for j in range(b_cols):
                label_count.setdefault(labels[i][j], 0)
                label_count[labels[i][j]] += 1
        if len(label_count) < 2:
            return None, 0, [-1, -1, -1, -1]

        max_label = -1
        second_max_label = -1
        max_value = -1
        second_max_value = -1
        for k, v in label_count.items():
            if (v > max_value):
                max_label = k
                max_value = v
        for k, v in label_count.items():
            if (v > second_max_value and k != max_label):
                second_max_label = k
                second_max_value = v

        ans = np.zeros((b_rows, b_cols), dtype='uint8')
        x1 = b_cols
        x2 = 0
        y1 = b_rows
        y2 = 0
        for i in range(b_rows):
            for j in range(b_cols):
                if (labels[i][j] == second_max_label):
                    ans[i][j] = 255
                    if j < x1:
                        x1 = j
                    if j > x2:
                        x2 = j
                    if i < y1:
                        y1 = i
                    if i > y2:
                        y2 = i
                else:
                    ans[i][j] = 0

        left_margin_count = 0
        right_margin_count = 0
        for i in range(b_rows):
            if ans[i][0] == 255:
                left_margin_count += 1
            if ans[i][b_cols - 1] == 255:
                right_margin_count += 1

        cv2.imwrite('./sub.png', ans)
        if second_max_value < 1500:
            print('foreground too little')
            return None, 0, [-1, -1, -1, -1]
        if left_margin_count > b_rows / 5 or right_margin_count > b_rows / 5:
            print('margin_touched')
            return None, 0, [-1, -1, -1, -1]
        else:
            return ans, second_max_value, [x1, x2, y1, y2]

    def __crop_and_make_up(self, image, x1, x2, y1, y2):
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        height = len(image)
        width = len(image[0])
        crop_image = np.zeros((y2 - y1 + 1, x2 - x1 + 1, 3), dtype='uint8')
        for i in range(y1, y2 + 1):
            for j in range(x1, x2 + 1):
                if (i >= 0 and i < height and j >= 0 and j < width):
                    crop_image[i - y1][j - x1] = image[i][j]
        return crop_image

    def pose_estimate_for_single_image(self, image, sub_image, pixel_count, coor):
        if sub_image is None:
            return None
        standard_height = 180
        assert (coor[3] - coor[2]) < standard_height
        upper = (coor[2] + coor[3]) / 2 - standard_height / 2
        bottomm = (coor[2] + coor[3]) / 2 + standard_height / 2
        left = (coor[0] + coor[1]) / 2 - standard_height / 2
        rightt = (coor[0] + coor[1]) / 2 + standard_height / 2

        model_input_image = self.__crop_and_make_up(image, left, rightt, upper, bottomm)
        model_input_image = cv2.resize(model_input_image, (299, 299))
        cv2.imwrite('./model_input.png', model_input_image)
        img_batch = np.expand_dims(model_input_image, 0)
        # joint_names = [
        #     'right ankle ',
        #     'right knee ',
        #     'right hip',
        #     'left hip',
        #     'left knee',
        #     'left ankle',
        #     'pelvis',
        #     'thorax',
        #     'upper neck',
        #     'head top',
        #     'right wrist',
        #     'right elbow',
        #     'right shoulder',
        #     'left shoulder',
        #     'left elbow',
        #     'left wrist'
        # ]
        # y, x是一维列表，存储关节点坐标
        y, x, a = self.__net_pose.estimate_joints(img_batch)
        y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)
        if x[0] < x[5]:
            self.__swap(x, 0, 5)
            self.__swap(y, 0, 5)
        if x[1] < x[4]:
            self.__swap(x, 1, 4)
            self.__swap(y, 1, 4)
        if x[2] < x[3]:
            self.__swap(x, 2, 3)
            self.__swap(y, 2, 3)
        body_part_dic = {}
        body_part_dic['neck'] = (x[9], y[9])
        body_part_dic['shoulder'] = (x[8], y[8])
        body_part_dic['l_hip'] = (x[3], y[3])
        body_part_dic['l_knee'] = (x[4], y[4])
        body_part_dic['l_ankle'] = (x[5], y[5])
        body_part_dic['r_hip'] = (x[2], y[2])
        body_part_dic['r_knee'] = (x[1], y[1])
        body_part_dic['r_ankle'] = (x[0], y[0])

        image_cp = model_input_image.copy()
        for i in range(2):
            cv2.line(image_cp, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 255, 0), 5)
        for i in range(2, 5):
            cv2.line(image_cp, (x[i], y[i]), (x[i + 1], y[i + 1]), (255, 0, 0), 5)
        cv2.line(image_cp, (x[8], y[8]), (x[9], y[9]), (255, 0, 0), 5)
        cv2.imwrite('./imgs/structure/structure_' + str(COUNTER.count) +  '.png', image_cp)
        COUNTER.increase()
        return body_part_dic

if __name__ == '__main__':
    from transform import FCN_POSE_ESTIMATOR
    bkgd = cv2.imread('D:/pyWorkspace/GaitShapelet/bkgd.png')
    fore = cv2.imread('D:/pyWorkspace/GaitShapelet/45.png')
    FCN_POSE_ESTIMATOR.pose_estimate_for_single_image(fore, bkgd)