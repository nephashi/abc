# 该脚本转换A数据集
import os
from transform.silh_to_series import SilhoueteeTransformer
import cv2
import numpy as np

def silh_extract(frame, bkgd):
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

    cv2.imwrite('./sub.png', sub)

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

image_path = 'M:/CASIA-Dataset-A/gaitdb/'
bkgd_path = 'M:/CASIA-Dataset-A/bkimages/'
output_root_path = 'M:/CASIA-Dataset-A/my_silhouetees/'
st = SilhoueteeTransformer()
subjects = os.listdir(image_path)
for subject in subjects:
    for i in range(1, 5):
        cur_bkgd_path = bkgd_path + subject + '-00_' + str(i) + '-bk.png'
        bkgd_image = cv2.imread(cur_bkgd_path)
        cur_path = image_path + subject + '/' + '00_' + str(i) + '/'
        images_names = os.listdir(cur_path)
        for image_name in images_names:
            output_path = output_root_path + subject + '/'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            output_path += str(i) + '/'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            output_path += image_name
            tmp = cur_path + image_name
            fore_image = cv2.imread(tmp)
            sub, _, __ = silh_extract(fore_image, bkgd_image)
            cv2.imwrite(output_path, sub)