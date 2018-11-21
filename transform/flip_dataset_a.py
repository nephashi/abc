import cv2
import os

root = 'M:/CASIA-Dataset-A/silhouettes/'
output_root = 'M:/CASIA-Dataset-A/silhouettes_b/'
subjects = os.listdir(root)
for subject in subjects:
    for i in [2, 4]:
        src_path = root + subject + '/00_' + str(i) + '/'
        dst_path = output_root + subject + '/00_' + str(i) + '/'
        image_names = os.listdir(src_path)
        for image_name in image_names:
            tmp_path = src_path + image_name
            tmp_path2 = dst_path + image_name
            img = cv2.imread(tmp_path)
            img = cv2.flip(img, 1)
            cv2.imwrite(tmp_path2, img)