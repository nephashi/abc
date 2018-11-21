import utils
import shutil

def copy_file(object_id):
    for i in range(1, 7):
        src_path = 'H:\DatasetB\silhouettes/'
        src_path += utils.extend_digit_string(str(object_id), 3)
        src_path += '/'
        src_path += utils.extend_digit_string(str(object_id), 3)
        src_path += '/nm-0'
        src_path += str(i)
        src_path += '/090/'
        dst_path = 'H:\gait_experiment\silh/'
        dst_path += utils.extend_digit_string(str(object_id), 3)
        dst_path += '-'
        dst_path += utils.extend_digit_string(str(i), 2)
        dst_path += '/'
        shutil.copytree(src_path, dst_path)

if __name__ == '__main__':
    for i in range(1, 60):
        copy_file(i)