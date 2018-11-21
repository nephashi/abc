class counter(object):
    count = 0

    def visit(self):
        self.count += 1
        return self.count

    def increase(self):
        self.count += 1

    def reset(self):
        self.count = 0

COUNTER = counter()

def extend_digit_string(str, num_digit):
    '''
    扩展整数字符串，前面补零到指定位数
    :param str:
    :param num_digit:
    :return:
    '''
    if (len(str) >= num_digit):
        return str
    pre = '0' * (num_digit - len(str))
    return pre + str

def get_video_file_name(object_id, video_id):
    str_object = extend_digit_string(str(object_id), 3)
    str_video = extend_digit_string(str(video_id), 2)
    rt = str_object + '-nm-' + str_video + '-090.avi'
    return rt

def get_output_json_file_name(object_id, video_id):
    str_object = extend_digit_string(str(object_id), 3)
    str_video = extend_digit_string(str(video_id), 2)
    rt = str_object + '_' + str_video + '.json'
    return rt