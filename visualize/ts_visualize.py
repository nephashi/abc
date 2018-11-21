'''
可视化图像
'''
import matplotlib.pyplot as plt

class TSVisualizer(object):

    def draw(self, series, subject, feature_name):
        '''
        画一个人的一个特征从六个视频里提取的的图
        :param series: 用load.load_ts读出来
        :param subject: 人的id
        :param feature_name:
        rt = {
            'l_thigh_angle': l_thigh_angle,
            'r_thigh_angle': r_thigh_angle,
            'l_shin_angle': l_shin_angle,
            'r_shin_angle': r_shin_angle,
            'neck_angle': neck_angle,
            'hw_ratio': hw_ratio,
            'area': area,
            'width': width,
            'y_centre': y_centre
        }
        :return:
        '''
        ylim_dic = {
            'area': (0, 5000),
            'neck_angle': (-1, 1),
            'l_thigh_angle': (-1.6, 1.6),
            'r_thigh_angle': (-1.6, 1.6),
            'r_shin_angle': (-1.6, 1.6),
            'width': (0, 100),
            'height': (0, 160)
        }
        for i in range(6):
            features = series[subject][i]['feature']
            ts = features[feature_name]
            plt.title(str(subject) + '_' + feature_name)
            plt.ylim(ylim_dic[feature_name][0], ylim_dic[feature_name][1])
            plt.xlim(0, 80)
            plt.plot(range(len(ts)), ts)
        plt.show()

def draw():
    from load import load_ts
    ts = load_ts('H:/gait_experiment/TS', [1, 3, 8, 15, 105])
    tsv = TSVisualizer()
    # tsv.draw(ts, 105, 'r_thigh_angle')
    tsv.draw(ts, 15, 'width')

if __name__ == '__main__':
    draw()