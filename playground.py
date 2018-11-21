import matplotlib.pyplot as plt
import json

# for i in range(1, 7):
#     with open('H:/gait_experiment/fcn_TS/001_0' + str(i) + '.json') as f:
#         jsonstr = f.read()
#         dic = json.loads(jsonstr)
#         data = dic['feature']['r_shin_angle']
#         plt.plot(range(len(data)), data)
# plt.ylim(-1,1)
# plt.xlim(0, 80)
# plt.show()
import utils
import os

for i in range(1, 125):
    for j in range(1, 7):
        path = 'H:\gait_experiment\silh/' + utils.extend_digit_string(str(i), 3) + '-' + utils.extend_digit_string(str(j), 2)
        if not os.path.exists(path):
            os.mkdir(path)
