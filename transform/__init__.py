'''
这个包主要包含把步态轮廓图像序列转化为步态特征序列的功能
'''
from .silh_pose_estimate import SilhPoseEstimator
from .fcn_pose_estimate import FCNPoseEstimator

SILH_POSE_ESTIMATOR = SilhPoseEstimator()
#FCN_POSE_ESTIMATOR = FCNPoseEstimator('model/MPII+LSP.ckpt')