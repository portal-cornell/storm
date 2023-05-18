import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped

rospy.init_node('realworld', anonymous=True)
ee_pub = rospy.Publisher('/ee_goal', PoseStamped, queue_size=1)

import math
def quaternion_to_euler(q):
    qw, qx, qy, qz = q
    yaw = math.atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz);
    pitch = math.asin(-2.0*(qx*qz - qw*qy));
    roll = math.atan2(2.0*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
    return roll, pitch, yaw

def euler_to_quaternion(euler):
    #convert degrees to rad
    euler = np.array(euler)*np.pi/180.0
    roll, pitch, yaw = euler
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def create_msg(pose, euler):
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "panda_link0"
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.pose.position.x = pose[0]
    pose_msg.pose.position.y = pose[1]
    pose_msg.pose.position.z = pose[2]
    quat = euler_to_quaternion(euler)
    print("QUAT = ", quat)
    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]
    return pose_msg

# rate = rospy.Rate(10)
while not rospy.is_shutdown():
    print(quaternion_to_euler([0, 0.707, 0.707, 0]))
    pose = list(map(float, input("Input Pose seperated by spaces = ").split()))
    print(pose)
    euler = list(map(float, input("Input Orientation seperated by spaces = ").split()))
    ee_pub.publish(create_msg(pose, euler))
