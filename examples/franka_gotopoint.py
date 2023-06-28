import copy
# from isaacgym import gymapi
# from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#



import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import time
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)
from storm_kit.util_file import get_gym_configs_path, join_path
import copy
import rospy
from quaternion import from_euler_angles, as_float_array
from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import yaml
import time
import math

class Franka_Realworld():
    def __init__(self, world_file='example.yml'):
        rospy.init_node('realworld', anonymous=True)
        self.joint_state_pub = rospy.Publisher('/franka_motion_control/joint_command', JointState, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.js = None
        self.world_file = world_file
        self.g_pos = np.array([0.7, 0.2, 0.5])
        self.g_q = np.array([ 0, 0.7071068, 0.7071068, 0])
        self.g_q = np.array([ 0.5, 0.5, 0.5, 0.5])
        self.current_human_pose = None
        self.world_params = None

    def js_callback(self, js):
        self.js = js
    
    def get_current_state(self):
        return {'name': self.js.name[:-2], 'position': np.array(self.js.position[:-2]), \
                'velocity': np.array(self.js.velocity[:-2]), 'acceleration': np.array(self.js.effort[:-2])}

    def publish_command(self, q_des, qd_des, qdd_des):
        pub_command = JointState()
        pub_command.header.stamp = rospy.Time.now()
        pub_command.name = [f'panda_joint{idx}' for idx in range(7)]
        pub_command.position = q_des
        pub_command.velocity = qd_des
        pub_command.effort = qdd_des
        self.joint_state_pub.publish(pub_command)

    
    
    def mpc_robot_interactive(self, args):
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'example.yml'

        device = torch.device('cuda', 0)
        print(device)
        tensor_args = {'device':device, 'dtype':torch.float32} 

        mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
        
        freq = 50
        sim_dt = 1.0/freq
        rate = rospy.Rate(freq)
        done = False
        t_step = 0
        start_time = rospy.get_time()
        
        while True:
            if done: break
            try:
                ct_start = time.time()
                # self.publish_objects()
                
                
                current_robot_state = self.get_current_state()
                
                mpc_control.update_params(goal_ee_pos=self.g_pos, goal_ee_quat=self.g_q)
                
                print("GOAL POSE = ", self.g_pos)
                print("GOAL QUAT = ", self.g_q)
                
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                # print(q_des, qd_des, qdd_des)
                # filtered_state_mpc = current_robot_state #mpc_control.current_state
                # curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                # curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                # pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                # print("EE POSE STATE = ", pose_state['ee_pos_seq'].cpu().numpy())
                # print("EE QUAT STATE = ", pose_state['ee_quat_seq'].cpu().numpy())
                # ee_error = mpc_control.get_current_error(current_robot_state)
                # print(len(ee_error))
                ct_end = time.time()
                print("TIMINGS =============", ct_start, ct_end, ct_end-ct_start)
                self.publish_command(q_des, qd_des, qdd_des)
                t_step += sim_dt
                rate.sleep()
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        mpc_control.close()
        return 1 
    
if __name__ == '__main__': 
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    fr = Franka_Realworld()
    
    fr.mpc_robot_interactive(args)


# self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)