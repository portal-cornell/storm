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
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)

import copy
import rospy
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker

from std_msgs.msg import Float32
from geometry_msgs.msg import WrenchStamped



class Franka_Realworld():
    def __init__(self):
        rospy.init_node('realworld', anonymous=True)
        self.joint_state_pub = rospy.Publisher('/franka_motion_control/joint_command', JointState, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.js = None
        # input()
        # self.publish()
        
    def js_callback(self, js):
        # print(type(data))
        # print(js.position)
        self.js = js
        
        
    def publish(self):
        js_copy = copy.deepcopy(self.js)
        pos = js_copy.position
        pos_t = tuple([p+0.000 for p in pos])
        print(pos_t)
        js_copy.position = pos_t
        self.joint_state_pub.publish(js_copy)
    
    def get_current_state(self):
        # import pdb; pdb.set_trace()

        return {'name': self.js.name[:], 'position': np.array(self.js.position[:]), \
                'velocity': np.array(self.js.velocity[:]), 'acceleration': np.array(self.js.effort[:])}

    def publish_command(self, q_des, qd_des, qdd_des):
        js_copy = copy.deepcopy(self.js)
        pos = js_copy.position
        # import pdb; pdb.set_trace()
        # blank_list = [0, 0.0]
        # blank_list.extend(list(q_des))
        # q_des = blank_list
        # blank_list = [0, 0.0]
        # blank_list.extend(list(qd_des))
        # qd_des = blank_list
        # blank_list = [0, 0.0]
        # blank_list.extend(list(qdd_des))
        # qdd_des = blank_list
        # print(q_des)
        # input()
        # pos_t = tuple([p+0.000 for p in pos])
        # print(pos_t)
        # import pdb; pdb.set_trace()
        js_copy.position = tuple(q_des)
        js_copy.velocity = tuple(qd_des)
        js_copy.effort = tuple(qdd_des)
        # print(js_copy)
        # input("SENDING")
        self.joint_state_pub.publish(js_copy)

    def mpc_robot_interactive(self, args):
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'collision_primitives_3d.yml'

        device = torch.device('cuda', 0)
        tensor_args = {'device':device, 'dtype':torch.float32} 

        mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        start_qdd = torch.zeros(n_dof, **tensor_args)

        # update goal:
        exp_params = mpc_control.exp_params
        
        current_state = copy.deepcopy(self.js)
        ee_list = []
        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

        franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0,
                                    0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        joint_pose = [-0.09515199, -1.27960808, -0.17832713, -2.67556323, -0.05136292,  1.94429217, 0.57877946]
        # x_des_list = [franka_bl_state]
        
        franka_bl_state = np.array([-0.095, -1.28, -0.18, -2.7, -0.05, 1.94,0.58,
                                    0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        ee_error = 10.0
        j = 0
        t_step = 0
        i = 0
        mpc_control.update_params(goal_state=franka_bl_state)
        sim_dt = mpc_control.exp_params['control_dt']

        while(i > -100):
            try:
                current_robot_state = self.get_current_state()
                # input()
                # print(current_robot_state)
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                print(q_des, qd_des, qdd_des)
                input()
                self.publish_command(q_des, qd_des, qdd_des)
                # break
                i+=1
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