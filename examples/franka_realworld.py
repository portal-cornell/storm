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
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import json
import yaml
import time


class Franka_Realworld():
    def __init__(self, world_file='example.yml'):
        rospy.init_node('realworld', anonymous=True)
        self.ee_sub = rospy.Subscriber('/ee_goal', PoseStamped, self.ee_callback, queue_size=1000)
        self.joint_state_pub = rospy.Publisher('/franka_motion_control/joint_command', JointState, queue_size=1)
        self.objects_pub = rospy.Publisher('/objects', MarkerArray, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.js = None
        self.world_file = world_file
        self.g_pos = np.array([0.5, 0.2, 0.4])
        self.g_q = np.array([ 0, 0.7071068, 0.7071068, 0])
        self.publish_objects()
    
    def publish_objects(self):
        marker_array = MarkerArray()
        world_yml = join_path(get_gym_configs_path(), self.world_file)
        with open(world_yml, 'r') as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)
        obj_id = 0
        if 'sphere' in world_params['world_model']['coll_objs']:
            spheres = world_params['world_model']['coll_objs']['sphere']
            for k, v in spheres.items():
                marker = Marker()
                marker.header.frame_id = 'panda_link0'
                marker.header.stamp = rospy.Time.now()
                marker.type = marker.SPHERE
                marker.ns = k
                marker.id = obj_id
                obj_id += 1
                marker.scale.x = v['radius']
                marker.scale.y = v['radius']
                marker.scale.z = v['radius']
                marker.pose.position.x = v['position'][0]
                marker.pose.position.y = v['position'][1]
                marker.pose.position.z = v['position'][2]
                marker.color.r = 1
                marker.color.a = 1
                marker_array.markers.append(marker)
        if 'cube' in world_params['world_model']['coll_objs']:
            cubes = world_params['world_model']['coll_objs']['cube']
            for k, v in cubes.items():
                marker = Marker()
                marker.header.frame_id = 'panda_link0'
                marker.header.stamp = rospy.Time.now()
                marker.type = marker.CUBE
                marker.ns = k
                marker.id = obj_id
                obj_id += 1
                marker.scale.x = v['dims'][0]
                marker.scale.y = v['dims'][1]
                marker.scale.z = v['dims'][2]
                # print(v)
                marker.pose.position.x = v['pose'][0]
                marker.pose.position.y = v['pose'][1]
                marker.pose.position.z = v['pose'][2]
                marker.pose.orientation.w = 1.0
                marker.color.g = 1
                marker.color.a = 1
                marker_array.markers.append(marker)
        # import pdb; pdb.set_trace()
        
        self.objects_pub.publish(marker_array)
    def js_callback(self, js):
        self.js = js

    def ee_callback(self, ee):
        print("INNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
        # input()
        # print(ee_pose)
        self.g_pos = np.array([ee.pose.position.x, ee.pose.position.y, ee.pose.position.z])
        self.g_q = np.array([ee.pose.orientation.w, ee.pose.orientation.x, ee.pose.orientation.y, ee.pose.orientation.z])
    
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
        
        # franka_bl_state = np.array([-0.0, -0.782, -0.0, -2.35, -0.0, 1.56,0.585,
        #                             0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        ee_error = 10.0
        j = 0
        t_step = 0
        i = 0
        # mpc_control.update_params(goal_state=franka_bl_state)
        g_pos = np.array([0.5, 0.3, 0.5])
        g_q = np.array([ 0, 0.7071068, 0.7071068, 0])
        g_q = np.array([ 0.5, 0.5, 0.5, 0.5 ])
        
        
        freq = 100
        sim_dt = 1.0/freq
        rate = rospy.Rate(freq)
        done = False
        start_time = rospy.get_time()
        while(i > -100):
            if done: break
            try:
                # self.publish_objects()
                current_robot_state = self.get_current_state()
                ct_start = time.time()
                mpc_control.update_params(goal_ee_pos=self.g_pos, goal_ee_quat=self.g_q)
                ct_end = time.time()
                print(ct_start, ct_end, ct_end-ct_start)
                print("GOAL POSE = ", self.g_pos)
                print("GOAL QUAT = ", self.g_q)
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                # print(q_des, qd_des, qdd_des)
                filtered_state_mpc = current_robot_state #mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                print("EE POSE STATE = ", pose_state['ee_pos_seq'].cpu().numpy())
                print("EE QUAT STATE = ", pose_state['ee_quat_seq'].cpu().numpy())
                ee_error = mpc_control.get_current_error(current_robot_state)
                # print(len(ee_error))
                self.publish_command(q_des, qd_des, qdd_des)
                t_step = rospy.get_time()-start_time
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