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
from storm_kit.mpc.task.human_task import ReacherTask
np.set_printoptions(precision=2)
from storm_kit.util_file import get_gym_configs_path, join_path
import copy
import rospy
from quaternion import from_euler_angles, as_float_array
from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import yaml
import time
import math
from nav_msgs.msg import Odometry

MOCAP_OFFSETS = [1.22, -0.90, -0.85]

class Franka_Realworld():
    def __init__(self, world_file='handover.yml'):
        rospy.init_node('realworld', anonymous=True, disable_signals=True)
        # self.ee_sub = rospy.Subscriber('/ee_goal', PoseStamped, self.ee_callback, queue_size=1000)
        self.joint_state_pub = rospy.Publisher('/franka_motion_control/joint_command', JointState, queue_size=1)
        self.objects_pub = rospy.Publisher('/objects', MarkerArray, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.js = None
        self.world_file = world_file
        self.g_pos = None
        self.g_q = None
        self.pos = np.array([0.3, 0.0, 0.5])
        self.q = np.array([ 0.5, 0.5, 0.5, 0.5])
        self.objects_pub = rospy.Publisher('/objects', MarkerArray, queue_size=1)
        self.human_forecast_subscriber = rospy.Subscriber('/human_forecast', MarkerArray, self.human_forecast_callback, queue_size=1)
        self.ee_odom_pub = rospy.Publisher('/ee_odom', Odometry, queue_size=1)
        self.current_human_pose = None
        self.future_human_pose = None
        self.world_params = None
        # self.publish_objects()
    
    def human_forecast_callback(self, marker_array):
        if self.current_human_pose is None:
            self.current_human_pose = {}
            self.future_human_pose = {}
        leftmost_joint = None
        for marker in marker_array.markers:
            # print(marker.ns)
            if 'current' in marker.ns:
                # print(marker.ns)
                left_edge, right_edge = marker.ns.split('-')[1].split('_')
                p1, p2 = marker.points
                self.current_human_pose[left_edge] = np.array([p1.x+MOCAP_OFFSETS[0], p1.y+MOCAP_OFFSETS[1], p1.z+MOCAP_OFFSETS[2]])
                self.current_human_pose[right_edge] = np.array([p2.x+MOCAP_OFFSETS[0], p2.y+MOCAP_OFFSETS[1], p2.z+MOCAP_OFFSETS[2]])
            if 'forecast-25' in marker.ns:
                # print(marker.ns)
                left_edge, right_edge = marker.ns.split('-')[2].split('_')
                p1, p2 = marker.points
                self.future_human_pose[left_edge] = np.array([p1.x+MOCAP_OFFSETS[0], p1.y+MOCAP_OFFSETS[1], p1.z+MOCAP_OFFSETS[2]])
                self.future_human_pose[right_edge] = np.array([p2.x+MOCAP_OFFSETS[0], p2.y+MOCAP_OFFSETS[1], p2.z+MOCAP_OFFSETS[2]])
        for joint in self.current_human_pose:
            if leftmost_joint is None or leftmost_joint[0] > self.current_human_pose[joint][0]:
                leftmost_joint = self.current_human_pose[joint]
        for joint in self.future_human_pose:
            if leftmost_joint is None or leftmost_joint[0] > self.future_human_pose[joint][0]:
                leftmost_joint = self.future_human_pose[joint]
        # print(leftmost_joint[0])

        if leftmost_joint[0] > 1.15: return
        if self.g_pos is None:
            self.g_pos = leftmost_joint
            q = as_float_array(from_euler_angles(90.0 * 0.01745, 0.0 * 0.01745, 90 * 0.01745))
            self.g_q = np.array([q[1], q[2], q[3], q[0]])
            self.g_q = np.array([0.5, 0.5, 0.5, 0.5])
            # self.g_q = as_float_array(from_euler_angles(0 * 0.01745, 0.0 * 0.01745, 90 * 0.01745))
            # print(self.g_q)
        elif self.g_pos[0] > leftmost_joint[0]:
            self.g_pos = leftmost_joint
        # self.g_pos = leftmost_joint
        # self.g_pos[0] -= 0.3

    def create_world_params(self):
        world_params = {'world_model': 
                        {'coll_objs': 
                         {
                             'cube': {}
                             }
                            }
                        }
        
        bones = [("RShoulderBack", "RElbowOut"),
                 ("LElbowOut", "LShoulderBack"),
                 ("LElbowOut", "LWristOut"),
                 ("RWristOut", "RElbowOut")]
        for idx, (left_joint, right_joint) in enumerate(bones):
            cube_obj = {}
            # print(self.current_human_pose)
            A = self.current_human_pose[left_joint]
            B = self.current_human_pose[right_joint]
            cube_obj['dims'] = [0.1, np.linalg.norm(A-B), 0.1]
            position = list((A+B)/2)
            direction = B-A
            roll = np.arccos(direction[0]/np.linalg.norm(direction))
            pitch = np.arccos(direction[1]/np.linalg.norm(direction))
            yaw = np.arccos(direction[2]/np.linalg.norm(direction))
            
            q = as_float_array(from_euler_angles(yaw, pitch, roll))
            cube_obj['pose'] = position + list(q)

            world_params['world_model']['coll_objs']['cube'][f'cube{idx+1}'] = cube_obj
        
        TORSO = ("RShoulderBack", "LShoulderBack", "WaistLBack", "WaistRBack")
        A = self.current_human_pose[TORSO[0]]
        B = self.current_human_pose[TORSO[1]]
        C = self.current_human_pose[TORSO[2]]
        D = self.current_human_pose[TORSO[3]]
        cube_obj = {}
        cube_obj['dims'] = [np.linalg.norm(A-D), np.linalg.norm(A-B), 0.1]
        position = list((A+B+C+D)/4)
        direction = B-A
        roll = np.arccos(direction[0]/np.linalg.norm(direction))
        pitch = np.arccos(direction[1]/np.linalg.norm(direction))
        yaw = np.arccos(direction[2]/np.linalg.norm(direction))
        
        q = as_float_array(from_euler_angles(yaw, pitch, roll))
        cube_obj['pose'] = position + list(q)

        world_params['world_model']['coll_objs']['cube'][f'cube{idx+2}'] = cube_obj
        # torso = 
        return world_params

    def publish_objects(self):
        marker_array = MarkerArray()
        # world_yml = join_path(get_gym_configs_path(), self.world_file)
        # with open(world_yml, 'r') as file:
        #     world_params = yaml.load(file, Loader=yaml.FullLoader)
        obj_id = 0
        # if 'sphere' in world_params['world_model']['coll_objs']:
        #     spheres = world_params['world_model']['coll_objs']['sphere']
        #     for k, v in spheres.items():
        #         marker = Marker()
        #         marker.header.frame_id = 'panda_link0'
        #         marker.header.stamp = rospy.Time.now()
        #         marker.type = marker.SPHERE
        #         marker.ns = k
        #         marker.id = obj_id
        #         obj_id += 1
        #         marker.scale.x = v['radius']
        #         marker.scale.y = v['radius']
        #         marker.scale.z = v['radius']
        #         marker.pose.position.x = v['position'][0]
        #         marker.pose.position.y = v['position'][1]
        #         marker.pose.position.z = v['position'][2]
        #         marker.color.r = 1
        #         marker.color.a = 1
        #         marker_array.markers.append(marker)
        # if 'cube' in world_params['world_model']['coll_objs']:
        #     cubes = world_params['world_model']['coll_objs']['cube']
        #     for k, v in cubes.items():
        #         print("PRINTING TABLE")
        #         marker = Marker()
        #         marker.header.frame_id = 'panda_link0'
        #         marker.header.stamp = rospy.Time.now()
        #         marker.type = marker.CUBE
        #         marker.ns = k
        #         marker.id = obj_id
        #         obj_id += 1
        #         marker.scale.x = v['dims'][0]
        #         marker.scale.y = v['dims'][1]
        #         marker.scale.z = v['dims'][2]
        #         # print(v)
        #         marker.pose.position.x = v['pose'][0]
        #         marker.pose.position.y = v['pose'][1]
        #         marker.pose.position.z = v['pose'][2]
        #         marker.pose.orientation.w = 1.0
        #         marker.color.g = 1
        #         marker.color.a = 1
        #         marker_array.markers.append(marker)
        
        # marker_array = MarkerArray()
        # print(self.world_params)
        # if 'cube' in self.world_params['world_model']['coll_objs']:
        #     cubes = self.world_params['world_model']['coll_objs']['cube']
        #     for k, v in cubes.items():
        #         print(k)
        #         marker = Marker()
        #         marker.header.frame_id = 'panda_link0'
        #         marker.header.stamp = rospy.Time.now()
        #         marker.type = marker.CUBE
        #         marker.ns = k
        #         marker.id = obj_id+100
        #         obj_id += 1
        #         marker.scale.x = v['dims'][0]
        #         marker.scale.y = v['dims'][1]
        #         marker.scale.z = v['dims'][2]
        #         # print(v)
        #         marker.pose.position.x = v['pose'][0]
        #         marker.pose.position.y = v['pose'][1]
        #         marker.pose.position.z = v['pose'][2]
        #         marker.pose.orientation.x = v['pose'][3]
        #         marker.pose.orientation.y = v['pose'][4]
        #         marker.pose.orientation.z = v['pose'][5]
        #         marker.pose.orientation.w = v['pose'][6]
        #         marker.color.g = 1
        #         marker.color.a = 0.3
        #         marker_array.markers.append(marker)

        if self.g_pos is not None:
            marker = Marker()
            marker.header.frame_id = 'panda_link0'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.ns = "End_effector"
            marker.id = obj_id+200
            obj_id += 1
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            # print(v)
            marker.pose.position.x = self.g_pos[0]
            marker.pose.position.y = self.g_pos[1]
            marker.pose.position.z = self.g_pos[2]
            # marker.color.g = 1
            marker.color.r = 1
            marker.color.a = 1
            marker_array.markers.append(marker)
        # import pdb; pdb.set_trace()
        
        self.objects_pub.publish(marker_array)
        # import pdb; pdb.set_trace()
        
        # self.objects_pub.publish(marker_array)
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
    
    # while self.g_pos is None:
        #     time.sleep(0.05)
        #     print("No goal")
        # flag = ""
        # while flag == "":
        #     if self.current_human_pose is not None:
        #         self.world_params = self.create_world_params()
        #         self.publish_objects()
        #     flag = input("Start?")
    
    def publish_ee(self, pos, quat):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0' # i.e. '/odom'
        # msg.child_frame_id = self.child_frame_id # i.e. '/base_footprint'

        msg.pose.pose.position = Point(pos[0], pos[1], pos[2])
        msg.pose.pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])

        # p_cov = np.array([0.0]*36).reshape(6,6)
        self.ee_odom_pub.publish(msg)

    def mpc_robot_interactive(self, args):
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'handover.yml'

        device = torch.device('cuda', 0)
        tensor_args = {'device':device, 'dtype':torch.float32} 

        mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
        
        freq = 50
        sim_dt = 1.0/freq
        rate = rospy.Rate(freq)
        done = False
        
        start_time = rospy.get_time()
        t_step = 0
        try:
            while not done:
                
                print("DONE = ", done)
                ct_start = time.time()
                current_robot_state = self.get_current_state()
                if self.g_pos is not None:
                    g_pos = copy.deepcopy(self.g_pos)
                    g_pos[0] -= 0.2
                    print(g_pos, self.g_q)
                    mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=self.g_q)
                else:
                    mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                
                print("GOAL POSE = ", self.g_pos)
                print("GOAL QUAT = ", self.g_q)
                
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                t_step = rospy.get_time()-start_time
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                # print(len(ee_error))
                ct_end = time.time()
                print("TIMINGS =============", ct_start, ct_end, ct_end-ct_start)
                filtered_state_mpc = current_robot_state #mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
                # get current pose:
                e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                self.publish_ee(e_pos, e_quat)
                self.publish_command(q_des, qd_des, qdd_des)
                if self.current_human_pose is not None:
                    self.world_params = self.create_world_params()
                    self.publish_objects()
                rate.sleep()
        except KeyboardInterrupt:
            print('Closing')
            done = True
            if done: self.objects_pub.publish(self.get_clean())
            # break
        mpc_control.close()
        return 1 
    
    def get_clean(self):
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns = "delete"
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        return marker_array_msg
    
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