import copy

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



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
from nav_msgs.msg import Odometry, Path

MOCAP_OFFSETS = [1.22, -0.90, -0.85]

def quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qw, qx, qy, qz])
import franka_gripper.msg

# Brings in the SimpleActionClient
import actionlib
def grasp_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (GraspAction) to the constructor.
    client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal = franka_gripper.msg.GraspGoal()
    goal.width = 0.04
    goal.epsilon.inner = 0.04
    goal.epsilon.outer = 0.04
    goal.speed = 0.1
    goal.force = 50.0

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A GraspResult

def open_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (GraspAction) to the constructor.
    client = actionlib.SimpleActionClient('/franka_gripper/move', franka_gripper.msg.MoveAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal = franka_gripper.msg.MoveGoal(width=0.1, speed=0.1)

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A GraspResult

class Franka_Realworld():
    def __init__(self, args, world_file='handover.yml'):
        rospy.init_node('realworld', anonymous=True, disable_signals=True)
        self.args = args
        # self.ee_sub = rospy.Subscriber('/ee_goal', PoseStamped, self.ee_callback, queue_size=1000)
        self.joint_state_pub = rospy.Publisher('/franka_motion_control/joint_command', JointState, queue_size=1)
        self.objects_pub = rospy.Publisher('/objects', MarkerArray, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.js = None
        self.world_file = world_file
        self.g_pos = None
        self.g_q = None
        self.pos = np.array([0.5, 0.0, 0.60])
        
        self.q = np.array([ 0.5, 0.5, 0.5, 0.5])
        self.objects_pub = rospy.Publisher('/objects', MarkerArray, queue_size=1)
        self.human_forecast_subscriber = rospy.Subscriber('/human_forecast', MarkerArray, self.human_forecast_callback, queue_size=1)
        self.path_pub = rospy.Publisher('/ee_path', Path, queue_size=1)
        self.ee_odom_pub = rospy.Publisher('/ee_odom', Odometry, queue_size=1)
        self.current_human_pose = None
        self.played = False
        self.human_pose = None
        self.world_params = None

    def human_forecast_callback(self, marker_array):
        if self.current_human_pose is None:
            self.current_human_pose = {}
            self.human_pose = {}
        leftmost_joint = None
        for marker in marker_array.markers:
            if self.args.forecast_type in marker.ns:
                left_edge, right_edge = marker.ns.split('-')[1].split('_')
                p1, p2 = marker.points
                self.current_human_pose[left_edge] = np.array([p1.x+MOCAP_OFFSETS[0], p1.y+MOCAP_OFFSETS[1], p1.z+MOCAP_OFFSETS[2]])
                self.current_human_pose[right_edge] = np.array([p2.x+MOCAP_OFFSETS[0], p2.y+MOCAP_OFFSETS[1], p2.z+MOCAP_OFFSETS[2]])
            
            if 'current' in marker.ns:
                left_edge, right_edge = marker.ns.split('-')[1].split('_')
                p1, p2 = marker.points
                self.human_pose[left_edge] = np.array([p1.x+MOCAP_OFFSETS[0], p1.y+MOCAP_OFFSETS[1], p1.z+MOCAP_OFFSETS[2]])
                self.human_pose[right_edge] = np.array([p2.x+MOCAP_OFFSETS[0], p2.y+MOCAP_OFFSETS[1], p2.z+MOCAP_OFFSETS[2]])
        joint = 'RWristOut'
        leftmost_joint = self.current_human_pose[joint]

        if self.g_pos is None:
            self.g_pos = leftmost_joint
            # self.g_pos = np.array([0.7, 0.2, 0.4])
            q = as_float_array(from_euler_angles(90.0 * 0.01745, 0.0 * 0.01745, 90 * 0.01745))
            self.g_q = np.array([q[1], q[2], q[3], q[0]])
            self.g_q = np.array([0.5, 0.5, 0.5, 0.5])
        # self.g_pos = np.array([0.6, 0.3, 0.4])
        # self.g_pos = leftmost_joint

    def publish_objects(self):
        marker_array = MarkerArray()
        obj_id = 0

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
            marker.pose.position.x = self.g_pos[0]
            marker.pose.position.y = self.g_pos[1]
            marker.pose.position.z = self.g_pos[2]
            marker.color.r = 1
            marker.color.a = 1
            marker_array.markers.append(marker)
        
        self.objects_pub.publish(marker_array)

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
    
    def publish_ee(self, pos, quat):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0' # i.e. '/odom'

        msg.pose.pose.position = Point(pos[0], pos[1], pos[2])
        msg.pose.pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])

        self.ee_odom_pub.publish(msg)

    def publish_path(self, trajs):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'panda_link0'
        # import pdb; pdb.set_trace()
        for idx in range(trajs.shape[1]):
            # for t in traj:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'panda_link0'
            pose.pose.position.x = torch.mean(trajs[:, idx, 0])
            pose.pose.position.y = torch.mean(trajs[:, idx, 1])
            pose.pose.position.z = torch.mean(trajs[:, 0, 2])
            path.poses.append(pose)
        self.path_pub.publish(path)

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
        # mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
        
        t_step = 0
        ee_path = []
        movement_start_time, movement_end_time = None, None
        start_time = rospy.get_time()

        s = "Can I get the salt from the table?"
        file = "file.mp3"
        result = open_client()
        # initialize tts, create mp3 and play
        from gtts import gTTS
        tts = gTTS(s, lang='en', tld="com")
        tts.save(file)
        # os.system("mpg123 " + file)
        import os
        to_grip = True
        # played = False
        try:
            while not done:
                if done: break
                # print("Hhs")
                ct_start = rospy.get_time()
                current_robot_state = self.get_current_state()
                # mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                
                # print(t_step)
                filtered_state_mpc = current_robot_state #mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                
                if ct_start-start_time > 10 and not self.played:
                    self.played=True
                    # t_step += 1.0
                    self.g_pos = np.array([0.8, -0.15, 0.4])
                    os.system("play " + file + " tempo 1.25")
                # if self.g_pos is not None:
                # print(played)

                if self.played and self.g_pos is not None:
                    if movement_end_time is not None and movement_start_time is not None:
                         mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                    else:
                        g_pos = copy.deepcopy(self.g_pos)
                        g_pos[0] -= 0.00
                        g_pos[1] -= 0.05
                        direction = g_pos-e_pos
                        if np.linalg.norm(direction) > 0.0001:
                            angle = np.arccos(direction[0]/np.linalg.norm(direction))
                            self.g_q = quaternion_from_euler(np.pi/2, 0, np.pi/2+angle)
                            # print("GOAL Q = ", self.g_q)
                            # print("GOAL POS = ", g_pos)
                        else:
                            self.g_q = e_quat
                        mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=self.g_q)
                else:
                    mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                
                # print("GOAL POSE = ", self.g_pos)
                # print("GOAL QUAT = ", self.g_q)
                
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                # t_step = rospy.get_time()-start_time
                t_step += sim_dt
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                # print(len(ee_error))
                ct_end = rospy.get_time()
                print("TIMINGS =============", ct_start, ct_end, ct_end-ct_start)
                filtered_state_mpc = current_robot_state #mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                print("EE POSE STATE = ", pose_state['ee_pos_seq'].cpu().numpy())
                # print("EE QUAT STATE = ", pose_state['ee_quat_seq'].cpu().numpy())
                # get current pose:
                e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                self.publish_ee(e_pos, e_quat)
                self.publish_command(q_des, qd_des, qdd_des)
                top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
                self.publish_path(top_trajs[:10])
                # if self.current_human_pose is not None:
                    # self.world_params = self.create_world_params()
                if self.played: self.publish_objects()
                
                
                if self.g_pos is not None and movement_end_time is None:
                    # g_pos = copy.deepcopy(self.human_pose['RWristOut'])
                    g_pos = copy.deepcopy(self.g_pos)
                    g_pos[0] -= 0.00
                    g_pos[1] -= 0.05
                    if movement_start_time is None: movement_start_time=rospy.get_time()
                    ee_path.append(e_pos)
                    displacement = 0
                    for i in range(1, len(ee_path)):
                        displacement += np.linalg.norm(ee_path[i]-ee_path[i-1])
                    

                    if np.linalg.norm(e_pos-g_pos) < 0.05:
                        if movement_end_time is None: 
                            movement_end_time = rospy.get_time()
                            mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                if movement_end_time is not None and movement_start_time is not None:
                    # self.g_pos = [self.human_pose['RWristOut'][0], self.human_pose['RWristOut'][1], self.human_pose['RWristOut'][2]]
                    g_pos = copy.deepcopy(self.g_pos)
                    self.publish_objects()
                    print("MOVEMENT TIME = ", movement_end_time-movement_start_time)
                    print("EE Displacemt = ", displacement)
                    # for i in range(100):
                    #     self.objects_pub.publish(self.get_clean())
                    done=True
                    mpc_control.update_params(goal_ee_pos=self.pos, goal_ee_quat=self.q)
                    t_step += 0.5
                    # if to_grip:
                    #     result = grasp_client()
                    #     ###update self.g_pos
                    #     to_grip = False
                    break
                rate.sleep()
                    
        except KeyboardInterrupt:
            print('Closing')
            done = True
            # break
        if done: self.objects_pub.publish(self.get_clean())
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
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    parser.add_argument('--forecast_type',type=str, default='current', help='Forecast Type')
    args = parser.parse_args()
    
    fr = Franka_Realworld(args)
    
    fr.mpc_robot_interactive(args)

