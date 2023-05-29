#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.autograd.profiler as profiler

from ..cost import PrimitiveCollisionCost
from ...mpc.rollout.arm_reacher import ArmReacher

import rospy
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray

class ArmHuman(ArmReacher):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        super(ArmHuman, self).__init__(exp_params=exp_params,
                                         tensor_args=tensor_args,
                                         world_params=world_params)
        self.current_human_pose = None
        self.forecast_human_pose = None
        self.world_params = None
    

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        cost = super(ArmHuman, self).cost_fn(state_dict, action_batch, no_coll, horizon_cost)
        if self.current_human_pose is None:
            return cost
        # primitive_collision_cost = PrimitiveCollisionCost(world_params=self.world_params, robot_params=self.exp_params['robot_params'], tensor_args=self.tensor_args, **self.exp_params['cost']['primitive_collision'])
        # link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        # if(not no_coll):
        #     human_coll_cost = primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
        #     cost += human_coll_cost
        return cost
    
