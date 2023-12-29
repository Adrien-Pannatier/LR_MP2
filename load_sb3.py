# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '120123143305'
log_dir = interm_dir + 'CARTPD_FLAGRUN_TORQUE_V6'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CARTESIAN_PD",
               "task_env":"FLAGRUN",
               "observation_space_mode": "FLAGRUN_Y_OBS"}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0
    dt = 0.01
    # penalize pitch and roll
    # dt values according to CPG-RL Paper
    vel_z_pen = 0.8 * dt * env.envs[0].env.robot.GetBaseLinearVelocity()[2]**2 #3
    roll_pen = 0.6 * dt * np.abs(env.envs[0].env.robot.GetBaseAngularVelocity()[0]**2) #1
    pitch_pen = 0.6 * dt * np.abs(env.envs[0].env.robot.GetBaseAngularVelocity()[1]**2) #2

    curr_dist_to_goal, angle = env.envs[0].env.get_distance_and_angle_to_goal()
    # lin_vel_x, lin_vel_y = self.get_robot_lin_vel()
    # minimize distance to goal (we want to move towards the goal)
    dist_reward = 100 * dt * (env.envs[0].env._prev_pos_to_goal - curr_dist_to_goal) #7
    angle_pen = 0.01 * dt * np.abs(angle) #8
    # minimize energy 
    energy = 0 

    for tau,vel in zip(env.envs[0].env._dt_motor_torques,env.envs[0].env._dt_motor_velocities):
      energy += np.abs(np.dot(tau,vel)) * env.envs[0].env._time_step 

    energy_pen = 0.002 * dt * energy #9

    reward = dist_reward \
            - angle_pen \
            - energy_pen \
            - vel_z_pen \
            - roll_pen \
            - pitch_pen 
    
    print(env.envs[0].env.robot.GetBaseOrientationRollPitchYaw())
    # motor_angle_tracking = 0.0
    # for i in range (4):
    #   motor_angle_tracking += -0.01 * (env.envs[0].env.robot.GetMotorAngles()[i*3]) ** 2 \
      
    # print('motor_angle_tracking %: ', motor_angle_tracking)
    # print(env.envs[0].env.robot.GetBaseAngularVelocity()[:2]**2)
    # print('reward', reward)
    # print('roll', env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()[0])
    # print('pitch', env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()[1])
    # print('delta_dist', env.envs[0].env._prev_pos_to_goal - curr_dist_to_goal)
    # print('angle', angle)
    # print('roll_pen %: ', roll_pen)
    # print('pitch_pen %: ', pitch_pen)
    # print('vel_z_pen %: ', vel_z_pen)
    # # print('z_height_tracking %: ', z_height_pen*percentage)
    # print('dist_reward %: ', dist_reward)
    # print('angle_pen %: ', angle_pen)
    # print('energy_pen %: ', energy_pen)
    # tot = dist_reward \
    #     + energy_pen \
    #     + angle_pen \
    #     + vel_z_pen \
    #     + roll_pen \
    #     + pitch_pen 
    # print('fct', env.envs[0].env.get_distance_and_angle_to_goal())
    # percentage = 100/tot
    # print('total (=100%): ', tot)
    # print('roll_pen %: ', roll_pen*percentage)
    # print('pitch_pen %: ', pitch_pen*percentage)
    # print('vel_z_pen %: ', vel_z_pen*percentage)
    # # print('z_height_tracking %: ', z_height_pen*percentage)
    # print('dist_reward %: ', dist_reward*percentage)
    # print('angle_pen %: ', angle_pen*percentage)
    # print('energy_pen %: ', energy_pen*percentage)
    # print('vel_x_tracking %: ', vel_x_tracking*percentage)
    # print('vel_y_tracking %: ', vel_y_tracking*percentage)
    # print('angle_goal_tracking %: ', angle_goal_tracking*percentage)
    # print('base height:', env.envs[0].env.robot.GetBasePosition()[2])
    print('----------------------------------------')
        
    
# [TODO] make plots: