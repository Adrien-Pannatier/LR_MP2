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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv
import numpy as np
from matplotlib import pyplot as plt


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
FOOT_Y = 0.0838 # this is the hip length 
SIDESIGN = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

labels_positions = np.array(["x", "y", "z"])
labels_joint = np.array(["hip", "thigh", "calf"])
save_plots = False
# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait='TROT')

TEST_STEPS = int(5 / (TIME_STEP))

t = np.arange(TEST_STEPS)*TIME_STEP

start_plot = 3000
end_plot = 4000

des_leg_pos = np.zeros((3, TEST_STEPS))
act_leg_pos = np.zeros((3, TEST_STEPS))

des_joint_angles = np.zeros((3, TEST_STEPS))
act_joint_angles = np.zeros((3, TEST_STEPS))

r = np.zeros((4, TEST_STEPS))
r_dot = np.zeros((4, TEST_STEPS))
theta = np.zeros((4, TEST_STEPS))
theta_dot = np.zeros((4, TEST_STEPS))
####
# Optimal kp and kd depending on situations:
# only Cartesian_PD: kp_cart = np.diag([450]*3), kd_cart = np.diag([16]*3)
# both PDs: kp = np.array([385,385,385]), kd = np.array([2.6, 2.6, 2.6]), kp_cart = np.diag([260]*3), kd_cart = np.diag([15]*3)
# only Joint_PD: kp = np.array([385,385,385]), kd = np.array([2.6, 2.6, 2.6])

class Hyperparameters:
   def __init__(self, kp=np.array([385, 385, 385]), kd=np.array([2.6, 2.6, 2.6]), kp_cart=np.diag([260]*3), kd_cart=np.diag([15]*3)) -> None:
      self.kp = kp
      self.kd = kd
      self.kp_cart = kp_cart
      self.kd_cart = kd_cart

def run_cpg(hyp = Hyperparameters(), do_plot = True, return_wanted = None):

############## Sample Gains
# joint PD gains
  kp=hyp.kp            # HYPERPARAMETERS
  kd=hyp.kd
  # Cartesian PD gains
  kpCartesian = hyp.kp_cart
  kdCartesian = hyp.kd_cart

  # data to fill
  linear_vel = []

  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12) 
    # get desired foot positions from CPG 
    xs,zs = cpg.update()
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    # loop through desired foot positions and calculate torques
    # get values for plot
    r[:, j] = cpg.get_r()
    r_dot[:, j] = cpg.get_dr()
    theta[:, j] = cpg.get_theta()
    theta_dot[:, j] = cpg.get_dtheta()

    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # Extract the desired values from q based on the loop index
      start_index = i * 3
      end_index = start_index + 3
      leg_q = q[start_index:end_index]
      leg_dq = dq[start_index:end_index]
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],SIDESIGN[i] * FOOT_Y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_des_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # returns joint angles for leg i
      # Add joint PD contribution to tau for leg i (Equation 4) 
      tau += kp*(leg_des_q - leg_q) + kd*(-leg_dq) 

      # get values for plots
      if i == 0:
          des_leg_pos[:, j] = leg_xyz
          des_joint_angles[:, j] = leg_des_q
          act_joint_angles[:, j] = leg_q
          J, foot_pos = env.robot.ComputeJacobianAndPosition(i)
          act_leg_pos[:, j] = foot_pos

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, foot_pos = env.robot.ComputeJacobianAndPosition(i) 
        foot_vel = J @ leg_dq 
        # get values for plots
        if i == 0:
          act_leg_pos[:, j] = foot_pos
        # Get current foot velocity in leg frame (Equation 2)
        foot_vel = J @ leg_dq 
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += np.matmul(J.T, np.matmul(kpCartesian, (leg_xyz - foot_pos)) + np.matmul(kdCartesian, (-foot_vel)))

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 
    print(env.robot.GetBasePosition()[2])
    linear_vel.append(env.robot.GetBaseLinearVelocity())

  ###############################################################################################
  ####################################----RETURNS----############################################
  ###############################################################################################
  
  if return_wanted == "maxvel":
     return linear_vel
  else:
     pass

  ###############################################################################################
  ####################################-----PLOTS-----############################################
  ###############################################################################################

  # Plot CPG States 3.1.1
  if do_plot == True:

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('CPG States for trot gait')

    # Plot FL
    axs[0, 0].plot(t[start_plot:end_plot], r[0, start_plot:end_plot], label='r', color='royalblue')
    axs[0, 0].plot(t[start_plot:end_plot], r_dot[0,start_plot:end_plot], linestyle='dashed', label='dr', color='royalblue')
    axs[0, 0].plot(t[start_plot:end_plot], theta[0,start_plot:end_plot], label='theta', color='firebrick')
    axs[0, 0].plot(t[start_plot:end_plot], theta_dot[0,start_plot:end_plot], linestyle='dashed', label='dtheta', color='firebrick')
    axs[0, 0].set_title('FR')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')

    # Plot FR
    axs[0, 1].plot(t[start_plot:end_plot], r[1, start_plot:end_plot], label='r', color='royalblue')
    axs[0, 1].plot(t[start_plot:end_plot], r_dot[1,start_plot:end_plot], linestyle='dashed', label='dr', color='royalblue')
    axs[0, 1].plot(t[start_plot:end_plot], theta[1,start_plot:end_plot], label='theta', color='firebrick')
    axs[0, 1].plot(t[start_plot:end_plot], theta_dot[1,start_plot:end_plot], linestyle='dashed', label='dtheta', color='firebrick')
    axs[0, 1].set_title('FL')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Amplitude')

    # Plot HL
    axs[1, 0].plot(t[start_plot:end_plot], r[2, start_plot:end_plot], label='r', color='royalblue')
    axs[1, 0].plot(t[start_plot:end_plot], r_dot[2,start_plot:end_plot], linestyle='dashed', label='dr', color='royalblue')
    axs[1, 0].plot(t[start_plot:end_plot], theta[2,start_plot:end_plot], label='theta', color='firebrick')
    axs[1, 0].plot(t[start_plot:end_plot], theta_dot[2,start_plot:end_plot], linestyle='dashed', label='dtheta', color='firebrick')
    axs[1, 0].set_title('RR')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Amplitude')

    # Plot HR
    axs[1, 1].plot(t[start_plot:end_plot], r[3, start_plot:end_plot], color='royalblue')
    axs[1, 1].plot(t[start_plot:end_plot], r_dot[3,start_plot:end_plot], linestyle='dashed', color='royalblue')
    axs[1, 1].plot(t[start_plot:end_plot], theta[3,start_plot:end_plot], color='firebrick')
    axs[1, 1].plot(t[start_plot:end_plot], theta_dot[3,start_plot:end_plot], linestyle='dashed', color='firebrick')
    axs[1, 1].set_title('RL')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Amplitude')

    fig.subplots_adjust(left=0.05, bottom=0.084, right=0.898, top=0.88, wspace = 0.171, hspace = 0.363)
    plt.legend()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    ##### Value comparison between desired and real##################################

    fig = plt.figure(figsize =(10, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    labels = np.array(["time [s]", "Position [m]"])
    ax1 = subfigs[0].subplots(3, 1, sharex=True)
    subfigs[0].suptitle("Foot positions")
    for i, ax in enumerate(ax1):
        ax.plot(t[start_plot:end_plot], des_leg_pos[i, start_plot:end_plot], label = "desired")
        ax.plot(t[start_plot:end_plot], act_leg_pos[i, start_plot:end_plot], label = "actual", color = "r")
        ax.set_title('Axis : ' + labels_positions[i])
        ax.grid(True)
        if i == 0:
          ax.set_ylim([-0.1, 0.1])
        if i == 1:
          ax.set_ylabel(labels[1])
          # ax.set_ylim([-0.2, 0.0])
        # ax.legend()
    subfigs[0].subplots_adjust(left = 0.2, bottom = 0.09, right = 0.96, top = 0.9)
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "angle [rad]"])
    ax2 = subfigs[1].subplots(3, sharex=True)
    subfigs[1].suptitle("Joint positions")
    for i, ax in enumerate(ax2):
        ax.plot(t[start_plot:end_plot], des_joint_angles[i, start_plot:end_plot], label="desired")
        ax.plot(t[start_plot:end_plot], act_joint_angles[i, start_plot:end_plot], label="actual", color="r")
        ax.set_title('Joint: ' + labels_joint[i])
        ax.grid(True)
        if i == 0:
          ax.set_ylim([-0.1, 0.1])
        if i == 1:
          ax.set_ylabel(labels[1])
        # ax.legend()
    plt.xlabel(labels[0], loc= 'center')
    if save_plots:
        plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")

    # plt.legend()
    handles, labels = ax1[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

if __name__ == '__main__':
    run_cpg()