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

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv
import numpy as np
from matplotlib import pyplot as plt


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
FOOT_Y = 0.0838 # this is the hip length 
SIDESIGN = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=False,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # noGt using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    record_video=False
                    )

labels_positions = np.array(["x", "y", "z"])
labels_joint = np.array(["hip", "thigh", "calf"])
save_plots = False
# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT")

TEST_STEPS = int(5 / (TIME_STEP))

t = np.arange(TEST_STEPS)*TIME_STEP

start_plot = 10
end_plot = 2000

des_leg_pos = np.zeros((3, TEST_STEPS))
act_leg_pos = np.zeros((3, TEST_STEPS))

des_joint_angles = np.zeros((3, TEST_STEPS))
act_joint_angles = np.zeros((3, TEST_STEPS))

r = np.zeros((4, TEST_STEPS))
r_dot = np.zeros((4, TEST_STEPS))
theta = np.zeros((4, TEST_STEPS))
theta_dot = np.zeros((4, TEST_STEPS))

kp_in = 0
kd_in = 0
kp_cat_in = 0
kd_cat_in = 0

kp_in = 285
kd_in = 2.2
kp_cat_in = 450
kd_cat_in = 15

class Hyperparameters:
   def __init__(self, kp=np.array([kp_in,kp_in,kp_in]), kd=np.array([kd_in, kd_in, kd_in]), kp_cart=np.diag([kp_cat_in]*3), kd_cart=np.diag([kd_cat_in]*3)):
      self.kp = kp
      self.kd = kd
      self.kp_cart = kp_cart
      self.kd_cart = kd_cart

def get_robot_lin_vel():
  yaw = env.robot.GetBaseOrientationRollPitchYaw()[2]
  lin_vel_x, lin_vel_y, null_vel_z = env.robot.GetBaseLinearVelocity() * [np.cos(yaw), np.sin(yaw), 0]
  return lin_vel_x, lin_vel_y

def compute_cost_of_transport(_dt_motor_torques, _dt_motor_velocities, positions, TIME_STEP):
  energy = 0 
  # compute power
  for tau,vel in zip(_dt_motor_torques, _dt_motor_velocities):
    energy += np.abs(np.dot(tau,vel)) * TIME_STEP

  # compute distance traveled
  x1, y1, z1 = positions[0]
  x2, y2, z2 = positions[-1]
  distance_traveled = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

  # compute cost of transport
  cost_of_transport = energy / (env.robot.GetBaseMassFromURDF()[0] * 9.81 * distance_traveled)
  return cost_of_transport
  

def run_cpg(hyp = Hyperparameters(), do_plot = False, return_wanted = None, omega_stance = 1*np.pi*2, omega_swing = 3*np.pi*2):

  ############## Sample Gains
  # joint PD gains
  kp=hyp.kp            # HYPERPARAMETERS
  kd=hyp.kd
  # Cartesian PD gains
  kpCartesian = hyp.kp_cart
  kdCartesian = hyp.kd_cart

  cpg._omega_stance = omega_stance
  cpg._omega_swing = omega_swing

  cpg._mu = 0.1
  # cpg._ground_clearance = 0.1
  # cpg._ground_penetration = 0.001
  # cpg._des_step_len = 0.1

  return_wanted = "robot_vel"

  # data to fill
  linear_vel = []

  # for energy calculation
  _dt_motor_torques = []
  _dt_motor_velocities = []
  positions = []



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
          # print('Cart_contribution: ', J.T @ (kpCartesian @ (leg_xyz - foot_pos) + kdCartesian @ (-foot_vel)) )
          act_leg_pos[:, j] = foot_pos
        # Get current foot velocity in leg frame (Equation 2)
        foot_vel = J @ leg_dq 
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += np.matmul(J.T, np.matmul(kpCartesian, (leg_xyz - foot_pos)) + np.matmul(kdCartesian, (-foot_vel)))

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

      _dt_motor_torques.append(env.robot.GetMotorTorques())
      _dt_motor_velocities.append(env.robot.GetMotorVelocities())
      positions.append(env.robot.GetBasePosition())

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 
    if return_wanted == "vel":
      linear_vel.append(env.robot.GetBaseLinearVelocity())
    elif return_wanted == "robot_vel":
      linear_vel.append(get_robot_lin_vel())

  ###############################################################################################
  ####################################----RETURNS----############################################
  ###############################################################################################
  print("\n")
  if return_wanted == "vel":
     # average of linear velocity in x in the 20 last steps
    print(f"linear vel {np.mean(np.array(linear_vel[-20:])[:,0])}")

  elif return_wanted == "robot_vel":
    # average of linear velocity in x in the 20 last steps
    print(f"robot linear vel {np.mean(np.array(linear_vel[-20:])[:,0])}")

  print(f"stance time {2*np.pi/cpg._omega_stance}")
  print(f"swing time {2*np.pi/cpg._omega_swing}")
  print(f"duty cycle ratio {cpg._omega_stance / (cpg._omega_stance + cpg._omega_swing)}")
  print(f"step duration {((2*np.pi)/cpg._omega_stance + (2*np.pi)/cpg._omega_swing) / 2}")

  # compute average cost of transport
  average_cost_of_transport = np.mean(compute_cost_of_transport(_dt_motor_torques, _dt_motor_velocities, positions, TIME_STEP))
  print(f"average cost of transport {average_cost_of_transport}")
  
  ###############################################################################################
  ####################################-----PLOTS-----############################################
  ###############################################################################################

  # Plot CPG States 3.1.1

  if do_plot == True:

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('CPG States for PACE gait')

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
    # plt.legend()
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
          ax.set_ylim([-0.2, 0.0])
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
    # run_cpg(omega_stance=1/2*np.pi*2, omega_swing=3*np.pi*2)
    run_cpg()