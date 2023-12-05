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


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # Extract the desired values from q based on the loop index
    start_index = i * 3
    end_index = start_index + 3
    leg_q = q[start_index:end_index]
    leg_dq = dq[start_index:end_index]
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_des_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # returns joint angles for leg i
    # Add joint PD contribution to tau for leg i (Equation 4)
    
    tau += kp*(leg_des_q - leg_q) + kd*(-leg_dq) # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, foot_pos = env.robot.ComputeJacobianAndPosition(i) # [TODO] 
      # Get current foot velocity in leg frame (Equation 2)
      foot_vel = J @ leg_dq # [TODO] 
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (leg_xyz - foot_pos) + kdCartesian @ (-foot_vel)) # [TODO]

        # Save data for plotting
    if i == 0:
          foot_pos_plot[j, :, 0] = leg_xyz  # save desired position
          foot_pos_plot[j, :, 1] = foot_pos  # save actual position
          joint_angles_plot[j, :, 0] = leg_des_q  # save desired joint angles
          joint_angles_plot[j, :, 1] = leg_q  # save actual joint angles

    
    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
    cpg_states[i] = cpg.X
    if i > 0:
      cpg_velocities[i-1] = cpg_states[i] - cpg_states[i-1]
      cpg_velocities[i-1,1,:] = cpg_velocities[i-1,1,:] % (2*np.pi)
      cpg_velocities[i-1] = cpg_velocities[i-1] / cpg._dt

    if i>=int(TEST_STEPS/2):  #only save after convergence to steady gait
        if np.sin(cpg.X[1,0])>0:
            stance_array_theoretical[j-int(TEST_STEPS/2)]=0
        else:
            stance_array_theoretical[j-int(TEST_STEPS/2)]=1
        stance_array_simulation[j-int(TEST_STEPS/2)]=env.robot.GetContactInfo()[3][0]
        lin_x_vel[j-int(TEST_STEPS/2)] = env.robot.GetBaseLinearVelocity()[0]

    energy += np.sum(env.robot.GetMotorTorques() *                              \
                     env.robot.GetMotorVelocities()) * TIME_STEP



##################################################### 
# PLOTS
#####################################################

##################################################### 
# A plot of the CPG states (r, θ, ˙ r, ˙θ)
#####################################################
  fig, ax = plt.subplots(2, 2)
  if ADD_CARTESIAN_PD:
      fig.suptitle('{}: CPG states (with Cartesian PD)'.format(cpg.gait))
  else:
      fig.suptitle('{}: CPG states (without Cartesian PD)'.format(cpg.gait))

  ax[0,0].plot(t,cpg_states[:, 0, :])
  ax[0,0].set_xlabel('time')
  ax[0,0].set_ylabel('r')
  ax[0,0].legend(['FR', 'FL', 'RR', 'RL'])

  ax[0,1].plot(t,cpg_states[:, 1, :])
  ax[0,1].set_xlabel('time')
  ax[0,1].set_ylabel('theta')
  ax[0,1].legend(['FR', 'FL', 'RR', 'RL'])

  ax[1,0].plot(t[0:-1],cpg_velocities[:, 0, :])
  ax[1,0].set_xlabel('time')
  ax[1,0].set_ylabel('r_dot')
  ax[1,0].legend(['FR', 'FL', 'RR', 'RL'])

  ax[1,1].plot(t[0:-1],cpg_velocities[:, 1, :])
  ax[1,1].set_xlabel('time')
  ax[1,1].set_ylabel('theta_dot')
  ax[1,1].legend(['FR', 'FL', 'RR', 'RL'])

##################################################### 
# A plot comparing the desired foot position vs. actual foot position
#####################################################
fig, ax = plt.subplots(3, 1)
if ADD_CARTESIAN_PD:
    fig.suptitle('Desired/Actual Foot Position Over Time (with Cartesian PD)')
else:
    fig.suptitle('Desired/Actual Foot Position Over Time (without Cartesian PD)')

ax[0].plot(t, foot_pos_plot[:, 0, :])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('X Position')
ax[0].legend(['Desired Foot Position', 'Actual Foot Position'])

ax[1].plot(t, foot_pos_plot[:, 1, :])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Y Position')
ax[1].legend(['Desired Foot Position', 'Actual Foot Position'])

ax[2].plot(t, foot_pos_plot[:, 2, :])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Z Position')
ax[2].legend(['Desired Foot Position', 'Actual Foot Position'])

##################################################### 
# A plot comparing the desired joint angles vs. actual joint angles
#####################################################
fig, ax = plt.subplots(3, 1)
if ADD_CARTESIAN_PD:
    fig.suptitle('Desired/Actual Joint Angles Over Time (with Cartesian PD)')
else:
    fig.suptitle('Desired/Actual Joint Angles Over Time (without Cartesian PD)')

ax[0].plot(t, joint_angles_plot[:, 0, :])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Hip Angle (q0)')
ax[0].legend(['Desired Joint Angle', 'Actual Joint Angle'])

ax[1].plot(t, joint_angles_plot[:, 1, :])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Thigh Angle (q1)')
ax[1].legend(['Desired Joint Angle', 'Actual Joint Angle'])

ax[2].plot(t, joint_angles_plot[:, 2, :])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Calf Angle (q2)')
ax[2].legend(['Desired Joint Angle', 'Actual Joint Angle'])

plt.show()

##################################################### 
# A discussion on hyperparameters
#####################################################
  print("Avg velocity: " + str(env.robot.GetBasePosition()[0]/(TEST_STEPS*TIME_STEP)))
  print("Avg velocity (without convergence) " + str(np.mean(lin_x_vel)))
  print("CoT: " + str(energy / (sum(env.robot.GetTotalMassFromURDF()) * 9.81 * env.robot.GetBasePosition()[0])))

  print("Duty Factor Theoretical = Stance duration / Stride duration = " +str(np.sum(stance_array_theoretical)/int(TEST_STEPS/2)))
  print("Duty Factor Simulation = Stance duration / Stride duration = "  +str(np.sum(stance_array_simulation)/int(TEST_STEPS/2)))

  print("😌")

#####################################################
# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    run_cpg()
