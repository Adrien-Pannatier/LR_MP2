import numpy as np
import matplotlib.pyplot as plt
##################################################### 
# PLOTS
#####################################################
# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()

r = np.array([1, 2, 3, 4])  # Replace with actual values
dr = np.array([0.1, 0.2, 0.3, 0.4])  # Replace with actual values
theta = np.array([0.5, 1.0, 1.5, 2.0])  # Replace with actual values
dtheta = np.array([0.01, 0.02, 0.03, 0.04])  # Replace with actual values

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle('FL FR HL HR')

# Plot FL
axs[0, 0].plot(r, dr)
axs[0, 0].set_title('FL')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('z (m)')

# Plot FR
axs[0, 1].plot(theta, dtheta)
axs[0, 1].set_title('FR')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('z (m)')

# Plot HL
axs[1, 0].plot(r, theta)
axs[1, 0].set_title('HL')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('z (m)')

# Plot HR
axs[1, 1].plot(dr, dtheta)
axs[1, 1].set_title('HR')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('z (m)')

plt.tight_layout()
plt.show()

# time = np.array([0.5, 1.0, 1.5, 2.0])  # Replace with actual values
# r = np.array([0.01, 0.02, 0.03, 0.04])  # Replace with actual values
# theta = np.array([0.5, 1.0, 1.5, 2.0])  # Replace with actual values

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
fig.suptitle('CPG Amplitudes (r) and CPG Phases (θ)')

# Top subplot: CPG Amplitudes (r)
axs[0].plot(t, r, label='FR')
axs[0].plot(t, r, label='FL')
axs[0].plot(t, r, label='HR')
axs[0].plot(t, r, label='HL')
axs[0].set_title('CPG Amplitudes (r)')
axs[0].set_xlabel('t (s)')
axs[0].set_ylabel('r amplitude')
axs[0].legend()

# Bottom subplot: CPG Phases (θ)
axs[1].plot(t, theta, label='FR')
axs[1].plot(t, theta, label='FL')
axs[1].plot(t, theta, label='HR')
axs[1].plot(t, theta, label='HL')
axs[1].set_title('CPG Phases (θ)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('θ amplitude')
axs[1].legend()


plt.tight_layout()
plt.show()