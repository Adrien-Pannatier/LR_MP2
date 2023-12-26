# hyper parameter research
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from run_cpg import run_cpg, Hyperparameters
from matplotlib.colors import LogNorm

def study_PD_maxvel(rg=10, incr=3):
    maxvel = np.zeros(rg**2)
    kp_tab = np.zeros(rg**2)
    kd_tab = np.zeros(rg**2)
    cmpt = 0
    for i in range(rg):
        for j in range(rg):

            print(f"iteration {cmpt+1}")
            kp_in = np.round(100 + i*incr*10,2)
            kd_in = np.round(0.1 + j*incr/10,2)
            kp_tab[cmpt] = (kp_in)
            kd_tab[cmpt] = (kd_in)
            kp = np.array([kp_in,kp_in,kp_in])   
            kd = np.array([kd_in, kd_in, kd_in])
            hyperparam = Hyperparameters(kp=kp, kd=kd)

            print(f"testing with kd_in = {kd_in}, kp_in = {kp_in}")
            val = np.array(run_cpg(hyp=hyperparam, do_plot=False, return_wanted="robot_vel"))
            # print(f"val = {type(val)} : {np.max(val[:,0])}")
            maxvel[cmpt] = np.max(val[:,0])
            print(f"max vel = {maxvel[cmpt]}")

            cmpt += 1
    results = np.array([kd_tab, kp_tab, maxvel]).T
    plot_2d(results, ['kd value', 'kp value', 'max vx'], n_data=rg, title='test tab')



def study_PD_meanvel(rg=6, incr=2):
    meanvel = np.zeros(rg**2)
    kp_tab = np.zeros(rg**2)
    kd_tab = np.zeros(rg**2)
    cmpt = 0
    for i in range(rg):
        for j in range(rg):

            print(f"iteration {cmpt+1}")
            kp_in = np.round(300 + i*incr*10,2)
            kd_in = np.round(2 + j*incr/10,2)
            kp_tab[cmpt] = (kp_in)
            kd_tab[cmpt] = (kd_in)
            kp = np.array([kp_in,kp_in,kp_in])   
            kd = np.array([kd_in, kd_in, kd_in])
            hyperparam = Hyperparameters(kp=kp, kd=kd)

            print(f"testing with kd_in = {kd_in}, kp_in = {kp_in}")
            val = np.array(run_cpg(hyp=hyperparam, do_plot=False, return_wanted="robot_vel"))
            # print(f"val = {type(val)} : {np.max(val[:,0])}")
            meanvel[cmpt] = np.mean(val[:,0])

            cmpt += 1
    results = np.array([kd_tab, kp_tab, meanvel]).T
    plot_2d(results, ['kd value', 'kp value', 'mean vx'], n_data=rg, title='test tab')

def study_PD_zvel(rg=20, incr=2):
    meanvelz = np.zeros(rg**2)
    kp_tab = np.zeros(rg**2)
    kd_tab = np.zeros(rg**2)
    cmpt = 0
    for i in range(rg):
        for j in range(rg):

            print(f"iteration {cmpt+1}")
            kp_in = np.round(200 + i*incr*10,2)
            kd_in = np.round(0.2 + j*incr/10,2)
            kp_tab[cmpt] = (kp_in)
            kd_tab[cmpt] = (kd_in)
            kp = np.array([kp_in,kp_in,kp_in])   
            kd = np.array([kd_in, kd_in, kd_in])
            hyperparam = Hyperparameters(kp=kp, kd=kd)

            print(f"testing with kd_in = {kd_in}, kp_in = {kp_in}")
            val = np.array(run_cpg(hyp=hyperparam, do_plot=False, return_wanted="vel"))
            # print(f"val = {type(val)} : {np.max(val[:,0])}")
            meanvelz[cmpt] = np.mean(val[:,2])

            cmpt += 1
    results = np.array([kd_tab, kp_tab, meanvelz]).T
    plot_2d(results, ['kd value', 'kp value', 'mean vz'], n_data=rg, title='test tab')


def plot_2d(results, labels, n_data=300, title='', log=False, cmap='nipy_spectral'):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    plt.title(title)

    plt.show()

def run_1():
    kp = 385
    kd = 2.6
    kp_cart = 260
    kd_cart = 15
    
    # hyperparameters 
    hyp = Hyperparameters(kp=np.array([kp,kp,kp]), kd=np.array([kd,kd,kd]), kp_cart=np.diag([kp_cart]*3), kd_cart=np.diag([kd_cart]*3))

    # run simulation
    average_speed, average_angular_speed, average_stride_length, average_stride_frequency, average_angular_stride_frequency, average_angular_stride_length, average_duty_factor, average_angular_duty_factor, average_cost_of_transport = run_cpg(hyp=hyp, do_plot=False, return_wanted="3.14")
    print("\n")
    print("average_speed", average_speed)
    print("average_angular_speed", average_angular_speed)
    print("average_stride_length", average_stride_length)
    print("average_stride_frequency", average_stride_frequency)
    print("average_angular_stride_frequency", average_angular_stride_frequency)
    print("average_angular_stride_length", average_angular_stride_length)
    print("average_duty_factor", average_duty_factor)
    print("average_angular_duty_factor", average_angular_duty_factor)
    print("average_cost_of_transport", average_cost_of_transport)


if __name__ == '__main__':
    run_1()
    # study_PD_maxvel()
    # study_PD_meanvel()
    # study_PD_zvel()