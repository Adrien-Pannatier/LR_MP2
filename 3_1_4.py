# hyper parameter research
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from run_cpg import run_cpg, Hyperparameters
from matplotlib.colors import LogNorm



def study_PD_maxvel(rg=10, incr=2):
    maxvel = np.empty((rg,rg))
    kp_tab = np.empty(rg)
    kd_tab = np.empty(rg)
    print(maxvel)
    for i in range(rg):
        for j in range(rg):
            print(f"iteration {i+j}")
            kp_in = np.round(100 + i*incr*10,2)
            kd_in = np.round(1 + j*incr/10,2)
            kp_tab.append(kp_in)
            kd_tab.append(kd_in)
            kp = np.array([kp_in,kp_in,kp_in])   
            kd = np.array([kd_in, kd_in, kd_in])
            hyperparam = Hyperparameters(kp=kp, kd=kd)

            print(f"testing with kd_in = {kd_in}, kp_in = {kp_in}")
            val = np.array(run_cpg(hyp=hyperparam, do_plot=False, return_wanted="maxvel"))
            # print(f"val = {type(val)} : {np.max(val[:,0])}")
            maxvel[i,j] = np.max(val[:,0])
    plot_2d(np.array([kd_tab,kp_tab,maxvel]).T, ['kd value', 'kp value', 'max vx'], n_data=rg**2, title='test tab')

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

if __name__ == '__main__':
    study_PD_maxvel()