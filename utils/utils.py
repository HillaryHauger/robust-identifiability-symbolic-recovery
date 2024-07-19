import matplotlib.pyplot as plt
import numpy as np

def infinity_norm(x):
    return np.max(np.abs(x))

def plot_result_jacobi(svs,t_grid,x_grid,target_noise=0,
                       titlesize=18,subtitle_size=16,labelsize=16,tickssize=16,fd_orders=[2,7]):
    # Plot derivative results
    first_order=fd_orders[0]
    last_order=fd_orders[1]
    fig, axes = plt.subplots(2,1)
    #plt.title(r'Smallest singular value of the Jacobian', fontsize=titlesize, fontname=fontname)
    c = axes[0].pcolor(t_grid, x_grid, svs[0])
    axes[0].set_title(f'{first_order}nd order finite differences: noise {target_noise}', fontsize=subtitle_size)
    axes[0].set_ylabel('x', fontsize=tickssize)
    axes[0].set_xticks([])
    fig.colorbar(c, ax=axes[0])
    c = axes[1].pcolor(t_grid, x_grid, svs[1])
    axes[1].set_title(f'{last_order}th order finite differences', fontsize=subtitle_size)
    axes[1].set_xlabel('t', fontsize=tickssize)
    axes[1].set_ylabel('x', fontsize=tickssize)
    fig.colorbar(c, ax=axes[1])
    plt.tight_layout()

