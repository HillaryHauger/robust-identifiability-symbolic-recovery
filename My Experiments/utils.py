import matplotlib.pyplot as plt

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

def compute_distance_svs(u,dx,fd_orders=range(2,8)):
    distance_svs = []
    for fd_order in fd_orders:
        ux = ps.FiniteDifference(order=fd_order, d=1, axis=0,
                             drop_endpoints=False)._differentiate(u, dx)
    
        u_flat, u_x_flat = u.flatten(), ux.flatten()
        features = np.concatenate([u_flat.reshape(len(u_flat),-1), u_x_flat.reshape(len(u_flat),-1)], axis=1).T 
        svs = svd(features, compute_uv=False)
        distance_svs.append(svs[-1])
    return distance_svs