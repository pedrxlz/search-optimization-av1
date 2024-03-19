import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt


def plot(xbest, fbest, objective_function, dom1, dom2):
    x = np.linspace(dom1[0], dom1[1], 100)
    y = np.linspace(dom2[0], dom2[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', rstride=10, cstride=10, alpha=0.6)
    
    colors = cm.rainbow(np.linspace(0, 1, len(xbest)))
    
    for i in range(len(xbest)):
        ax.scatter(xbest[i][0], xbest[i][1], fbest[i], color=colors[i], marker='o')
    plt.tight_layout()
    plt.show()

    
    