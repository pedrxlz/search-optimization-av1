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
    ax.scatter(xbest[0], xbest[1], fbest, c='r', marker='o')
    plt.tight_layout()
    plt.show()

    
    