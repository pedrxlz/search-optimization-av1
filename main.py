
import csv

import numpy as np

from algorithms.global_random_search import global_random_search
from algorithms.local_random_search import local_random_search
from plot import plot


# Encontrar Minimo
def f1(x1, x2):
    return x1**2 + x2**2

f1Dom = [(-100, 100), (-100, 100)]

# Encontrar máximo
def f2(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-(x1-1.7)**2 + (x2-1.7)**2)

f2Dom = [(-2, 4), (-2, 5)]

# Encontrar mínimo
def f3(x1 , x2):
    return x1 ** 2 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

f3Dom = [(-2, 2), (-2, 2)]

# Encontrar mínimo
def f4(x1, x2):
    return (x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10)

f4Dom = [(-5.12, 5.12), (-5.12, 5.12)]

# Encontrar máximo
def f5(x1, x2):
    return (x1 * np.cos(x1)) / 20 + 2 * np.exp(-x1 ** 2 - (x2 - 1) ** 2) + 0.01 * x1 * x2

f5Dom = [(-10, 10), (-10, 10)]

# Encontrar máximo
def f6(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

f6Dom = [(-1, 3), (-1, 3)]

# Encontrar mínimo
def f7(x1, x2):
    return - np.sin(x1) * (np.sin(x1 ** 2 / np.pi)) ** 2 * 10 - np.sin(x2) * (np.sin(2 * x2 ** 2 / np.pi)) ** 2 * 10

f7Dom = [(0, np.pi), (0, np.pi)]

# Encontrar mínimo
def f8(x1, x2):
    return - (x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

f8Dom = [(-200, 20), (-200, 20)]

args = {
    "name": "f5",
    "objective_function": f5,
    "type": "maximize", 
    "sigma": .5,
    "max_iter": 1000,
    "domain": f5Dom,
    "num_executions": 100,
    "plot": False
}

header = ["Function", "Execution", "xbest", "fbest"]
algorithms = [
    {
        "name": "local_random_search",
        "function": local_random_search
    },
    {
        "name": "global_random_search",
        "function": global_random_search
    },
    # {
    #     "name": "hill_climbing",
    #     "function": hill_climbing
    # }
]

for algorithm in algorithms:
    with open(f"{algorithm["name"]}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i in range(args['num_executions']):
            xbest, fbest = algorithm["function"](args['objective_function'], args['sigma'], args['max_iter'], args['domain'][0], args['domain'][1], type=args["type"])
            
            row = [args['name'], i+1, xbest, fbest]
            writer.writerow(row)

            print(f"Execution {i+1}:")
            print(f'xbest: ', xbest)
            print(f'fbest: ', fbest)

            if args['plot']:
                plot(xbest, fbest, args['objective_function'], args['domain'][0], args['domain'][1])
        




