import numpy as np


def gerarCanditadoVizinho(x, e):
    """
    Gera um candidato vizinho de x.
    """
    return np.random.uniform(low= x - e, high= x + e)

def hillClimbing(f, initial_sigma, max_iter, domain_lower_bounds, domain_upper_bounds, t = "minimize", max_vizinhos= 1000):
    i = 0
    xAtual, yAtual = np.random.uniform(domain_lower_bounds[0], domain_lower_bounds[1]), \
            np.random.uniform(domain_upper_bounds[0], domain_upper_bounds[1])
    fAtual = f(xAtual, yAtual)
    melhoria = True
    while melhoria and i < max_iter:
        melhoria = False
        for _ in range(max_vizinhos):
            x_vizinho = gerarCanditadoVizinho(xAtual, initial_sigma)
            y_vizinho = gerarCanditadoVizinho(yAtual, initial_sigma)
            f_vizinho = f(x_vizinho, y_vizinho)
            
            if t == "minimize" and f_vizinho < fAtual:
                xAtual = x_vizinho
                yAtual = y_vizinho
                fAtual = f_vizinho
                melhoria = True
                break
            
            elif t == "maximize" and f_vizinho > fAtual:
                xAtual = x_vizinho
                yAtual = y_vizinho
                fAtual = f_vizinho
                melhoria = True
                break
           
        i += 1

    return [xAtual, yAtual], fAtual