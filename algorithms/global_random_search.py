import numpy as np


def global_random_search(objective_function, initial_sigma, maximum_iterations, domain_lower_bounds, domain_upper_bounds, t = "minimize"):
    """
    Implementa o algoritmo de Busca Aleatória Global (GRS) com opção de
    minimização ou maximização.
    
    Args:
        objective_function: Uma função que recebe dois argumentos (x1, x2) e
                            retorna um valor escalar fitness.
        initial_sigma: Um float representando o tamanho inicial do passo para
                        gerar soluções candidatas.
        maximum_iterations: Um inteiro que especifica o número máximo de iterações.
        domain_lower_bounds: Uma tupla (lower_bound1, lower_bound2) que define os
                            limites inferiores do domínio de busca.
        domain_upper_bounds: Uma tupla (upper_bound1, upper_bound2) que define os
                            limites superiores do domínio de busca.
        t: Uma string que especifica o t de busca: "minimize" ou "maximize".
        
    Returns:
        Uma tupla que contém a melhor solução encontrada (xbest) e seu valor
        fitness (fbest).
    """
    
    xbest = [np.random.uniform(domain_lower_bounds[0], domain_lower_bounds[1]), \
            np.random.uniform(domain_upper_bounds[0], domain_upper_bounds[1])]
    fbest = objective_function(xbest[0], xbest[1])
    
    for _ in range(maximum_iterations):
        x_cand = [np.random.uniform(domain_lower_bounds[0], domain_lower_bounds[1]), \
            np.random.uniform(domain_upper_bounds[0], domain_upper_bounds[1])]
        f_cand = objective_function(x_cand[0], x_cand[1])
        
        if t == "maximize":
            if f_cand > fbest:
                xbest = x_cand
                fbest = f_cand
        elif t == "minimize":
            if f_cand < fbest:
                xbest = x_cand
                fbest = f_cand

    return xbest, fbest