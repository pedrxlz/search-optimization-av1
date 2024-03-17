
import numpy as np


def local_random_search(objective_function, initial_sigma, maximum_iterations, domain_lower_bounds, domain_upper_bounds, type = "minimize"):
  """
  Implementa o algoritmo de Busca Aleatória Local (LRS) com opção de 
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
      type: Uma string que especifica o type de busca: "minimize" ou "maximize".

  Returns:
      Uma tupla que contém a melhor solução encontrada (xbest) e seu valor 
      fitness (fbest).
  """

  if type not in ["minimize", "maximize"]:
    raise ValueError("Tipo inválido: " + type)

  # Inicializa a melhor solução
  xbest = np.random.uniform(domain_lower_bounds[0], domain_upper_bounds[0]), \
          np.random.uniform(domain_lower_bounds[1], domain_upper_bounds[1])
  fbest = objective_function(xbest[0], xbest[1])

  # Iterações
  for _ in range(maximum_iterations):
    # Gera solução candidata com ruído gaussiano (opcional)
    if initial_sigma > 0:
      x = np.random.normal(xbest, initial_sigma, size=2)
      # Limita a solução candidata aos limites do domínio
      x = np.clip(x, domain_lower_bounds, domain_upper_bounds)
    else:
      x = np.random.uniform(domain_lower_bounds[0], domain_upper_bounds[0]), \
          np.random.uniform(domain_lower_bounds[1], domain_upper_bounds[1])

    # Avalia a solução candidata
    fval = objective_function(x[0], x[1])

    # Atualiza a melhor solução
    if type == "maximize":
      if fval > fbest:
        xbest = x
        fbest = fval
    elif type == "minimize":
      if fval < fbest:
        xbest = x
        fbest = fval

  return xbest, fbest



    